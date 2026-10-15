#include <Wire.h>
#include <Protocentral_FDC1004.h>
#include "FS.h"
#include "SD_MMC.h"

// ================= USER CONFIGURATION =================
#define I2C_SDA_PIN        2
#define I2C_SCL_PIN        1

// Onboard SD_MMC 1-Bit Mode Pin Assignments
#define SD_MMC_CLK         39
#define SD_MMC_CMD         38
#define SD_MMC_D0          40

// Logging Configurations (Easily Adjustable)
#define RECORD_INTERVAL_MS    50                  // Measurement time interval in milliseconds
#define BUFFER_SIZE           200                   // Number of samples stored in RAM before writing to SD
#define FILE_NAME             "/fdc1004_data_pre.csv"  // Destination log file path on SD card
#define AUTO_CALIB_INTERVAL_MS 300000               // Every 5 minutes perform calibration on active channels
// ======================================================

#define SAMPLE_RATE    FDC1004_RATE_100HZ
#define DEFAULT_CAPDAC 0

const uint8_t ACTIVE_CHANNELS[2] = {1, 3};  // Only channels 2 and 4 are measured/saved

FDC1004 fdc;

// Csatornánként külön tároljuk a CAPDAC értékeket (0..31)
uint8_t capdac_vals[4] = {DEFAULT_CAPDAC, DEFAULT_CAPDAC, DEFAULT_CAPDAC, DEFAULT_CAPDAC};

const fdc1004_measurement_t measurements[4] = {
  FDC1004_MEASUREMENT_1,
  FDC1004_MEASUREMENT_2,
  FDC1004_MEASUREMENT_3,
  FDC1004_MEASUREMENT_4
};

// Data structure to buffer individual sample sets
struct MeasurementSample {
  uint32_t timestampMs;
  float cap_pf[4];
};

MeasurementSample dataBuffer[BUFFER_SIZE];
uint16_t bufferIndex = 0;
uint32_t lastRecordTime = 0;
uint32_t lastAutoCalibTime = 0;
bool sdPresent = false;

// Segédfüggvény egyetlen csatorna kiolvasására (delta_c és teljes kapacitás)
bool readChannelDelta(uint8_t ch_idx, float &delta_c) {
  fdc.triggerSingleMeasurement(measurements[ch_idx], SAMPLE_RATE);
  delay(15);

  uint16_t value[2];
  if (!fdc.readMeasurement(measurements[ch_idx], value)) {
    int32_t raw_value = ((int32_t)value[0] << 8) | ((value[1] >> 8) & 0xFF);
    if (raw_value & 0x00800000) {
      raw_value |= 0xFF000000;
    }
    delta_c = (float)raw_value / 524288.0f; // -16 pF .. +16 pF
    return true;
  }
  return false;
}

// Writes filled RAM buffer to SD Card in CSV format
void writeBufferToSD() {
  if (!sdPresent) {
    bufferIndex = 0;
    return;
  }

  File logFile = SD_MMC.open(FILE_NAME, FILE_APPEND);
  if (!logFile) {
    Serial.println("[SD ERROR] Failed to open file for writing!");
    sdPresent = false;
    bufferIndex = 0;
    return;
  }

  for (uint16_t i = 0; i < bufferIndex; i++) {
    logFile.print(dataBuffer[i].timestampMs);
    for (uint8_t idx = 0; idx < 2; idx++) {
      uint8_t ch = ACTIVE_CHANNELS[idx];
      logFile.print(",");
      if (isnan(dataBuffer[i].cap_pf[ch])) {
        logFile.print("NAN");
      } else {
        logFile.print(dataBuffer[i].cap_pf[ch], 4);
      }
    }
    logFile.println();
  }

  logFile.close();
  Serial.printf("[SD LOG] Flushed %d records to SD card.\n", bufferIndex);
  bufferIndex = 0;
}

// Munkapont-kereső kalibráló algoritmus
void calibrateChannel(uint8_t ch_idx) {
  if (ch_idx >= 4) {
    Serial.println("[CALIB HIBA] Ervenytelen csatornaszam! (1-4 lehetseges)");
    return;
  }

  Serial.printf("\n>>> Kalibracio inditasa a(z) %d. csatornan (CIN%d)... <<<\n", ch_idx + 1, ch_idx + 1);

  uint8_t best_capdac = 0;
  float min_diff = 999.0f;
  float best_delta = 0.0f; // TODO: visszaallitani -16ra
  const float TARGET_DELTA = -10.0f; // Cel ertek: -16 pF

  for (uint8_t test_dac = 0; test_dac <= FDC1004_CAPDAC_MAX; test_dac++) {
    fdc.configureMeasurementSingle(measurements[ch_idx], ch_idx, test_dac);
    delay(10);

    float delta_c = 0.0f;
    if (readChannelDelta(ch_idx, delta_c)) {
      float diff = fabs(delta_c - TARGET_DELTA);
      Serial.printf("  Teszt CAPDAC = %2d -> Delta = %6.2f pF (elteres a -16 pF-tol: %5.2f pF)\n", 
                    test_dac, delta_c, diff);

      if (diff < min_diff) {
        min_diff = diff;
        best_capdac = test_dac;
        best_delta = delta_c;
      }
    }
  }

  capdac_vals[ch_idx] = best_capdac;
  fdc.configureMeasurementSingle(measurements[ch_idx], ch_idx, best_capdac);

  Serial.printf(">>> Kalibracio KESZ! Optimalis CAPDAC: %d (Mert delta: %.2f pF, Offset: %.2f pF) <<<\n\n",
                best_capdac, best_delta, best_capdac * 3.125f);
}

void autoCalibrateActiveChannels() {
  Serial.println("[AUTO CALIB] 5 perc eltelt, aktiv csatornák kalibrálása...");
  for (uint8_t idx = 0; idx < 2; idx++) {
    calibrateChannel(ACTIVE_CHANNELS[idx]);
  }
}

// Serial parancsok figyelése és feldolgozása
void handleSerialCommands() {
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();

    if (cmd.startsWith("calib")) {
      int channel = 0;
      
      if (cmd.length() > 5) {
        String numStr = cmd.substring(5);
        numStr.trim();
        channel = numStr.toInt();
      } else {
        Serial.println("\n[CALIB] Melyik csatornat szeretned kalibralni? (Add meg a szamot: 1, 2, 3 vagy 4):");
        while (!Serial.available()) {
          delay(10);
        }
        channel = Serial.parseInt();
      }

      if (channel == 2 || channel == 4) {
        calibrateChannel(channel - 1);
      } else {
        Serial.println("[CALIB HIBA] Csak a mért csatornák kalibrálhatók: 2 és 4!");
      }
    }
  }
}

void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial.println("\n--- FDC1004 SD_MMC Datalogger (ESP32-S3) ---");
  Serial.println("Tipp: Ird be a Serial Monitorba: 'calib <1-4>' a munkapont kalibralasahoz!");

  Wire.begin(I2C_SDA_PIN, I2C_SCL_PIN);

  for (uint8_t i = 0; i < 4; i++) {
    fdc.configureMeasurementSingle(measurements[i], i, capdac_vals[i]);
  }

  // Initial calibration on active channels before measurement starts
  lastAutoCalibTime = millis();
  autoCalibrateActiveChannels();

  // SD MMC Initialization (1-bit bus mode)
  SD_MMC.setPins(SD_MMC_CLK, SD_MMC_CMD, SD_MMC_D0);
  if (!SD_MMC.begin("/sdcard", true)) { // true = 1-bit mode
    Serial.println("[SD WARNING] SD Card (SD_MMC) initialization failed! Logging disabled.");
    sdPresent = false;
  } else {
    Serial.println("[SD] SD Card (SD_MMC) initialized successfully.");
    sdPresent = true;

    // Header / session marker creation
    if (!SD_MMC.exists(FILE_NAME)) {
      File logFile = SD_MMC.open(FILE_NAME, FILE_WRITE);
      if (logFile) {
        logFile.println("timestamp_ms,CIN2_pF,CIN4_pF");
        logFile.close();
      }
    } else {
      File logFile = SD_MMC.open(FILE_NAME, FILE_APPEND);
      if (logFile) {
        logFile.println("=== APPEND SESSION START ===");
        logFile.close();
      }
    }
  }
}

void loop() {
  handleSerialCommands();

  uint32_t currentMillis = millis();

  if (currentMillis - lastAutoCalibTime >= AUTO_CALIB_INTERVAL_MS) {
    lastAutoCalibTime = currentMillis;
    autoCalibrateActiveChannels();
  }

  // Periodical sampling check
  if (currentMillis - lastRecordTime >= RECORD_INTERVAL_MS) {
    lastRecordTime = currentMillis;

    MeasurementSample sample;
    sample.timestampMs = currentMillis;

    for (uint8_t ch = 0; ch < 4; ch++) {
      sample.cap_pf[ch] = NAN;
    }

    for (uint8_t idx = 0; idx < 2; idx++) {
      uint8_t ch = ACTIVE_CHANNELS[idx];
      float delta_c = 0.0f;
      if (readChannelDelta(ch, delta_c)) {
        float current_offset = capdac_vals[ch] * 3.125f;
        sample.cap_pf[ch] = delta_c + current_offset;
      }
    }

    // Console output
    Serial.printf("[%lu ms] C2: %.2f | C4: %.2f\n",
                  sample.timestampMs,
                  sample.cap_pf[1], sample.cap_pf[3]);

    // Push into buffer
    dataBuffer[bufferIndex++] = sample;

    // Write to SD when buffer fills
    if (bufferIndex >= BUFFER_SIZE) {
      writeBufferToSD();
    }
  }
}