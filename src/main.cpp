#include <Wire.h>
#include <Protocentral_FDC1004.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>

#define I2C_SDA_PIN 41
#define I2C_SCL_PIN 40

#define SAMPLE_RATE   FDC1004_RATE_100HZ
#define DEFAULT_CAPDAC 0

// Teljes szabványos Nordic UART Service (NUS)
#define SERVICE_UUID           "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
#define CHARACTERISTIC_UUID_RX "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"
#define CHARACTERISTIC_UUID_TX "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"

FDC1004 fdc;

BLEServer *pServer = NULL;
BLECharacteristic *pTxCharacteristic = NULL;
bool deviceConnected = false;
bool oldDeviceConnected = false;

// Csatornánként külön tároljuk a CAPDAC értékeket (0..31)
uint8_t capdac_vals[4] = {DEFAULT_CAPDAC, DEFAULT_CAPDAC, DEFAULT_CAPDAC, DEFAULT_CAPDAC};

class ServerCallbacks : public BLEServerCallbacks {
  void onConnect(BLEServer* pServer) {
    deviceConnected = true;
    Serial.println("[BLE] Csatlakozva!");
  }
  void onDisconnect(BLEServer* pServer) {
    deviceConnected = false;
    Serial.println("[BLE] Lecsatlakozva!");
  }
};

const fdc1004_measurement_t measurements[4] = {
  FDC1004_MEASUREMENT_1,
  FDC1004_MEASUREMENT_2,
  FDC1004_MEASUREMENT_3,
  FDC1004_MEASUREMENT_4
};

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

// Munkapont-kereső kalibráló algoritmus
void calibrateChannel(uint8_t ch_idx) {
  if (ch_idx >= 4) {
    Serial.println("[CALIB HIBA] Ervenytelen csatornaszam! (1-4 lehetseges)");
    return;
  }

  Serial.printf("\n>>> Kalibracio inditasa a(z) %d. csatornan (CIN%d)... <<<\n", ch_idx + 1, ch_idx + 1);

  uint8_t best_capdac = 0;
  float min_diff = 999.0f;
  float best_delta = 0.0f;
  const float TARGET_DELTA = -16.0f; // Cel ertek: -16 pF

  // Végigpásztázzuk a lehetséges CAPDAC értékeket (0-tól 31-ig)
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

  // A megtalált legjobb CAPDAC elmentése és beállítása
  capdac_vals[ch_idx] = best_capdac;
  fdc.configureMeasurementSingle(measurements[ch_idx], ch_idx, best_capdac);

  Serial.printf(">>> Kalibracio KESZ! Optimalis CAPDAC: %d (Mert delta: %.2f pF, Offset: %.2f pF) <<<\n\n",
                best_capdac, best_delta, best_capdac * 3.125f);
}

// Serial parancsok figyelése és feldolgozása
void handleSerialCommands() {
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();

    if (cmd.startsWith("calib")) {
      int channel = 0;
      
      // Ha a parancsban már benne volt a szám (pl. "calib 1")
      if (cmd.length() > 5) {
        String numStr = cmd.substring(5);
        numStr.trim();
        channel = numStr.toInt();
      } else {
        // Ha csak simán "calib"-ot kapott, megvárja a számot
        Serial.println("\n[CALIB] Melyik csatornat szeretned kalibralni? (Add meg a szamot: 1, 2, 3 vagy 4):");
        while (!Serial.available()) {
          delay(10);
        }
        channel = Serial.parseInt();
      }

      if (channel >= 1 && channel <= 4) {
        calibrateChannel(channel - 1);
      } else {
        Serial.println("[CALIB HIBA] A csatorna szama 1 es 4 kozott kell legyen!");
      }
    }
  }
}

void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial.println("\n--- FDC1004 BLE & Kalibracio (ESP32-S3) ---");
  Serial.println("Tipp: Ird be a Serial Monitorba: 'calib <1-4>' a munkapont kalibralasahoz!");

  Wire.begin(I2C_SDA_PIN, I2C_SCL_PIN);

  for (uint8_t i = 0; i < 4; i++) {
    fdc.configureMeasurementSingle(measurements[i], i, capdac_vals[i]);
  }

  // BLE Inicializálás
  BLEDevice::init("ESP32-S3-FDC1004");
  pServer = BLEDevice::createServer();
  pServer->setCallbacks(new ServerCallbacks());

  BLEService *pService = pServer->createService(SERVICE_UUID);

  pTxCharacteristic = pService->createCharacteristic(
                        CHARACTERISTIC_UUID_TX,
                        BLECharacteristic::PROPERTY_NOTIFY | 
                        BLECharacteristic::PROPERTY_READ
                      );
  pTxCharacteristic->addDescriptor(new BLE2902());

  BLECharacteristic *pRxCharacteristic = pService->createCharacteristic(
                                           CHARACTERISTIC_UUID_RX,
                                           BLECharacteristic::PROPERTY_WRITE
                                         );

  pService->start();

  BLEAdvertising *pAdvertising = BLEDevice::getAdvertising();
  pAdvertising->addServiceUUID(SERVICE_UUID);
  pAdvertising->setScanResponse(true);
  pAdvertising->setMinPreferred(0x06);
  pAdvertising->setMinPreferred(0x12);
  BLEDevice::startAdvertising();

  Serial.println("[BLE] Várakozás kapcsolatra...");
}

void loop() {
  // Serial port figyelése parancsokra
  handleSerialCommands();

  float cap_values[4];

  for (uint8_t i = 0; i < 4; i++) {
    float delta_c = 0.0f;
    if (readChannelDelta(i, delta_c)) {
      float current_offset = capdac_vals[i] * 3.125f;
      cap_values[i] = delta_c + current_offset;
    } else {
      cap_values[i] = NAN;
    }
  }

  char outputBuffer[80];
  snprintf(outputBuffer, sizeof(outputBuffer), "C1: %.2f | C2: %.2f | C3: %.2f | C4: %.2f\r\n",
           cap_values[0], cap_values[1], cap_values[2], cap_values[3]);

  Serial.print(outputBuffer);

  if (deviceConnected && pTxCharacteristic != NULL) {
    pTxCharacteristic->setValue((uint8_t*)outputBuffer, strlen(outputBuffer));
    pTxCharacteristic->notify();
  }

  // Újrahirdetés kapcsolatbontáskor
  if (!deviceConnected && oldDeviceConnected) {
    delay(500);
    pServer->startAdvertising();
    oldDeviceConnected = deviceConnected;
  }
  if (deviceConnected && !oldDeviceConnected) {
    oldDeviceConnected = deviceConnected;
  }

  delay(200);
}