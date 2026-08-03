/**
 * @file RC_measure.ino
 * @brief High-precision capacitance meter firmware for ESP32-C3 SuperMini
 * @hardware Charge Pin: GPIO2, Sense/ADC Pin: GPIO3, Resistor Sense Pin: GPIO1
 * Buttons: MENU (GPIO21), VALUE (GPIO10), OK (GPIO20)
 * Display: 128x64 I2C OLED (U8g2 Library)
 * EEPROM: 24C02WP on I2C bus (Address 0x50)
 */

#include <Wire.h>
#include <U8g2lib.h>
#include <algorithm> // Required for std::sort
#include <cmath>
#include "hal/gpio_ll.h"
#include "hal/gpio_hal.h"
#include "soc/gpio_struct.h"

// ----------------------------------------------------------------------------
// Serial Logging Level Configuration
// ----------------------------------------------------------------------------
enum SerialOutputLevel : uint8_t {
    SERIAL_ALL_VERBOSE     = 0, // Detailed log + EEPROM + Measurements
    SERIAL_ALL_VALUES_ONLY = 1, // CSV values: f, time_us, cap_pF, res_ohm
    SERIAL_EEPROM_ONLY     = 2  // EEPROM status and hex dumps only
};

SerialOutputLevel serialLogLevel = SERIAL_ALL_VALUES_ONLY;

// ----------------------------------------------------------------------------
// Constants & Configuration
// ----------------------------------------------------------------------------
const uint16_t SCREEN_WIDTH  = 128;
const uint16_t SCREEN_HEIGHT = 64;

// 24C02WP I2C EEPROM Address
const uint8_t EEPROM_ADDR = 0x50;

// Pin Definitions
const int CHARGE_PIN = 2;
const int SENSE_PIN  = 3;
const int RES_PIN    = 1;  // GPIO1: Resistor select (INPUT_PULLUP)
const int BTN_MENU   = 21;
const int BTN_VALUE  = 10;
const int BTN_OK     = 20;

// Resistor Values (ohms)
const float R_LOW  = 100000.0f;   // 100 kΩ / 1 MΩ
const float R_HIGH = 10000000.0f; // 10 MΩ

// Source Default Calibration Constants
const float DEFAULT_K                  = -logf(1.0f - 0.632f);//0.596f);
const float DEFAULT_CALIBRATION_FACTOR = 1000.0f;     // Scale factor adjustment
const float DEFAULT_CALIBRATION_OFFSET = 0.0f;//-88130.0f;   // Offset adjustment in pF

// Mutable Calibration Constants (Runtime)
float K                  = DEFAULT_K;
float CALIBRATION_FACTOR = DEFAULT_CALIBRATION_FACTOR;
float CALIBRATION_OFFSET = DEFAULT_CALIBRATION_OFFSET;

const float ARRAY_LOWER_PERC = 0.19f; // Lower quartile threshold
const float ARRAY_UPPER_PERC = 0.81f; // Upper quartile threshold

const uint64_t CPU_KHZ = 160000UL;
const uint64_t TIMEOUT_CYC = 50000 * CPU_KHZ;
const uint16_t DEBOUNCE_MS = 20;

const int MAX_SAMPLES = 5000;

// ----------------------------------------------------------------------------
// EEPROM & Hash Structures
// ----------------------------------------------------------------------------
struct DeviceSettings {
    float k;
    float factor;
    float offset;
    uint8_t serialLogLevel;
    uint8_t sampleIndex;
};

/**
 * @brief Computes 32-bit FNV-1a Hash over a byte array for integrity verification.
 */
uint32_t calculateHash(const uint8_t* data, size_t len) {
    uint32_t hash = 2166136261UL; // FNV-1a offset basis
    for (size_t i = 0; i < len; i++) {
        hash ^= data[i];
        hash *= 16777619UL;      // FNV-1a prime
    }
    return hash;
}

/**
 * @brief Writes bytes to 24C02WP EEPROM with 5ms write cycle delay per byte.
 */
void writeEEPROM(uint8_t memAddr, const uint8_t* data, size_t len) {
    for (size_t i = 0; i < len; i++) {
        Wire.beginTransmission(EEPROM_ADDR);
        Wire.write((uint8_t)(memAddr + i));
        Wire.write(data[i]);
        Wire.endTransmission();
        delay(5); // 24C02 EEPROM write cycle delay (tWR = 5ms)
    }
}

/**
 * @brief Reads bytes from 24C02WP EEPROM.
 */
bool readEEPROM(uint8_t memAddr, uint8_t* data, size_t len) {
    Wire.beginTransmission(EEPROM_ADDR);
    Wire.write(memAddr);
    if (Wire.endTransmission() != 0) return false;
    uint8_t bytesRead = Wire.requestFrom((uint8_t)EEPROM_ADDR, (uint8_t)len);
    if (bytesRead < len) return false;
    for (size_t i = 0; i < len; i++) {
        if (Wire.available()) {
            data[i] = Wire.read();
        } else {
            return false;
        }
    }
    return true;
}

/**
 * @brief Dumps EEPROM byte contents formatted as hexadecimal over Serial.
 */
void dumpEEPROM() {
    if (serialLogLevel != SERIAL_ALL_VERBOSE && serialLogLevel != SERIAL_EEPROM_ONLY) return;

    Serial.println("\n========== 24C02 EEPROM HEX DUMP ==========");
    uint8_t buffer[32];
    if (readEEPROM(0, buffer, sizeof(buffer))) {
        for (size_t i = 0; i < sizeof(buffer); i++) {
            if (i % 16 == 0) {
                Serial.printf("\n0x%02X: ", (unsigned int)i);
            }
            Serial.printf("%02X ", buffer[i]);
        }
        Serial.println("\n===========================================\n");
    } else {
        Serial.println("Error: Failed to read EEPROM for dump.");
    }
}

// Global Menu & Settings declaration (needed for EEPROM save/load functions)
const int SAMPLE_COUNT = 4;
int sampleIndex = 1; // Default: 100 samples

/**
 * @brief Saves calibration parameters and user settings (serial level & sample count) to EEPROM with Hash.
 */
bool saveSettingsToEEPROM() {
    DeviceSettings settings;
    settings.k = K;
    settings.factor = CALIBRATION_FACTOR;
    settings.offset = CALIBRATION_OFFSET;
    settings.serialLogLevel = static_cast<uint8_t>(serialLogLevel);
    settings.sampleIndex = static_cast<uint8_t>(sampleIndex);

    uint32_t hash = calculateHash((const uint8_t*)&settings, sizeof(settings));
    // Save settings data starting at byte 0
    writeEEPROM(0, (const uint8_t*)&settings, sizeof(settings));
    // Save hash immediately after settings data
    writeEEPROM(sizeof(settings), (const uint8_t*)&hash, sizeof(hash));

    if (serialLogLevel == SERIAL_ALL_VERBOSE || serialLogLevel == SERIAL_EEPROM_ONLY) {
        Serial.println("Calibration & Settings saved to 24C02 EEPROM with Hash!");
    }
    return true;
}

/**
 * @brief Loads calibration parameters and user settings from 24C02 EEPROM if hash matches.
 */
bool loadSettingsFromEEPROM() {
    Wire.beginTransmission(EEPROM_ADDR);
    if (Wire.endTransmission() != 0) {
        if (serialLogLevel == SERIAL_ALL_VERBOSE || serialLogLevel == SERIAL_EEPROM_ONLY) {
            Serial.println("EEPROM (0x50) not detected on I2C bus.");
        }
        return false;
    }

    DeviceSettings settings;
    uint32_t storedHash = 0;

    if (!readEEPROM(0, (uint8_t*)&settings, sizeof(settings))) return false;
    if (!readEEPROM(sizeof(settings), (uint8_t*)&storedHash, sizeof(storedHash))) return false;

    uint32_t computedHash = calculateHash((const uint8_t*)&settings, sizeof(settings));
    if (computedHash == storedHash) {
        K = settings.k;
        CALIBRATION_FACTOR = settings.factor;
        CALIBRATION_OFFSET = settings.offset;

        if (settings.serialLogLevel < 3) {
            serialLogLevel = static_cast<SerialOutputLevel>(settings.serialLogLevel);
        }
        if (settings.sampleIndex < SAMPLE_COUNT) {
            sampleIndex = settings.sampleIndex;
        }

        if (serialLogLevel == SERIAL_ALL_VERBOSE || serialLogLevel == SERIAL_EEPROM_ONLY) {
            Serial.println("Valid EEPROM configuration & settings loaded successfully!");
        }
        return true;
    } else {
        if (serialLogLevel == SERIAL_ALL_VERBOSE || serialLogLevel == SERIAL_EEPROM_ONLY) {
            Serial.println("EEPROM hash mismatch. Keeping default configuration.");
        }
        return false;
    }
}

/**
 * @brief Resets calibration variables, serial verbosity, and sample selection to default values.
 */
void resetSettingsToDefaults() {
    K                  = DEFAULT_K;
    CALIBRATION_FACTOR = DEFAULT_CALIBRATION_FACTOR;
    CALIBRATION_OFFSET = DEFAULT_CALIBRATION_OFFSET;
    serialLogLevel     = SERIAL_ALL_VERBOSE;
    sampleIndex        = 1; // Default to N=100
    if (serialLogLevel == SERIAL_ALL_VERBOSE || serialLogLevel == SERIAL_EEPROM_ONLY) {
        Serial.println("Calibration and settings reset to defaults.");
    }
}

/**
 * @brief Displays available Serial commands over the terminal.
 */
void printHelp() {
    Serial.println("\n--- Available Serial Commands ---");
    Serial.println(" m or menu   : Simulate MENU button press");
    Serial.println(" v or value  : Simulate VALUE button press");
    Serial.println(" o or ok     : Simulate OK button press");
    Serial.println(" s or serial : Cycle Serial level (Verbose -> Values Only CSV -> EEPROM Only)");
    Serial.println(" save        : Save calibration & settings (serial level, sample count) to EEPROM");
    Serial.println(" load        : Load calibration & settings from EEPROM");
    Serial.println(" reset       : Reset calibration & settings to default values");
    Serial.println(" h or help   : Print this help menu");
    Serial.println("----------------------------------\n");
}

// ----------------------------------------------------------------------------
// Enums & Menu Structures
// ----------------------------------------------------------------------------
enum MenuItem {
    MENU_TARE,
    MENU_SAMPLES,
    MENU_RUN,
    MENU_CALIB
};

enum RunMode {
    RUN_CONTINUOUS,
    RUN_STOP,
    RUN_ONCE
};

MenuItem currentMenu = MENU_TARE;

// --- Menu Configurations ---
const bool TARE_VALUES[]    = { true, false };
const char* TARE_LABELS[]   = { "ABS", "REL" };
const int TARE_COUNT        = 2;
int tareIndex               = 0;

const int SAMPLE_VALUES[]   = { 10, 100, 1000, 5000 };
const char* SAMPLE_LABELS[] = { "10", "100", "1000", "5000" };

const RunMode RUN_VALUES[]  = { RUN_CONTINUOUS, RUN_STOP, RUN_ONCE };
const char* RUN_LABELS[]    = { "RUN", "STOP", "ONCE" };
const int RUN_COUNT         = 3;
int runIndex                = 0;

const char* CALIB_LABELS[]  = { "Calibrate", "Reset Def", "Load EEPROM", "Save EEPROM" };
const int CALIB_COUNT       = 4;
int calibIndex              = 0;

// ----------------------------------------------------------------------------
// Calibration Mode State & Data Structures
// ----------------------------------------------------------------------------
bool inCalibMode = false;
int calibElement = 0;
int calibDigits[4] = {0, 0, 0, 0};
int calibUnitIndex = 0;
const char* CALIB_UNITS[] = { "pF", "nF", "uF" };
const float CALIB_UNIT_MULT[] = { 1.0f, 1000.0f, 1000000.0f };

const int MAX_CALIB_POINTS = 10;
float calibRawX[MAX_CALIB_POINTS];
float calibTargetY[MAX_CALIB_POINTS];
int calibPointCount = 0;
char calibStatusMsg[32] = "Set val & [MEAS]";

// ----------------------------------------------------------------------------
// Global Measurement State & Buffers
// ----------------------------------------------------------------------------
uint32_t sampleBuffer[MAX_SAMPLES];
uint32_t sampleCount = 0;
float avgTimeUs = 0.0f;
float capacitanceBase = 0.0f;
float capacitancePicofarads = 0.0f;
float displayCapacitance = 0.0f;
float tareValue = 0.0f;
bool measurementRequested = true;
unsigned long lastDisplayUpdate = 0;

// Display Setup (I2C Address 0x3C)
U8G2_SSD1306_128X64_NONAME_F_HW_I2C u8g2(U8G2_R0, U8X8_PIN_NONE);

// ----------------------------------------------------------------------------
// Helper Functions
// ----------------------------------------------------------------------------
static inline IRAM_ATTR bool fastRead() {
    return GPIO.in.val & (1UL << SENSE_PIN);
}

bool buttonPressed(int pin) {
    static bool lastReading[40] = {false};
    static bool stableState[40] = {false};
    static uint32_t lastChangeTime[40] = {0};

    bool reading = digitalRead(pin);

    if (reading != lastReading[pin]) {
        lastReading[pin] = reading;
        lastChangeTime[pin] = millis();
    }

    if ((millis() - lastChangeTime[pin]) >= DEBOUNCE_MS) {
        if (reading != stableState[pin]) {
            stableState[pin] = reading;
            if (stableState[pin] == LOW) { // INPUT_PULLUP: LOW = Pressed
                return true;
            }
        }
    }

    return false;
}

/**
 * @brief Filters noise and calculates average cycle count using Interquartile Range (IQR) outlier rejection.
 */
float computeFilteredAverageCycles(uint32_t* buffer, int count) {
    if (count <= 0) return 0.0f;
    if (count < 4) {
        uint64_t sum = 0;
        for (int i = 0; i < count; i++) sum += buffer[i];
        return (float)sum / count;
    }

    std::sort(buffer, buffer + count);
    
    float q1 = buffer[int(count * ARRAY_LOWER_PERC)];
    float q3 = buffer[int(count * ARRAY_UPPER_PERC)];
    float iqr = q3 - q1;
    
    float lowerBound = q1 - (1.5f * iqr);
    float upperBound = q3 + (1.5f * iqr);

    uint64_t validSum = 0;
    uint32_t validCount = 0;
    for (int i = 0; i < count; i++) {
        if (buffer[i] >= lowerBound && buffer[i] <= upperBound) {
            validSum += buffer[i];
            validCount++;
        }
    }

    if (validCount == 0) {
        return buffer[count / 2];
    }

    return (float)validSum / validCount;
}

/**
 * @brief Fits recorded etalon points using Least-Squares Linear Regression, then saves to EEPROM.
 */
void performCalibrationFitting() {
    if (calibPointCount == 0) return;
    if (calibPointCount == 1) {
        float currentSlope = CALIBRATION_FACTOR / K;
        CALIBRATION_OFFSET = calibTargetY[0] - (calibRawX[0] * currentSlope);
    } else {
        double sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
        for (int i = 0; i < calibPointCount; i++) {
            sumX  += calibRawX[i];
            sumY  += calibTargetY[i];
            sumXY += (double)calibRawX[i] * calibTargetY[i];
            sumX2 += (double)calibRawX[i] * calibRawX[i];
        }

        double denom = (calibPointCount * sumX2) - (sumX * sumX);
        if (fabs(denom) > 1e-9) {
            double m = ((double)calibPointCount * sumXY - sumX * sumY) / denom;
            double b = (sumY - m * sumX) / (double)calibPointCount;

            CALIBRATION_OFFSET = (float)b;
            K = DEFAULT_K;
            CALIBRATION_FACTOR = (float)(m * K);
        }
    }

    saveSettingsToEEPROM();
    dumpEEPROM();
}

// ----------------------------------------------------------------------------
// Measurement Logic
// ----------------------------------------------------------------------------
uint32_t measureChargeTime() {
    gpio_set_level((gpio_num_t)CHARGE_PIN, 0);
    delayMicroseconds(200);

    noInterrupts();
    uint32_t startCycles = esp_cpu_get_cycle_count();
    gpio_set_level((gpio_num_t)CHARGE_PIN, 1);
    while (true) {
        if (fastRead()) break;
        if ((esp_cpu_get_cycle_count() - startCycles) > TIMEOUT_CYC) break;
    }

    uint32_t elapsedCycles = esp_cpu_get_cycle_count() - startCycles;
    interrupts();

    gpio_set_level((gpio_num_t)CHARGE_PIN, 0);
    return elapsedCycles;
}

// ----------------------------------------------------------------------------
// Setup Routine
// ----------------------------------------------------------------------------
void setup() {
    pinMode(CHARGE_PIN, OUTPUT);
    pinMode(SENSE_PIN, INPUT);
    pinMode(BTN_MENU, INPUT_PULLUP);
    pinMode(BTN_VALUE, INPUT_PULLUP);
    pinMode(BTN_OK, INPUT_PULLUP);
    pinMode(RES_PIN, INPUT_PULLUP);

    digitalWrite(CHARGE_PIN, LOW);
    delay(100);

    // Swallow initial transients
    buttonPressed(BTN_MENU);
    buttonPressed(BTN_VALUE);
    buttonPressed(BTN_OK);

    Serial.begin(115200);
    delay(200);
    Serial.setTimeout(0);
    printHelp();
    // Initialize I2C Bus
    Wire.begin();

    if (loadSettingsFromEEPROM()) {
        dumpEEPROM();
    }

    if (serialLogLevel == SERIAL_ALL_VERBOSE) {
        Serial.println("Serial begin.");
        printHelp();
    }
    Serial.flush();

    u8g2.begin();
    u8g2.setContrast(1);//27);
    u8g2.setFontMode(1);
    u8g2.setBitmapMode(1);
}

// ----------------------------------------------------------------------------
// Main Loop
// ----------------------------------------------------------------------------
void loop() {
    bool tareAbsolute = TARE_VALUES[tareIndex];
    int numSamples    = SAMPLE_VALUES[sampleIndex];
    RunMode runMode   = RUN_VALUES[runIndex];
    float activeResistor = (digitalRead(RES_PIN) == LOW) ? R_LOW : R_HIGH;

    // --- 1. Acquisition & Statistical Processing ---
    if (!inCalibMode) {
        if (runMode == RUN_CONTINUOUS || measurementRequested) {
            sampleBuffer[sampleCount] = measureChargeTime();
            sampleCount++;

            if (sampleCount >= (uint32_t)numSamples) {
                float avgCycles = computeFilteredAverageCycles(sampleBuffer, sampleCount);
                avgTimeUs = avgCycles / 160.0f;

                capacitanceBase = (avgTimeUs / 1000000.0f) / (activeResistor * K) * CALIBRATION_FACTOR;
                capacitancePicofarads = (capacitanceBase * 1e12f) + CALIBRATION_OFFSET;
                sampleCount = 0;
                measurementRequested = false;

                // Output according to current Serial Output Level configuration
                if (serialLogLevel == SERIAL_ALL_VERBOSE) {
                    Serial.printf("Time: %.2f us   Cap: %.2f pF   (Res: %.1f Mohm)\n",
                        avgTimeUs, capacitancePicofarads, activeResistor / 1000000.0f);
                } else if (serialLogLevel == SERIAL_ALL_VALUES_ONLY) {
                    // CSV format: f, time_us, cap_pF, res_ohm
                    Serial.printf("f,%.2f,%.2f,%.0f\n", avgTimeUs, capacitancePicofarads, activeResistor);
                }
            }
        }
    }

    // --- 2. Input Handling (Hardware Buttons + Serial Keywords) ---
    bool menuTriggered  = buttonPressed(BTN_MENU);
    bool valueTriggered = buttonPressed(BTN_VALUE);
    bool okTriggered    = buttonPressed(BTN_OK);

    // Read Serial inputs for command strings
    if (Serial.available() > 0) {
        String cmd = Serial.readStringUntil('\n');
        cmd.trim();
        cmd.toLowerCase();

        if (cmd == "menu" || cmd == "m") {
            menuTriggered = true;
        } else if (cmd == "value" || cmd == "v") {
            valueTriggered = true;
        } else if (cmd == "ok" || cmd == "o") {
            okTriggered = true;
        } else if (cmd == "s" || cmd == "serial") {
            // Cycle Serial level settings: Verbose -> Values Only (CSV) -> EEPROM Only
            serialLogLevel = static_cast<SerialOutputLevel>((serialLogLevel + 1) % 3);
            
            Serial.print("[System] Serial level set to: ");
            if (serialLogLevel == SERIAL_ALL_VERBOSE) {
                Serial.println("ALL (Verbose)");
            } else if (serialLogLevel == SERIAL_ALL_VALUES_ONLY) {
                Serial.println("ALL (Values Only CSV: f,time_us,cap_pF,res_ohm)");
            } else {
                Serial.println("ONLY EEPROM");
            }
        } else if (cmd == "save") {
            saveSettingsToEEPROM();
            dumpEEPROM();
        } else if (cmd == "load") {
            if (loadSettingsFromEEPROM()) {
                dumpEEPROM();
            }
        } else if (cmd == "reset") {
            resetSettingsToDefaults();
        } else if (cmd == "help" || cmd == "h") {
            printHelp();
        } else if (cmd.startsWith("b ") || cmd.startsWith("bright ")) {
            int val = cmd.substring(cmd.indexOf(' ') + 1).toInt();
            val = constrain(val, 0, 255);
            u8g2.setContrast(val);
            Serial.printf("[System] OLED brightness set to: %d\n", val);
        }
    }

    if (inCalibMode) {
        // --- Calibration Interactive Input Navigation ---
        if (menuTriggered) {
            calibElement = (calibElement + 1) % 7;
        }

        if (valueTriggered) {
            if (calibElement <= 3) {
                calibDigits[calibElement] = (calibDigits[calibElement] + 1) % 10;
            } else if (calibElement == 4) {
                calibUnitIndex = (calibUnitIndex + 1) % 3;
            }
        }

        if (okTriggered) {
            if (calibElement == 5) { // [MEASURE]
                float targetVal = (calibDigits[0] * 100.0f + calibDigits[1] * 10.0f +
                                   calibDigits[2] * 1.0f + calibDigits[3] * 0.1f) * CALIB_UNIT_MULT[calibUnitIndex];

                int numS = MAX_SAMPLES;
                for (int i = 0; i < numS; i++) {
                    sampleBuffer[i] = measureChargeTime();
                }
                float avgCycles = computeFilteredAverageCycles(sampleBuffer, numS);
                float tUs = avgCycles / 160.0f;
                float rawX = (tUs / 1000000.0f) / activeResistor * 1e12f;

                if (calibPointCount < MAX_CALIB_POINTS) {
                    calibRawX[calibPointCount] = rawX;
                    calibTargetY[calibPointCount] = targetVal;
                    calibPointCount++;
                    snprintf(calibStatusMsg, sizeof(calibStatusMsg), "Point #%d Saved!", calibPointCount);
                } else {
                    snprintf(calibStatusMsg, sizeof(calibStatusMsg), "Buffer Full!");
                }
            } else if (calibElement == 6) { // [END]
                performCalibrationFitting();
                inCalibMode = false;
                snprintf(calibStatusMsg, sizeof(calibStatusMsg), "Saved to EEPROM!");
            }
        }
    } else {
        // --- Normal Mode Navigation ---
        if (menuTriggered) {
            currentMenu = (MenuItem)((currentMenu + 1) % 4);
        }

        if (valueTriggered) {
            switch (currentMenu) {
                case MENU_TARE:
                    tareIndex = (tareIndex + 1) % TARE_COUNT;
                    break;
                case MENU_SAMPLES:
                    sampleIndex = (sampleIndex + 1) % SAMPLE_COUNT;
                    sampleCount = 0;
                    measurementRequested = true;
                    break;
                case MENU_RUN:
                    runIndex = (runIndex + 1) % RUN_COUNT;
                    break;
                case MENU_CALIB:
                    calibIndex = (calibIndex + 1) % CALIB_COUNT;
                    break;
            }
        }

        if (okTriggered) {
            switch (currentMenu) {
                case MENU_TARE:
                    if (!TARE_VALUES[tareIndex]) {
                        tareValue = capacitancePicofarads;
                        sampleCount = 0;
                        measurementRequested = true;
                    }
                    break;
                case MENU_SAMPLES:
                    break;
                case MENU_RUN:
                    sampleCount = 0;
                    measurementRequested = true;
                    break;

                case MENU_CALIB:
                    if (calibIndex == 0) { // "Calibrate"
                        runIndex = 1; // RUN_STOP
                        sampleCount = 0;
                        measurementRequested = false;

                        inCalibMode = true;
                        calibElement = 0;
                        calibPointCount = 0;
                        snprintf(calibStatusMsg, sizeof(calibStatusMsg), "Set val & [MEAS]");
                    } else if (calibIndex == 1) { // "Reset Def"
                        resetSettingsToDefaults();
                    } else if (calibIndex == 2) { // "Load EEPROM"
                        if (loadSettingsFromEEPROM()) {
                            dumpEEPROM();
                        }
                    } else if (calibIndex == 3) { // "Save EEPROM"
                        saveSettingsToEEPROM();
                        dumpEEPROM();
                    }
                    break;
            }
        }
    }

    // --- 3. Tare Processing ---
    if (tareAbsolute) {
        displayCapacitance = capacitancePicofarads;
    } else {
        displayCapacitance = capacitancePicofarads - tareValue;
    }

    bool isLowResistor = (digitalRead(RES_PIN) == LOW);

    // --- 4. Non-Blocking OLED Rendering (Every 100ms) ---
    if (millis() - lastDisplayUpdate >= 100) {
        lastDisplayUpdate = millis();
        u8g2.firstPage();
        do {
            if (inCalibMode) {
                // CALIBRATION INPUT SCREEN
                u8g2.setFont(u8g2_font_5x7_tf);
                u8g2.drawStr(2, 8, "CALIBRATION MODE");
                char buf[32];
                snprintf(buf, sizeof(buf), "Pts:%d", calibPointCount);
                u8g2.drawStr(92, 8, buf);
                u8g2.drawLine(0, 10, 128, 10);

                u8g2.setFont(u8g2_font_7x14_tr);
                // Digit 100s
                if (calibElement == 0) u8g2.drawBox(6, 14, 12, 16);
                u8g2.setCursor(8, 27);
                u8g2.setDrawColor(calibElement == 0 ? 0 : 1);
                u8g2.print(calibDigits[0]);
                u8g2.setDrawColor(1);

                // Digit 10s
                if (calibElement == 1) u8g2.drawBox(18, 14, 12, 16);
                u8g2.setCursor(20, 27);
                u8g2.setDrawColor(calibElement == 1 ? 0 : 1);
                u8g2.print(calibDigits[1]);
                u8g2.setDrawColor(1);

                // Digit 1s
                if (calibElement == 2) u8g2.drawBox(30, 14, 12, 16);
                u8g2.setCursor(32, 27);
                u8g2.setDrawColor(calibElement == 2 ? 0 : 1);
                u8g2.print(calibDigits[2]);
                u8g2.setDrawColor(1);

                // Dot
                u8g2.drawStr(43, 27, ".");

                // Digit 0.1s
                if (calibElement == 3) u8g2.drawBox(48, 14, 12, 16);
                u8g2.setCursor(50, 27);
                u8g2.setDrawColor(calibElement == 3 ? 0 : 1);
                u8g2.print(calibDigits[3]);
                u8g2.setDrawColor(1);

                // Unit
                if (calibElement == 4) u8g2.drawBox(68, 14, 24, 16);
                u8g2.setCursor(70, 27);
                u8g2.setDrawColor(calibElement == 4 ? 0 : 1);
                u8g2.print(CALIB_UNITS[calibUnitIndex]);
                u8g2.setDrawColor(1);

                u8g2.setFont(u8g2_font_5x7_tf);
                // [MEASURE]
                if (calibElement == 5) u8g2.drawBox(6, 34, 52, 12);
                u8g2.setCursor(8, 43);
                u8g2.setDrawColor(calibElement == 5 ? 0 : 1);
                u8g2.print("[MEASURE]");
                u8g2.setDrawColor(1);

                // [END]
                if (calibElement == 6) u8g2.drawBox(68, 34, 34, 12);
                u8g2.setCursor(70, 43);
                u8g2.setDrawColor(calibElement == 6 ? 0 : 1);
                u8g2.print("[END]");
                u8g2.setDrawColor(1);

                u8g2.drawLine(0, 49, 128, 49);
                u8g2.setCursor(2, 59);
                u8g2.print(calibStatusMsg);

            } else {
                // MAIN UI SCREEN
                u8g2.setFont(u8g2_font_5x7_tf);
                char buf[32];
                snprintf(buf, sizeof(buf), "R:%s", isLowResistor ? "1M" : "10M");
                u8g2.drawStr(98, 7, buf);

                u8g2.setFont(u8g2_font_ncenR18_tr);
                char unitStr[8];
                float value = displayCapacitance;
                float absValue = fabs(value);
                strcpy(unitStr, "pF");
                if (absValue >= 1000000000.0f) {
                    value /= 1000000000.0f;
                    strcpy(unitStr, "mF");
                } else if (absValue >= 1000000.0f) {
                    value /= 1000000.0f;
                    strcpy(unitStr, "uF");
                } else if (absValue >= 1000.0f) {
                    value /= 1000.0f;
                    strcpy(unitStr, "nF");
                }

                char formatBuf[32];
                float displayValue = fabs(value);
                if (displayValue < 10.0f) {
                    snprintf(formatBuf, sizeof(formatBuf), "%s%.3f", (value < 0 ? "-" : " "), displayValue);
                } else if (displayValue < 100.0f) {
                    snprintf(formatBuf, sizeof(formatBuf), "%s%.2f", (value < 0 ? "-" : " "), displayValue);
                } else {
                    snprintf(formatBuf, sizeof(formatBuf), "%s%.1f", (value < 0 ? "-" : " "), displayValue);
                }

                u8g2.drawStr(2, 20, formatBuf);
                u8g2.setFont(u8g2_font_ncenB10_tr);
                u8g2.drawStr(98, 20, unitStr);

                // Menu Grid
                u8g2.setFont(u8g2_font_5x7_tf);
                snprintf(buf, sizeof(buf), "TARE %s", TARE_LABELS[tareIndex]);
                if (currentMenu == MENU_TARE) u8g2.drawBox(0, 25, 58, 11);
                u8g2.setCursor(2, 33);
                u8g2.setDrawColor(currentMenu == MENU_TARE ? 0 : 1);
                u8g2.print(buf);
                u8g2.setDrawColor(1);

                snprintf(buf, sizeof(buf), "N=%s", SAMPLE_LABELS[sampleIndex]);
                if (currentMenu == MENU_SAMPLES) u8g2.drawBox(62, 25, 64, 11);
                u8g2.setCursor(64, 33);
                u8g2.setDrawColor(currentMenu == MENU_SAMPLES ? 0 : 1);
                u8g2.print(buf);
                u8g2.setDrawColor(1);

                snprintf(buf, sizeof(buf), "%s", RUN_LABELS[runIndex]);
                if (currentMenu == MENU_RUN) u8g2.drawBox(0, 38, 58, 11);
                u8g2.setCursor(2, 46);
                u8g2.setDrawColor(currentMenu == MENU_RUN ? 0 : 1);
                u8g2.print(buf);
                u8g2.setDrawColor(1);

                snprintf(buf, sizeof(buf), "%s", CALIB_LABELS[calibIndex]);
                if (currentMenu == MENU_CALIB) u8g2.drawBox(62, 38, 64, 11);
                u8g2.setCursor(64, 46);
                u8g2.setDrawColor(currentMenu == MENU_CALIB ? 0 : 1);
                u8g2.print(buf);
                u8g2.setDrawColor(1);

                // Progress Bar
                u8g2.drawFrame(0, 52, 128, 10);
                int progress = (runMode == RUN_STOP) ? 100 : (sampleCount * 100UL) / numSamples;
                if (progress > 100) progress = 100;

                int barWidth = (progress * 124) / 100;
                if (barWidth > 124) barWidth = 124;
                if (barWidth > 0) {
                    u8g2.drawBox(2, 54, barWidth, 6);
                }
            }
        } while (u8g2.nextPage());
    }
}