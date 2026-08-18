#include <Wire.h>
#include <Protocentral_FDC1004.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>

#define I2C_SDA_PIN 1
#define I2C_SCL_PIN 2

#define SAMPLE_RATE   FDC1004_RATE_100HZ
#define CAPDAC_VAL    17
#define CAPDAC_OFFSET (CAPDAC_VAL * 3.125f)

// Teljes szabványos Nordic UART Service (NUS)
#define SERVICE_UUID           "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
#define CHARACTERISTIC_UUID_RX "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"
#define CHARACTERISTIC_UUID_TX "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"

FDC1004 fdc;

BLEServer *pServer = NULL;
BLECharacteristic *pTxCharacteristic = NULL;
bool deviceConnected = false;
bool oldDeviceConnected = false;

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

void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial.println("\n--- FDC1004 BLE (Nordic UART) ---");

  Wire.begin(I2C_SDA_PIN, I2C_SCL_PIN);

  for (uint8_t i = 0; i < 4; i++) {
    fdc.configureMeasurementSingle(measurements[i], i, CAPDAC_VAL);
  }

  // BLE Inicializálás
  BLEDevice::init("ESP32-S3-FDC1004");
  pServer = BLEDevice::createServer();
  pServer->setCallbacks(new ServerCallbacks());

  BLEService *pService = pServer->createService(SERVICE_UUID);

  // TX Karakterisztika (ESP -> Telefon)
  pTxCharacteristic = pService->createCharacteristic(
                        CHARACTERISTIC_UUID_TX,
                        BLECharacteristic::PROPERTY_NOTIFY | 
                        BLECharacteristic::PROPERTY_READ
                      );
  pTxCharacteristic->addDescriptor(new BLE2902());

  // RX Karakterisztika (Telefon -> ESP) – erre van szüksége a Serial Terminal appnak a profil felismeréséhez
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
  float cap_values[4];

  for (uint8_t i = 0; i < 4; i++) {
    fdc.triggerSingleMeasurement(measurements[i], SAMPLE_RATE);
    delay(15);

    uint16_t value[2];
    if (!fdc.readMeasurement(measurements[i], value)) {
      int32_t raw_value = ((int32_t)value[0] << 8) | ((value[1] >> 8) & 0xFF);
      if (raw_value & 0x00800000) {
        raw_value |= 0xFF000000;
      }
      cap_values[i] = ((float)raw_value / 524288.0f) + CAPDAC_OFFSET;
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