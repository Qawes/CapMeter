#include <Wire.h>
#include <Protocentral_FDC1004.h>

// I2C pin definíciók ESP32-S3-hoz
#define I2C_SDA_PIN 1
#define I2C_SCL_PIN 2

#define SAMPLE_RATE FDC1004_RATE_100HZ

#define CAPDAC_VAL    17
#define CAPDAC_OFFSET (CAPDAC_VAL * 3.125f)

FDC1004 fdc;

// A 4 mérési slot tömbben a könnyű ciklikus kezeléshez
const fdc1004_measurement_t measurements[4] = {
  FDC1004_MEASUREMENT_1,
  FDC1004_MEASUREMENT_2,
  FDC1004_MEASUREMENT_3,
  FDC1004_MEASUREMENT_4
};

void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial.println("\n--- ProtoCentral FDC1004 4-Csatornás Mérés (ESP32-S3) ---");

  // I2C busz indítása a megadott lábakon
  Wire.begin(I2C_SDA_PIN, I2C_SCL_PIN);

  // Mind a 4 csatorna konfigurálása single-ended módban (CINx a GND-hez képest, 0 pF CAPDAC offset)
  for (uint8_t i = 0; i < 4; i++) {
    fdc.configureMeasurementSingle(measurements[i], i, CAPDAC_VAL);
  }

  Serial.println("Mind a 4 csatorna konfigurálva.");
}

void loop() {
  float cap_values[4];

  for (uint8_t i = 0; i < 4; i++) {
    // Mérés indítása az adott sloton
    fdc.triggerSingleMeasurement(measurements[i], SAMPLE_RATE);
    
    // Várakozás a mérés lefutására (100 Hz mintaaránynál ~12-15 ms elegendő)
    delay(15);

    uint16_t value[2];
    if (!fdc.readMeasurement(measurements[i], value)) {
      // 24-bites nyers fixpontos érték előállítása
      int32_t raw_value = ((int32_t)value[0] << 8) | ((value[1] >> 8) & 0xFF);
      if (raw_value & 0x00800000) {
        raw_value |= 0xFF000000; // Előjelkiterjesztés
      }

      // Kapacitás (pF) = nyers_érték / 2^19
      cap_values[i] = (float)raw_value / 524288.0f;
    } else {
      cap_values[i] = NAN; // Hiba esetén 'Not a Number'
    }
  }

  // Eredmények kiíratása egy sorban
  Serial.printf("CIN1: %8.4f pF | CIN2: %8.4f pF | CIN3: %8.4f pF | CIN4: %8.4f pF\n",
                cap_values[0], cap_values[1], cap_values[2], cap_values[3]);

  delay(200);
}