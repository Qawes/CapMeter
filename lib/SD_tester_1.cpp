#include <Arduino.h>
#include "SdFat.h"
#include "sdios.h"

// ESP32-S3 CAM SPI lábkiosztása (Freenove / AI-Thinker)
#define SD_CS_PIN    37
#define SD_MOSI_PIN  38
#define SD_MISO_PIN  40
#define SD_SCK_PIN   39

// SPI és SdFat konfiguráció
SPIClass sdSPI = SPIClass(HSPI);
SdFat sd;
FsFile file;

bool sdInitialized = false;

// --- Segédfüggvény: SD inicializálása ---
bool initSD() {
  if (sdInitialized) return true;

  sdSPI.begin(SD_SCK_PIN, SD_MISO_PIN, SD_MOSI_PIN, SD_CS_PIN);

  // SPI konfiguráció 16 MHz órajellel és egyedi SPI busszal
  SdSpiConfig spiConfig(SD_CS_PIN, DEDICATED_SPI, SD_SCK_MHZ(16), &sdSPI);

  if (!sd.begin(spiConfig)) {
    Serial.println("[HIBA] Nem sikerült csatolni az SD kártyát!");
    sd.initErrorPrint(&Serial);
    return false;
  }

  sdInitialized = true;
  return true;
}

// --- Parancs: Információk lekérése ---
void printSDInfo() {
  if (!initSD()) return;

  uint32_t cardSize = sd.card()->sectorCount();
  if (cardSize == 0) {
    Serial.println("[HIBA] Nem sikerült beolvasni a szektorok számát.");
    return;
  }

  uint32_t sizeMB = cardSize / 2048; // 512 bájt / szektor -> 2048 szektor = 1 MB
  uint32_t freeMB = sd.freeClusterCount() * sd.sectorsPerCluster() / 2048;

  Serial.println("\n--- SD KÁRTYA INFORMÁCIÓK ---");
  Serial.print("FAT Típus: ");
  if (sd.fatType() == FAT_TYPE_FAT16) Serial.println("FAT16");
  else if (sd.fatType() == FAT_TYPE_FAT32) Serial.println("FAT32");
  else if (sd.fatType() == FAT_TYPE_EXFAT) Serial.println("ExFAT");
  else Serial.println("Ismeretlen");

  Serial.printf("Kapacitás: %u MB\n", sizeMB);
  Serial.printf("Szabad hely: %u MB\n", freeMB);
  Serial.printf("Felhasznált: %u MB\n", sizeMB - freeMB);
  Serial.println("-----------------------------");
}

// --- Parancs: Írási / Olvasási Teszt ---
void runSDTest() {
  if (!initSD()) return;

  Serial.println("\n--- SD ÍRÁS / OLVASÁS TESZT ---");
  const char *testPath = "/teszt.txt";

  // Írás
  Serial.printf("Fájl írása (%s)...\n", testPath);
  if (!file.open(testPath, O_WRONLY | O_CREAT | O_TRUNC)) {
    Serial.println("[HIBA] Nem sikerült megnyitni a fájlt írásra!");
    return;
  }
  file.println("ESP32-S3 CAM teszt üzenet.");
  file.printf("Időbélyeg: %lu ms\n", millis());
  file.close();
  Serial.println("[OK] Írás befejezve.");

  // Olvasás
  Serial.printf("Fájl olvasása (%s)...\n", testPath);
  if (!file.open(testPath, O_RDONLY)) {
    Serial.println("[HIBA] Nem sikerült megnyitni a fájlt olvasásra!");
    return;
  }
  Serial.println("--- Tartalom ---");
  while (file.available()) {
    Serial.write(file.read());
  }
  Serial.println("\n----------------");
  file.close();
  Serial.println("[SIKER] A teszt sikeresen befejeződött.");
}

// --- Parancs: SD Kártya Formázása (SdFat formázóval) ---
void formatSDCard() {
  // A formázáshoz inicializáljuk az alacsony szintű kártyakezelőt
  sdSPI.begin(SD_SCK_PIN, SD_MISO_PIN, SD_MOSI_PIN, SD_CS_PIN);
  SdSpiConfig spiConfig(SD_CS_PIN, DEDICATED_SPI, SD_SCK_MHZ(16), &sdSPI);

  Serial.println("\n[FIGYELEM] SD kártya formázása folyamatban...");
  Serial.println("Kérlek várj, ez a kártya méretétől függően eltarthat pár másodpercig...");

  // Formázás a beépített SdFat formázó motorral (FAT32 vagy ExFAT kártyamérettől függően)
  bool success = sd.format(&Serial);

  if (success) {
    Serial.println("[SIKER] Az SD kártya formázása sikeresen befejeződött!");
  } else {
    Serial.println("[HIBA] A formázás sikertelen volt!");
    Serial.print("Kártya hibakód: 0x");
    Serial.println(sd.card()->errorCode(), HEX);
    Serial.print("Kártya hibaadat: 0x");
    Serial.println(sd.card()->errorData(), HEX);
  }

  // Újracsatolás
  sdInitialized = false;
  initSD();
}

// --- Parancsfeldolgozó ---
void handleCommand(String cmd) {
  cmd.trim();
  cmd.toUpperCase();

  if (cmd.length() == 0) return;

  if (cmd == "TEST") {
    runSDTest();
  } else if (cmd == "FORMAT") {
    formatSDCard();
  } else if (cmd == "INFO") {
    printSDInfo();
  } else if (cmd == "HELP") {
    Serial.println("\n--- Elérhető parancsok ---");
    Serial.println("  TEST   - Írási és olvasási teszt futtatása");
    Serial.println("  INFO   - SD kártya adatainak lekérése");
    Serial.println("  FORMAT - SD kártya teljes formázása");
    Serial.println("  HELP   - Parancslista kiírása");
    Serial.println("---------------------------");
  } else {
    Serial.printf("[HIBA] Ismeretlen parancs: '%s'. Írd be: 'HELP'\n", cmd.c_str());
  }
}

void setup() {
  Serial.begin(115200);
  while (!Serial) { delay(10); }
  delay(1000);

  Serial.println("\n=================================");
  Serial.println(" ESP32-S3 CAM SD Teszt & Kezelő");
  Serial.println("=================================");
  Serial.println("Írd be a parancsot (pl. 'TEST', 'FORMAT', 'INFO', 'HELP'):\n");

  initSD();
}

void loop() {
  if (Serial.available()) {
    String input = Serial.readStringUntil('\n');
    handleCommand(input);
  }
}