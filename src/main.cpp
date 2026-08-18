#include "FS.h"
#include "SD_MMC.h"
#include "vfs_api.h"
#include "esp_vfs_fat.h"

// A dokumentációd szerinti pinek definíciója
#define SD_MMC_CLK  39
#define SD_MMC_CMD  38
#define SD_MMC_D0   40

void cardInfo() {
  uint16_t cardType = SD_MMC.cardType();
  if(cardType == CARD_NONE){
    Serial.println("❌ Nincs kártya behelyezve.");
    return;
  }

  Serial.print("📁 Kártya típusa: ");
  if(cardType == CARD_MMC) Serial.println("MMC");
  else if(cardType == CARD_SD) Serial.println("SDSC");
  else if(cardType == CARD_SDHC) Serial.println("SDHC");
  else Serial.println("ISMERETLEN");

  uint64_t cardSize = SD_MMC.cardSize() / (1024 * 1024);
  Serial.printf("💾 Kártya mérete: %llu MB\n", cardSize);
  Serial.printf("📊 Felhasznált terület: %llu MB\n", SD_MMC.usedBytes() / (1024 * 1024));
}

bool initSD() {
  // Pinek beállítása a rajz szerint
  if (!SD_MMC.setPins(SD_MMC_CLK, SD_MMC_CMD, SD_MMC_D0)) {
    Serial.println("❌ Pin konfigurációs hiba!");
    return false;
  }

  // Inicializálás 1-bites módban (true = 1-bit)
  // A harmadik paraméter (false) azt jelenti, hogy NEM formázza automatikusan bootoláskor
  if (!SD_MMC.begin("/sdcard", true, false)) { 
    Serial.println("❌ SD kártya mountolása sikertelen.");
    return false;
  }
  
  Serial.println("✅ SD kártya sikeresen inicializálva!");
  cardInfo();
  return true;
}

void formatSD() {
  Serial.println("\n⚠️ A kártya formázása nem támogatott ebben az Arduino SD_MMC API-ban.");
  Serial.println("A biztonságos alternatíva: a kártya lecsatolása és újra inicializálása.");

  SD_MMC.end();
  delay(500);

  Serial.println("🔄 Kártya újracsatlakoztatása...");
  initSD();
}

bool writeAndReadTest(const char *path, const char *message) {
  File file = SD_MMC.open(path, FILE_WRITE);
  if (!file) {
    Serial.printf("❌ A fájl megnyitása sikertelen írásra: %s\n", path);
    return false;
  }

  file.println(message);
  file.printf("Timestamp: %lu ms\n", millis());
  file.close();

  Serial.printf("✍️ Írás kész: %s\n", path);

  file = SD_MMC.open(path, FILE_READ);
  if (!file) {
    Serial.printf("❌ A fájl megnyitása sikertelen olvasásra: %s\n", path);
    return false;
  }

  Serial.printf("📖 Olvasás: %s\n", path);
  while (file.available()) {
    Serial.write(file.read());
  }
  Serial.println();
  file.close();

  return true;
}

void setup() {
  Serial.begin(115200);
  while(!Serial) { delay(10); } // Várakozás az USB soros kapcsolatra
  
  Serial.println("\n=== ESP32-S3 SDMMC Tesztelő és Formázó ===");
  Serial.println("Parancsok listája:");
  Serial.println("  FORMAT -> SD kártya lecsatolása és újra inicializálása");
  Serial.println("  INFO   -> Kártya adatainak lekérése");
  Serial.println("  TEST   -> Írás + olvasás teszt az SD kártyára");
  Serial.println("  WRITE <szöveg> -> Közvetlen fájlba írás a /teszt.txt fájlba");
  Serial.println("  READ   -> A /teszt.txt fájl visszaolvasása\n");

  initSD();
}

void loop() {
  if (Serial.available() > 0) {
    String input = Serial.readStringUntil('\n');
    input.trim();
    input.toUpperCase();

    if (input == "FORMAT") {
      formatSD();
    }
    else if (input == "INFO") {
      cardInfo();
    }
    else if (input == "TEST") {
      Serial.println("🧪 Futtatok írás/olvasás tesztet...");
      writeAndReadTest("/teszt.txt", "ESP32 SDMMC teszt üzenet");
    }
    else if (input == "READ") {
      File file = SD_MMC.open("/teszt.txt", FILE_READ);
      if (!file) {
        Serial.println("❌ A /teszt.txt fájl nem található.");
      } else {
        Serial.println("📖 /teszt.txt tartalma:");
        while (file.available()) {
          Serial.write(file.read());
        }
        Serial.println();
        file.close();
      }
    }
    else if (input.startsWith("WRITE ")) {
      String msg = input.substring(6);
      if (msg.length() == 0) {
        Serial.println("❌ Üres üzenet. Használat: WRITE <szöveg>");
      } else {
        writeAndReadTest("/teszt.txt", msg.c_str());
      }
    }
    else if (input != "") {
      Serial.printf("❓ Ismeretlen parancs: '%s'. Használd a 'TEST', 'READ', 'WRITE <szöveg>', 'INFO' vagy 'FORMAT' parancsokat.\n", input.c_str());
    }
  }
}
