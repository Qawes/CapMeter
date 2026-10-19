import numpy as np
import random
from io import BytesIO
from scipy.io import wavfile
from gtts import gTTS
from pydub import AudioSegment

sample_rate = 44100  # Standard CD minőségű mintavételezés
LANGUAGE = "hu"  # Választható: "hu" vagy "en"

def generate_beep(freq, duration, sample_rate=44100):
    """Létrehoz egy adott frekvenciájú és hosszúságú sipoló hangot (beeps)"""
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    # Szinusz hullám + egy kis lecsengés (envelope), hogy ne kattanjon a hang
    tone = np.sin(2 * np.pi * freq * t)
    envelope = np.exp(-t * 5) # gyors lecsengés a metronóm tikkogásához
    return tone * envelope

def generate_speech(text, language):
    """A szöveget 44,1 kHz-es mono hangmintákká alakítja."""
    speech = BytesIO()
    gTTS(text=text, lang=language).write_to_fp(speech)
    speech.seek(0)
    speech_audio = AudioSegment.from_file(speech, format="mp3")
    speech_audio = speech_audio.set_frame_rate(sample_rate).set_channels(1).set_sample_width(2)
    return np.asarray(speech_audio.get_array_of_samples(), dtype=np.int16) / 32768.0

def generate_announcement(direction, freq, language):
    """Elkészíti a következő feladatról szóló bemondást."""
    directions = {
        "hu": {"LR": "Jobbra-balra", "UD": "Föl-le"},
        "en": {"LR": "left-right", "UD": "up-down"},
    }
    if language not in directions:
        raise ValueError("A nyelv csak 'hu' vagy 'en' lehet.")

    if language == "hu":
        text = f"{directions[language][direction]} mozgás következik, {freq:g} hertzes sebességgel."
    else:
        text = f"Next: {directions[language][direction]} movement at {freq:g} hertz."

    return generate_speech(text, language)

def generate_break_announcement(language):
    """Elkészíti a hosszú szünet kezdetét jelző bemondást."""
    text = (
        "Egy perc szünet következik."
        if language == "hu"
        else "A one-minute break will now begin."
    )
    return generate_speech(text, language)

def build_test_audio(
    tasks_sequence, rest_duration=10, language=LANGUAGE, announcement_pause=2
):
    """
    tasks_sequence: pl. [('LR', 2.0, 15), ('UD', 0.5, 15)] 
    ahol (irány, frekvencia_Hz, időtartam_sec)
    """
    audio_signal = []
    
    # 1. Alapvonal (Baseline) - 10 mp csend az elején
    audio_signal.append(np.zeros(int(sample_rate * 10)))
    
    for direction, freq, duration in tasks_sequence:
        # A) VÁLTÁS JELZÉSE (Pl. egy dupla magas sipolás, hogy tudja, új feladat jön)
        start_signal = generate_beep(880, 0.1) # Magas hang (A5)
        audio_signal.append(start_signal)
        
        # B) PIHENŐ ÉS FELKÉSZÜLÉS - bemondás a szünet végén
        announcement = generate_announcement(direction, freq, language)
        rest_samples = int(sample_rate * rest_duration)
        pause_samples = int(sample_rate * announcement_pause)
        if len(announcement) + pause_samples > rest_samples:
            raise ValueError(
                "A bemondás és az utána következő szünet hosszabb a teljes pihenőnél."
            )
        leading_silence = rest_samples - len(announcement) - pause_samples
        audio_signal.append(
            np.concatenate(
                (
                    np.zeros(leading_silence),
                    announcement,
                    np.zeros(pause_samples),
                )
            )
        )
        
        # C) METRONÓM TIKKOGÁS A MOZGÁSHOZ
        # Kiszámoljuk, hány másodpercenként kell tikkannia a metronómnak
        # Pl. 2 Hz-nél 0.5 mp-enként (1 Hz-es tikkogás a két szélsőértékhez)
        interval = 1.0 / (freq * 2) 
        num_ticks = int(duration / interval)
        
        task_audio = np.zeros(int(sample_rate * duration))
        for i in range(num_ticks):
            tick = generate_beep(440, 0.05) # Normál metronóm hang (A4)
            idx = int(i * interval * sample_rate)
            if idx + len(tick) < len(task_audio):
                task_audio[idx:idx+len(tick)] = tick
                
        audio_signal.append(task_audio)
        
    # Összefűzés és konvertálás 16-bites WAV-vá
    full_audio = np.concatenate(audio_signal)
    full_audio = (full_audio * 32767 / np.max(np.abs(full_audio))).astype(np.int16)
    
    return full_audio

# Példa használat:
# Irányok: 'LR' (jobbra-balra), 'UD' (föl-le)
# [(irány, Hz, másodperc)]
tasks = [
    ('LR', 2.0, 10),
    ('UD', 1.0, 10),
    ('LR', 0.5, 10),
    ('UD', 0.5, 10),
    ('LR', 1.0, 10)
]

random_seed = random.randrange(2**32)
random.seed(random_seed)
first_block_tasks = tasks.copy()
random.shuffle(first_block_tasks)

second_block_tasks = tasks.copy()
random.shuffle(second_block_tasks)
while second_block_tasks == first_block_tasks:
    random.shuffle(second_block_tasks)

print(f"1. blokk feladatsorrendje: {first_block_tasks}")
print(f"2. blokk feladatsorrendje: {second_block_tasks}")

first_block_audio = build_test_audio(
    first_block_tasks, rest_duration=20, language=LANGUAGE, announcement_pause=2
)
second_block_audio = build_test_audio(
    second_block_tasks, rest_duration=20, language=LANGUAGE, announcement_pause=2
)
long_break_duration = 60
break_announcement = generate_break_announcement(LANGUAGE)
break_samples = sample_rate * long_break_duration
if len(break_announcement) > break_samples:
    raise ValueError("A hosszú szünet bemondása hosszabb egy percnél.")
block_break = np.concatenate(
    (
        np.rint(break_announcement * 32767).astype(np.int16),
        np.zeros(break_samples - len(break_announcement), dtype=np.int16),
    )
)
audio_data = np.concatenate(
    (first_block_audio, block_break, second_block_audio)
)
filename = f'Functional_tests/alvas_1_teszt_{LANGUAGE}_seed_{random_seed}.wav'
wavfile.write(filename, 44100, audio_data)
print("Hangsáv sikeresen legyártva!")