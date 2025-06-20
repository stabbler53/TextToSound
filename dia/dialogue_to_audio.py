import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import time
import tiktoken
import subprocess
from tkinter import Tk, filedialog
from pydub import AudioSegment
from dia.model import Dia
import torch

# ============ SETTINGS ============ #
MAX_TOKENS = 1500
CHUNK_FOLDER = "SoundOutput"
FINAL_OUTPUT_PATH = os.path.join(CHUNK_FOLDER, "final_dialogue.wav")
SUBTITLE_PATH = os.path.join(CHUNK_FOLDER, "final_dialogue.srt")
CHUNK_TEXT_PATH = os.path.join(CHUNK_FOLDER, "chunked_text.txt")
SILENCE_DURATION_MS = 1200
SLOWDOWN_FACTOR = 0.8
MAX_SPEAKERS = 8  # Optional: limit number of unique speakers (Dia only supports S1–S8 voices)

# ============ Auto Detect Device ============ #
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️ Running on: {device.upper()}")

# ============ Load Nari Dia ============ #
print("📦 Loading Nari Dia with DAC...")
model = Dia.from_pretrained("nari-labs/Dia-1.6B", compute_dtype="float16", device=device, load_dac=True)

# ============ File Picker ============ #
def select_input_file():
    root = Tk()
    root.withdraw()
    return filedialog.askopenfilename(
        title="Select dialogue .txt file",
        filetypes=[("Text files", "*.txt")]
    )

# ============ Load JSON Array ============ #
def load_json_array_dialogue(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            return [d for d in data if "speaker" in d and "text" in d]
        except Exception as e:
            print("❌ Invalid JSON:", e)
            return []

# ============ Speaker Mapping ============ #
def map_speakers(dialogue):
    speaker_map = {}
    current_voice = 1

    for turn in dialogue:
        name = turn["speaker"]
        if name not in speaker_map:
            if current_voice > MAX_SPEAKERS:
                print(f"⚠️ Too many speakers. Only first {MAX_SPEAKERS} will have voices.")
                speaker_map[name] = f"[S{MAX_SPEAKERS}]"
            else:
                speaker_map[name] = f"[S{current_voice}]"
                current_voice += 1

    print("\n🎤 Speaker Mapping:")
    for speaker, voice in speaker_map.items():
        print(f"  {speaker} → {voice}")
    return speaker_map

# ============ Chunking Function ============ #
def chunk_dialogue(dialogue, max_tokens, speaker_map):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    chunks, current_chunk, current_tokens = [], [], 0

    for turn in dialogue:
        speaker_label = speaker_map[turn["speaker"]]
        line = f"{speaker_label} {turn['text']}"
        token_count = len(tokenizer.encode(line))

        if current_tokens + token_count > max_tokens:
            chunks.append(current_chunk)
            current_chunk = [line]
            current_tokens = token_count
        else:
            current_chunk.append(line)
            current_tokens += token_count

    if current_chunk:
        chunks.append(current_chunk)
    return chunks

# ============ Generate Audio + SRT ============ #
def generate_audio_and_subtitles(chunks):
    os.makedirs(CHUNK_FOLDER, exist_ok=True)
    audio_paths, srt_entries, current_time = [], [], 0.0

    for i, chunk in enumerate(chunks):
        chunk_file = os.path.join(CHUNK_FOLDER, f"chunk_{i+1}.mp3")
        input_text = " ".join(chunk)
        print(f"\n🎙️ Chunk {i+1}/{len(chunks)} | Tokens: {len(input_text.split())}")

        if os.path.exists(chunk_file):
            print("⏩ Skipped (already exists)")
            audio_paths.append(chunk_file)
            duration = AudioSegment.from_file(chunk_file).duration_seconds
        else:
            try:
                start_time = time.time()
                output = model.generate(input_text, use_torch_compile=False)
                model.save_audio(chunk_file, output)

                # Slow audio
                original = AudioSegment.from_file(chunk_file)
                slowed = original._spawn(original.raw_data, overrides={
                    "frame_rate": int(original.frame_rate * SLOWDOWN_FACTOR)
                }).set_frame_rate(original.frame_rate)
                slowed.export(chunk_file, format="mp3")

                duration = slowed.duration_seconds
                print(f"✅ Done in {time.time() - start_time:.2f}s (slowed)")
                audio_paths.append(chunk_file)
            except Exception as e:
                print(f"❌ Error on chunk {i+1}:", e)
                continue

        # Subtitle
        start = time.strftime('%H:%M:%S', time.gmtime(current_time)) + ",000"
        end = time.strftime('%H:%M:%S', time.gmtime(current_time + duration)) + ",000"
        readable_text = "\n".join(chunk)
        srt_entries.append(f"{i+1}\n{start} --> {end}\n{readable_text}\n")

        current_time += duration + (SILENCE_DURATION_MS / 1000)

    with open(SUBTITLE_PATH, "w", encoding="utf-8") as f:
        f.writelines("\n".join(srt_entries))

    print(f"📝 Subtitles saved: {SUBTITLE_PATH}")
    return audio_paths

# ============ Merge Audio ============ #
def merge_audio(audio_paths):
    final = AudioSegment.empty()
    for path in audio_paths:
        final += AudioSegment.from_file(path) + AudioSegment.silent(duration=SILENCE_DURATION_MS)
    final.export(FINAL_OUTPUT_PATH, format="wav")
    print(f"🎧 Final audio: {FINAL_OUTPUT_PATH}")
    return final

# ============ MAIN ============ #
def main():
    print("📂 Please select your dialogue file...")
    path = select_input_file()
    if not path:
        print("❌ No file selected.")
        return

    dialogue = load_json_array_dialogue(path)
    if not dialogue:
        print("⚠️ No valid dialogue.")
        return

    speaker_map = map_speakers(dialogue)

    print(f"\n🧠 Chunking {len(dialogue)} turns...")
    chunks = chunk_dialogue(dialogue, MAX_TOKENS, speaker_map)
    print(f"🔀 {len(chunks)} chunks created.")

    with open(CHUNK_TEXT_PATH, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks):
            f.write(f"--- Chunk {i+1} ---\n")
            f.write("\n".join(chunk))
            f.write("\n\n")
    print(f"📄 Chunked text saved: {CHUNK_TEXT_PATH}")

    audio_paths = generate_audio_and_subtitles(chunks)
    if not audio_paths:
        print("❌ No audio generated. Aborting.")
        return

    final_audio = merge_audio(audio_paths)

    print("▶️ Opening output in default audio player...")
    subprocess.Popen(["start", FINAL_OUTPUT_PATH], shell=True)

    print("🎉 Done! All files saved in:", CHUNK_FOLDER)

if __name__ == "__main__":
    main()
