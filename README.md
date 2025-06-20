# Dialogue-to-Audio Generator with Nari Dia

This Python script takes structured dialogue from a `.txt` file (formatted as a list of JSON objects), splits it into natural chunks based on token limits, assigns consistent speaker labels, and generates high-quality speech audio using the [Nari Dia](https://github.com/nari-labs/dia) TTS model.

The final output includes:

* 🎧 Merged `.wav` audio of the full dialogue
* 🔉 Individual `.mp3` audio files for each chunk
* 📝 `.srt` subtitle file
* 📄 Human-readable chunked text file

---

## ✅ Features

* Chunking logic respects token limits (default: 150 tokens)
* Automatic speaker-to-voice mapping (\[S1], \[S2], ...) using dynamic mapping
* Handles duplicate or similar speaker names reliably
* Slows down speech using `pydub` post-processing
* Supports GPU acceleration (CUDA) if available
* Uses FFmpeg-compatible audio export
* Outputs all files to a folder called `SoundOutput`

---

## 📁 Input Format

The input should be a `.txt` file containing a **JSON array** like this:

```json
[
  {"speaker": "Ali", "text": "Hey Beth, have you been following the latest news?"},
  {"speaker": "Beth", "text": "Yes, it's been pretty intense."}
]
```

---

## 📦 Installation & Setup

### 1. Clone the Repo

```bash
git clone https://github.com/stabbler53/TextToSound.git
cd TextToSound
```

### 2. Create Environment (Optional but Recommended)

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

### 4. Install Additional Dependencies

* **FFmpeg** is required for audio playback and export.

#### Windows FFmpeg Installation:

* Download: [https://www.gyan.dev/ffmpeg/builds/](https://www.gyan.dev/ffmpeg/builds/)
* Extract and add `ffmpeg/bin` to your system `PATH`
* Verify with:

```bash
ffmpeg -version
```

---

## 🚀 Usage

Run the script:

```bash
python dialogue_to_audio.py
```

A file picker will open. Choose your dialogue `.txt` file.
The output will be saved to:

```
SoundOutput/
├── final_dialogue.wav
├── final_dialogue.srt
├── chunked_text.txt
├── chunk_1.mp3
├── chunk_2.mp3
└── ...
```

---

## 📋 requirements.txt

```txt
pydub
nari-dia
tiktoken
torch
torchvision
torchaudio
descript-audio-codec
```

---

## 🧠 Notes

* The model is large (\~6 GB), so first load will take time.
* If you see an error like `nvrtc-builtins64_118.dll not found`, switch to CPU or install CUDA 11.8.
* If audio sounds too fast, speed is reduced by 15% automatically.
* The `dia/` model folder is excluded to avoid embedding sub-repositories. Install with: `pip install nari-dia`

---

## 🧑‍💻 Author

* Created by Mohamed Rashad

