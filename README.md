# speech-to-text
Speech-to-Text project with Whisper, fine-tuning, and speaker diarization
# 🎙️ Speech-to-Text (ASR) — From Basics to Production

A complete, well-documented **Automatic Speech Recognition** project built on
[Whisper](https://github.com/openai/whisper) and
[faster-whisper](https://github.com/SYSTRAN/faster-whisper). It goes from a
3-line transcription script all the way to a production-style pipeline with
speaker diarization, confidence scoring, and a web app.

> Built as a learning + portfolio project covering the full ASR workflow:
> audio preprocessing → mel spectrograms → Transformer decoding → structured output.

---

## ✨ Features

| Level | What it does |
|------|--------------|
| **1 — Basic** | Transcribe any audio file in 99+ languages, with timestamps |
| **2 — Real-time** | Live microphone transcription using faster-whisper |
| **3 — Fine-tuning** | Train Whisper on custom data (Hindi via Common Voice) |
| **4 — Advanced** | Production pipeline: preprocessing, word-level timestamps, confidence scores, long-audio support, **speaker diarization**, multi-format export, and a **Streamlit web app** |
| **Bonus** | Generate `.srt` subtitles · translate any language → English |

---

## 📁 Files

| File | Description |
|------|-------------|
| `speech_to_text_complete.py` | Levels 1–3 + bonuses, fully commented |
| `speech_to_text_advanced.py` | Level 4: `AdvancedTranscriber` class, CLI, and Streamlit app |
| `speech_to_text_advanced_colab.ipynb` | Run the advanced pipeline in Google Colab (free GPU) |
| `requirements.txt` | All dependencies |

---

## 🚀 Quick start

```bash
git clone https://github.com/<your-username>/speech-to-text.git
cd speech-to-text
pip install -r requirements.txt

# ffmpeg is required (system package):
sudo apt install ffmpeg          # Ubuntu
# brew install ffmpeg            # macOS
```

### Transcribe a file (CLI)
```bash
python speech_to_text_advanced.py meeting.mp3 --model base --formats srt,json,csv
```

### Launch the web app
```bash
streamlit run speech_to_text_advanced.py
```
Drag in an audio file, see the transcript, download it in any format.

### Run on Google Colab
Open `speech_to_text_advanced_colab.ipynb` in Colab, enable a GPU
(*Runtime → Change runtime type → T4 GPU*), and run the cells top to bottom.

---

## 🧠 How it works

```
Audio file (.mp3 / .wav)
      ↓  load, convert to mono, resample to 16 kHz, normalize
Waveform (array of samples)
      ↓  log-mel spectrogram (a 2-D "image" of the sound)
Transformer encoder → decoder
      ↓
Text + word-level timestamps + confidence
```

| Model | Params | Speed | Accuracy |
|-------|--------|-------|----------|
| tiny  | 39M    | fastest | basic |
| base  | 74M    | fast | good *(default)* |
| small | 244M   | medium | better |
| medium| 769M   | slow | great |
| large-v3 | 1.5B | slowest | best |

---

## 🗣️ Speaker diarization (optional)

To label "who spoke when" (SPEAKER_00, SPEAKER_01, …):

1. Create a free token at https://huggingface.co/settings/tokens
2. Accept the terms at https://huggingface.co/pyannote/speaker-diarization-3.1
3. Set it as an environment variable, then run with `--speakers`:

```bash
export HF_TOKEN=hf_xxxxxxxx
python speech_to_text_advanced.py interview.mp3 --speakers
```

If the token or terms are missing, diarization is skipped gracefully and
the rest of the pipeline still works.

---

## 📦 Output formats

`.txt` · `.srt` (subtitles) · `.vtt` (web captions) · `.json` (full structured data) · `.csv` (spreadsheet-friendly)

---

## 🛠️ Tech stack

faster-whisper · OpenAI Whisper · PyTorch · Hugging Face Transformers & Datasets ·
librosa · pyannote.audio · Streamlit · pandas

## 📄 License

MIT
