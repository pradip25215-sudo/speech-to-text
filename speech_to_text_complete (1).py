"""
=============================================================================
 COMPLETE SPEECH-TO-TEXT (AUDIO → TEXT) PROJECT
 From Zero to Production — Full Code with Explanation
=============================================================================

 What is Speech-to-Text (ASR)?
 ─────────────────────────────
 ASR = Automatic Speech Recognition
 It takes audio (your voice, a podcast, a meeting) and converts it into
 written text. This is what powers:
   - Siri, Alexa, Google Assistant
   - YouTube auto-captions
   - Meeting transcription tools (Otter.ai, etc.)

 How does it work internally?
 ────────────────────────────
 Audio File (.mp3/.wav)
      ↓
 Load as waveform (array of numbers representing sound)
      ↓
 Convert to Mel Spectrogram (a 2D "image" of the sound)
      ↓
 Feed into Transformer model (encoder reads the image, decoder writes text)
      ↓
 Output: Text transcription

 This file has 3 LEVELS:
   Level 1 → Basic: Transcribe any audio file (3 lines of code)
   Level 2 → Intermediate: Real-time microphone transcription
   Level 3 → Advanced: Fine-tune Whisper on custom data

 SETUP (run these in terminal first):
   pip install openai-whisper        # Core Whisper model
   pip install faster-whisper        # 4x faster version
   pip install sounddevice           # For microphone recording
   pip install numpy                 # Number crunching
   pip install scipy                 # Audio file saving
   pip install torch                 # PyTorch (deep learning)
   pip install transformers datasets # Hugging Face (for fine-tuning)

 Also install ffmpeg (required by Whisper):
   Ubuntu:  sudo apt install ffmpeg
   Mac:     brew install ffmpeg
   Windows: choco install ffmpeg
=============================================================================
"""

import os
import time

# ═══════════════════════════════════════════════════════════════════════════
# LEVEL 1: BASIC — Transcribe an Audio File
# ═══════════════════════════════════════════════════════════════════════════
# This is the simplest way. You give it an audio file, it gives you text.
# Supports: mp3, wav, m4a, flac, ogg, webm
# Supports: 99+ languages (English, Hindi, Spanish, etc.)
# ═══════════════════════════════════════════════════════════════════════════

def level_1_basic_transcription():
    """
    WHAT THIS DOES:
    ───────────────
    1. Loads the Whisper model into memory
    2. Reads your audio file
    3. Converts audio → mel spectrogram (sound "picture")
    4. Feeds it through the Transformer model
    5. Returns the text

    MODEL SIZES (pick one):
    ───────────────────────
    ┌──────────┬────────────┬──────────┬───────────────┐
    │ Model    │ Parameters │ Speed    │ Accuracy      │
    ├──────────┼────────────┼──────────┼───────────────┤
    │ tiny     │ 39M        │ ~1 sec   │ Basic         │
    │ base     │ 74M        │ ~2 sec   │ Good ← START  │
    │ small    │ 244M       │ ~5 sec   │ Better        │
    │ medium   │ 769M       │ ~15 sec  │ Great         │
    │ large    │ 1.5B       │ ~30 sec  │ Best          │
    │ turbo    │ 809M       │ ~6 sec   │ Great + Fast  │
    └──────────┴────────────┴──────────┴───────────────┘
    """
    import whisper

    # ── Step 1: Load the model ──
    # First time: downloads model weights (~140MB for 'base')
    # After that: loads from cache (~2 seconds)
    print("Loading Whisper model...")
    model = whisper.load_model("base")
    # Options: "tiny", "base", "small", "medium", "large", "turbo"

    # ── Step 2: Transcribe ──
    # Give it any audio file path
    audio_file = "test_audio.mp3"  # ← Change this to your file

    if not os.path.exists(audio_file):
        print(f"\n[!] File '{audio_file}' not found.")
        print("    Create a test file first using: level_1b_record_test_audio()")
        print("    Or provide your own .mp3/.wav file.")
        return

    print(f"Transcribing: {audio_file}")
    start_time = time.time()

    result = model.transcribe(audio_file)

    elapsed = time.time() - start_time
    print(f"Done in {elapsed:.1f} seconds\n")

    # ── Step 3: Read the results ──
    # result is a dictionary with these keys:
    #   result["text"]     → Full transcription as one string
    #   result["segments"] → List of time-stamped chunks
    #   result["language"] → Detected language code (e.g., "en", "hi")

    print("=" * 60)
    print(f"DETECTED LANGUAGE: {result['language']}")
    print("=" * 60)
    print(f"\nFULL TEXT:\n{result['text']}\n")

    # ── Step 4: Show timestamps ──
    print("TIMESTAMPS:")
    print("-" * 60)
    for segment in result["segments"]:
        start = segment["start"]  # Start time in seconds
        end = segment["end"]      # End time in seconds
        text = segment["text"]    # Text for this chunk

        # Convert seconds to MM:SS format
        start_fmt = f"{int(start // 60):02d}:{int(start % 60):02d}"
        end_fmt = f"{int(end // 60):02d}:{int(end % 60):02d}"

        print(f"  [{start_fmt} → {end_fmt}]  {text.strip()}")


def level_1b_record_test_audio():
    """
    HELPER: Record a test audio file from your microphone.
    Speaks for 5 seconds, saves as WAV file.
    """
    import sounddevice as sd
    from scipy.io.wavfile import write

    duration = 5        # seconds
    sample_rate = 16000  # 16kHz (what Whisper expects)

    print(f"Recording for {duration} seconds... SPEAK NOW!")
    audio = sd.rec(
        int(duration * sample_rate),  # total samples = duration × rate
        samplerate=sample_rate,
        channels=1,                   # mono (single channel)
        dtype='float32'               # 32-bit float format
    )
    sd.wait()  # Wait until recording finishes
    print("Recording complete!")

    # Save as WAV file
    # Whisper can read WAV, MP3, M4A, FLAC, OGG, WEBM
    output_file = "test_audio.wav"
    write(output_file, sample_rate, audio)
    print(f"Saved to: {output_file}")
    print("Now run: level_1_basic_transcription()")


# ═══════════════════════════════════════════════════════════════════════════
# LEVEL 2: INTERMEDIATE — Real-Time Microphone Transcription
# ═══════════════════════════════════════════════════════════════════════════
# This listens to your microphone continuously and transcribes in real-time.
# Uses faster-whisper (4x faster than original Whisper).
# ═══════════════════════════════════════════════════════════════════════════

def level_2_realtime_transcription():
    """
    WHAT THIS DOES:
    ───────────────
    1. Opens your microphone
    2. Records audio in chunks (every 3 seconds)
    3. Sends each chunk to faster-whisper
    4. Prints the text as you speak
    5. Press Ctrl+C to stop

    WHY faster-whisper?
    ───────────────────
    - Original Whisper: uses PyTorch, slow on CPU
    - faster-whisper: uses CTranslate2 (optimized C++ engine)
    - Result: 4x faster, uses 50% less memory
    - Same accuracy as original
    """
    from faster_whisper import WhisperModel
    import sounddevice as sd
    import numpy as np

    # ── Step 1: Load faster-whisper model ──
    print("Loading faster-whisper model...")
    model = WhisperModel(
        "base",               # Model size
        device="cpu",          # "cpu" or "cuda" (GPU)
        compute_type="int8"    # int8 = faster on CPU, float16 for GPU
    )
    print("Model loaded! Start speaking...\n")

    # ── Step 2: Configure audio settings ──
    SAMPLE_RATE = 16000   # 16kHz — Whisper's expected input rate
    CHUNK_DURATION = 3    # Transcribe every 3 seconds
    CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_DURATION  # 48000 samples per chunk

    print("=" * 50)
    print(" REAL-TIME TRANSCRIPTION (Ctrl+C to stop)")
    print("=" * 50)

    try:
        while True:
            # ── Step 3: Record a chunk from mic ──
            audio_chunk = sd.rec(
                CHUNK_SAMPLES,
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype='float32'
            )
            sd.wait()  # Wait for chunk to finish recording

            # ── Step 4: Flatten and normalize audio ──
            # sounddevice returns shape (samples, channels)
            # Whisper wants shape (samples,) — a 1D array
            audio_flat = audio_chunk.flatten()

            # Skip silent chunks (saves processing time)
            # np.abs(audio).mean() gives average volume level
            if np.abs(audio_flat).mean() < 0.01:
                # Too quiet — probably silence
                continue

            # ── Step 5: Transcribe the chunk ──
            segments, info = model.transcribe(
                audio_flat,
                beam_size=5,        # Beam search width (higher = more accurate)
                language=None,      # Auto-detect language (or set "en", "hi")
                vad_filter=True     # Voice Activity Detection — skip silence
            )

            # ── Step 6: Print results ──
            for segment in segments:
                text = segment.text.strip()
                if text:
                    print(f"  [{info.language}] {text}")

    except KeyboardInterrupt:
        print("\n\nStopped. Goodbye!")


# ═══════════════════════════════════════════════════════════════════════════
# LEVEL 3: ADVANCED — Fine-Tune Whisper on Your Own Data
# ═══════════════════════════════════════════════════════════════════════════
# Train Whisper on YOUR specific audio data to make it more accurate for
# your use case (e.g., Hindi audio, medical terms, domain-specific vocab).
# Uses Hugging Face Transformers + Datasets.
# ═══════════════════════════════════════════════════════════════════════════

def level_3_finetune_whisper():
    """
    WHAT THIS DOES:
    ───────────────
    1. Loads a pre-trained Whisper model from Hugging Face
    2. Loads a Hindi speech dataset (Common Voice)
    3. Preprocesses audio → features + labels
    4. Fine-tunes the model on Hindi data
    5. Saves your custom model

    WHEN TO FINE-TUNE:
    ──────────────────
    - You need better accuracy for a specific language
    - You have domain-specific vocabulary (medical, legal, etc.)
    - You have your own audio dataset to train on
    - The default Whisper isn't accurate enough for your use case

    REQUIREMENTS:
    ─────────────
    - GPU recommended (at least 8GB VRAM for "small" model)
    - Can work on CPU but will be VERY slow (hours vs minutes)
    - ~5GB disk space for dataset + model
    """
    from transformers import (
        WhisperProcessor,
        WhisperForConditionalGeneration,
        Seq2SeqTrainingArguments,
        Seq2SeqTrainer,
    )
    from datasets import load_dataset, Audio
    import torch
    import evaluate

    # ══════════════════════════════════════════════
    # STEP 1: Load Pre-trained Model + Processor
    # ══════════════════════════════════════════════
    # Processor = handles audio → features + text → tokens
    # Model = the actual neural network

    MODEL_NAME = "openai/whisper-small"  # 244M parameters
    # Options: "openai/whisper-tiny", "openai/whisper-base",
    #          "openai/whisper-small", "openai/whisper-medium"

    print(f"Loading model: {MODEL_NAME}")
    processor = WhisperProcessor.from_pretrained(MODEL_NAME)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)

    # Tell the model to generate in Hindi
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="hi",   # Hindi
        task="transcribe"  # "transcribe" or "translate" (to English)
    )

    # ══════════════════════════════════════════════
    # STEP 2: Load Dataset
    # ══════════════════════════════════════════════
    # Mozilla Common Voice = free, open-source speech dataset
    # Available in 100+ languages including Hindi
    # Each sample has: audio file + text transcription

    print("Loading Hindi speech dataset...")
    dataset = load_dataset(
        "mozilla-foundation/common_voice_11_0",
        "hi",                    # Hindi language
        split="train[:100]",     # Start with 100 samples (for testing)
        trust_remote_code=True
    )

    # Also load a small validation set
    eval_dataset = load_dataset(
        "mozilla-foundation/common_voice_11_0",
        "hi",
        split="validation[:20]",
        trust_remote_code=True
    )

    # Resample audio to 16kHz (what Whisper expects)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    eval_dataset = eval_dataset.cast_column("audio", Audio(sampling_rate=16000))

    print(f"Training samples: {len(dataset)}")
    print(f"Validation samples: {len(eval_dataset)}")

    # ══════════════════════════════════════════════
    # STEP 3: Preprocess Data
    # ══════════════════════════════════════════════
    # Convert raw audio → mel spectrogram features
    # Convert text → token IDs

    def preprocess(batch):
        """
        This function runs on each sample in the dataset:
        1. Takes the raw audio waveform
        2. Converts it to log-mel spectrogram features (what the model reads)
        3. Takes the text transcription
        4. Converts it to token IDs (numbers the model understands)
        """
        # Extract audio array and sampling rate
        audio = batch["audio"]

        # Convert audio → input features (mel spectrogram)
        # Shape: (80, 3000) — 80 mel bands × 3000 time steps
        batch["input_features"] = processor.feature_extractor(
            audio["array"],
            sampling_rate=audio["sampling_rate"]
        ).input_features[0]

        # Convert text → token IDs
        # "namaste" → [50258, 50266, 50359, ...]
        batch["labels"] = processor.tokenizer(
            batch["sentence"]
        ).input_ids

        return batch

    print("Preprocessing data...")
    dataset = dataset.map(
        preprocess,
        remove_columns=dataset.column_names  # Remove old columns
    )
    eval_dataset = eval_dataset.map(
        preprocess,
        remove_columns=eval_dataset.column_names
    )

    # ══════════════════════════════════════════════
    # STEP 4: Data Collator
    # ══════════════════════════════════════════════
    # Batches samples together and pads them to same length.
    # This is needed because audio clips have different lengths.

    import dataclasses
    from typing import Any, Dict, List, Union

    class WhisperDataCollator:
        """
        Custom data collator that:
        1. Pads input features (audio) to same length in a batch
        2. Pads labels (text tokens) to same length in a batch
        3. Replaces padding tokens with -100 (tells model to ignore them)
        """
        def __init__(self, processor):
            self.processor = processor

        def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
            # Separate inputs and labels
            input_features = [
                {"input_features": f["input_features"]} for f in features
            ]
            label_features = [
                {"input_ids": f["labels"]} for f in features
            ]

            # Pad input features
            batch = self.processor.feature_extractor.pad(
                input_features, return_tensors="pt"
            )

            # Pad labels
            labels_batch = self.processor.tokenizer.pad(
                label_features, return_tensors="pt"
            )

            # Replace padding with -100 (ignored in loss calculation)
            labels = labels_batch["input_ids"].masked_fill(
                labels_batch.attention_mask.ne(1), -100
            )

            # Remove the BOS token (beginning of sequence)
            if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all():
                labels = labels[:, 1:]

            batch["labels"] = labels
            return batch

    data_collator = WhisperDataCollator(processor)

    # ══════════════════════════════════════════════
    # STEP 5: Evaluation Metric
    # ══════════════════════════════════════════════
    # WER = Word Error Rate
    # Lower is better. 0% = perfect, 100% = everything wrong
    # WER = (Substitutions + Insertions + Deletions) / Total Words

    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # Replace -100 with pad token id
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        # Decode predictions and labels back to text
        pred_str = processor.tokenizer.batch_decode(
            pred_ids, skip_special_tokens=True
        )
        label_str = processor.tokenizer.batch_decode(
            label_ids, skip_special_tokens=True
        )

        # Calculate WER
        wer = wer_metric.compute(
            predictions=pred_str,
            references=label_str
        )
        return {"wer": wer}

    # ══════════════════════════════════════════════
    # STEP 6: Training Configuration
    # ══════════════════════════════════════════════

    training_args = Seq2SeqTrainingArguments(
        output_dir="./whisper-hindi-finetuned",  # Save directory
        per_device_train_batch_size=8,            # Samples per batch
        gradient_accumulation_steps=2,            # Effective batch = 8×2 = 16
        learning_rate=1e-5,                       # Learning rate
        warmup_steps=50,                          # Gradual LR warmup
        num_train_epochs=3,                       # Training epochs
        eval_strategy="epoch",                    # Evaluate after each epoch
        save_strategy="epoch",                    # Save after each epoch
        fp16=torch.cuda.is_available(),           # Use FP16 if GPU available
        predict_with_generate=True,               # Generate text for eval
        generation_max_length=225,                # Max output length
        logging_steps=10,                         # Log every 10 steps
        load_best_model_at_end=True,              # Keep best model
        metric_for_best_model="wer",              # Best = lowest WER
        greater_is_better=False,                  # Lower WER = better
        report_to="none",                         # No wandb/mlflow
    )

    # ══════════════════════════════════════════════
    # STEP 7: Create Trainer and START Training
    # ══════════════════════════════════════════════

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor.feature_extractor,
    )

    print("\n" + "=" * 50)
    print(" STARTING FINE-TUNING")
    print("=" * 50)
    print(f" Model: {MODEL_NAME}")
    print(f" Language: Hindi")
    print(f" Training samples: {len(dataset)}")
    print(f" Epochs: {training_args.num_train_epochs}")
    print(f" Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    print("=" * 50 + "\n")

    trainer.train()

    # ══════════════════════════════════════════════
    # STEP 8: Save the Fine-Tuned Model
    # ══════════════════════════════════════════════

    save_path = "./whisper-hindi-finetuned/final"
    model.save_pretrained(save_path)
    processor.save_pretrained(save_path)
    print(f"\nModel saved to: {save_path}")
    print("You can now load it with:")
    print(f'  model = WhisperForConditionalGeneration.from_pretrained("{save_path}")')


def level_3b_use_finetuned_model():
    """
    After fine-tuning, use your custom model to transcribe audio.
    """
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    import torch

    # Load YOUR fine-tuned model
    model_path = "./whisper-hindi-finetuned/final"

    print(f"Loading fine-tuned model from: {model_path}")
    processor = WhisperProcessor.from_pretrained(model_path)
    model = WhisperForConditionalGeneration.from_pretrained(model_path)

    # Load and process audio
    import whisper
    audio = whisper.load_audio("test_audio.wav")
    audio = whisper.pad_or_trim(audio)

    # Convert to mel spectrogram
    mel = whisper.log_mel_spectrogram(audio)

    # Or use the processor directly:
    input_features = processor(
        audio,
        sampling_rate=16000,
        return_tensors="pt"
    ).input_features

    # Generate transcription
    with torch.no_grad():
        predicted_ids = model.generate(input_features)

    # Decode tokens → text
    transcription = processor.batch_decode(
        predicted_ids,
        skip_special_tokens=True
    )[0]

    print(f"\nTranscription: {transcription}")


# ═══════════════════════════════════════════════════════════════════════════
# BONUS: Generate Subtitles (SRT file)
# ═══════════════════════════════════════════════════════════════════════════

def bonus_generate_subtitles():
    """
    Generate an SRT subtitle file from any audio/video.
    You can upload the .srt to YouTube, VLC, etc.
    """
    import whisper

    model = whisper.load_model("base")
    result = model.transcribe("test_audio.wav")

    # Write SRT file
    output_file = "subtitles.srt"
    with open(output_file, "w", encoding="utf-8") as f:
        for i, segment in enumerate(result["segments"], 1):
            # SRT format requires HH:MM:SS,mmm timestamps
            start = segment["start"]
            end = segment["end"]

            start_srt = (
                f"{int(start//3600):02d}:{int((start%3600)//60):02d}"
                f":{int(start%60):02d},{int((start%1)*1000):03d}"
            )
            end_srt = (
                f"{int(end//3600):02d}:{int((end%3600)//60):02d}"
                f":{int(end%60):02d},{int((end%1)*1000):03d}"
            )

            f.write(f"{i}\n")
            f.write(f"{start_srt} --> {end_srt}\n")
            f.write(f"{segment['text'].strip()}\n\n")

    print(f"Subtitles saved to: {output_file}")


# ═══════════════════════════════════════════════════════════════════════════
# BONUS: Translate Any Language Audio → English Text
# ═══════════════════════════════════════════════════════════════════════════

def bonus_translate_to_english():
    """
    Whisper can also TRANSLATE audio from any language to English.
    Give it Hindi audio → get English text.
    """
    import whisper

    model = whisper.load_model("base")

    # task="translate" converts any language → English
    result = model.transcribe(
        "hindi_audio.mp3",
        task="translate"  # ← This is the magic parameter
    )

    print(f"Original language: {result['language']}")
    print(f"English translation: {result['text']}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN — Run any level
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════╗
    ║     SPEECH-TO-TEXT: COMPLETE PROJECT             ║
    ╠══════════════════════════════════════════════════╣
    ║  1. Record test audio (from mic)                ║
    ║  2. Level 1: Basic transcription                ║
    ║  3. Level 2: Real-time mic transcription        ║
    ║  4. Level 3: Fine-tune Whisper (advanced)       ║
    ║  5. Bonus: Generate subtitles (.srt)            ║
    ║  6. Bonus: Translate audio → English            ║
    ╚══════════════════════════════════════════════════╝
    """)

    choice = input("Pick a number (1-6): ").strip()

    if choice == "1":
        level_1b_record_test_audio()
    elif choice == "2":
        level_1_basic_transcription()
    elif choice == "3":
        level_2_realtime_transcription()
    elif choice == "4":
        level_3_finetune_whisper()
    elif choice == "5":
        bonus_generate_subtitles()
    elif choice == "6":
        bonus_translate_to_english()
    else:
        print("Invalid choice. Pick 1-6.")
