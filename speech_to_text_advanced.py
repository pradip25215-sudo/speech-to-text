"""
=============================================================================
 SPEECH-TO-TEXT — ADVANCED MODULE (Level 4)
 Production-grade transcription pipeline + web demo
=============================================================================

 This file is the ADVANCED extension of your speech_to_text_complete.py.
 Levels 1-3 taught the concepts. This file packages them like real software.

 WHAT YOU GET HERE:
 ──────────────────
   1. AdvancedTranscriber class   → one clean object that does everything
   2. Audio preprocessing         → mono, 16kHz resample, normalization
   3. Word-level timestamps       → exact time of every single word
   4. Confidence scores           → how sure the model is (per segment)
   5. Long-audio handling         → 2-hour podcasts work fine
   6. Speaker diarization         → "who spoke when" (Speaker 1 / Speaker 2)
   7. Multi-format export         → .srt .vtt .json .csv .txt
   8. Streamlit web app           → drag-drop a file, see the transcript

 WHY THIS MATTERS FOR YOUR PORTFOLIO:
 ────────────────────────────────────
 Anyone can call model.transcribe(). What separates a junior demo from a
 real engineering project is: clean class design, preprocessing, error
 handling, multiple output formats, and a usable interface. That's this file.

 SETUP:
 ──────
   pip install faster-whisper        # fast ASR engine
   pip install librosa soundfile     # audio loading + preprocessing
   pip install pandas                # CSV export
   pip install streamlit             # web app (optional)
   pip install pyannote.audio        # speaker diarization (optional)
   # ffmpeg required:  sudo apt install ffmpeg   (or brew install ffmpeg)

 RUN:
 ────
   python speech_to_text_advanced.py            # CLI demo on a file
   streamlit run speech_to_text_advanced.py     # launches the web app
=============================================================================
"""

import os
import json
from dataclasses import dataclass, field, asdict
from typing import List, Optional


# ═══════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════
# Instead of passing around messy dictionaries, we define clean typed objects.
# This is what makes the rest of the code readable and bug-resistant.

@dataclass
class Word:
    """A single recognized word with its exact timing and confidence."""
    text: str
    start: float        # seconds
    end: float          # seconds
    probability: float  # 0.0 - 1.0, how confident the model is


@dataclass
class Segment:
    """A chunk of speech (roughly a sentence)."""
    text: str
    start: float
    end: float
    confidence: float            # average confidence for the segment
    speaker: Optional[str] = None  # set later by diarization, e.g. "SPEAKER_00"
    words: List[Word] = field(default_factory=list)


@dataclass
class Transcript:
    """The full result of transcribing one audio file."""
    text: str
    language: str
    duration: float
    segments: List[Segment] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════
# CORE: ADVANCED TRANSCRIBER
# ═══════════════════════════════════════════════════════════════════════════

class AdvancedTranscriber:
    """
    A reusable transcription engine.

    Example
    -------
    >>> t = AdvancedTranscriber(model_size="base")
    >>> result = t.transcribe("meeting.mp3")
    >>> print(result.text)
    >>> t.export(result, "meeting", formats=["srt", "json", "csv"])
    """

    def __init__(self, model_size: str = "base", device: str = "auto",
                 compute_type: str = "auto"):
        """
        Parameters
        ----------
        model_size   : tiny | base | small | medium | large-v3 | turbo
        device       : "cpu", "cuda", or "auto" (picks GPU if available)
        compute_type : "int8" (fast CPU), "float16" (GPU), or "auto"
        """
        from faster_whisper import WhisperModel

        # Auto-pick the best device/precision so the same code runs
        # on a laptop CPU and on the IIIT-D GPU server without edits.
        if device == "auto" or compute_type == "auto":
            try:
                import torch
                has_gpu = torch.cuda.is_available()
            except ImportError:
                has_gpu = False
            if device == "auto":
                device = "cuda" if has_gpu else "cpu"
            if compute_type == "auto":
                compute_type = "float16" if has_gpu else "int8"

        print(f"[AdvancedTranscriber] Loading '{model_size}' on "
              f"{device} ({compute_type})...")
        self.model = WhisperModel(model_size, device=device,
                                  compute_type=compute_type)
        self.model_size = model_size
        print("[AdvancedTranscriber] Ready.")

    # ───────────────────────────────────────────────────────────────────────
    # 1. AUDIO PREPROCESSING
    # ───────────────────────────────────────────────────────────────────────
    @staticmethod
    def preprocess_audio(path: str, target_sr: int = 16000):
        """
        Load ANY audio file and clean it for the model:
          - convert to mono (single channel)
          - resample to 16 kHz (Whisper's native rate)
          - peak-normalize the volume (quiet recordings get boosted)

        Returns a 1-D float32 numpy array. Garbage-in/garbage-out is the
        #1 cause of bad transcripts, so this step genuinely improves accuracy.
        """
        import librosa
        import numpy as np

        # librosa loads, downmixes to mono, and resamples in one call
        audio, _ = librosa.load(path, sr=target_sr, mono=True)

        # Peak normalization: scale so the loudest sample is ~1.0
        peak = np.abs(audio).max()
        if peak > 0:
            audio = audio / peak * 0.95

        return audio.astype(np.float32)

    # ───────────────────────────────────────────────────────────────────────
    # 2. TRANSCRIBE (word timestamps + confidence + long-audio)
    # ───────────────────────────────────────────────────────────────────────
    def transcribe(self, path: str, language: Optional[str] = None,
                   preprocess: bool = True, translate: bool = False) -> Transcript:
        """
        Transcribe an audio file of ANY length.

        faster-whisper streams the audio internally, so a 2-hour file uses
        roughly the same memory as a 10-second one. We just collect the
        results into our clean Transcript object.

        Parameters
        ----------
        language   : "en", "hi", ... or None to auto-detect
        preprocess : run preprocess_audio() first (recommended)
        translate  : if True, translate any language → English
        """
        import math

        audio_input = self.preprocess_audio(path) if preprocess else path

        segments_iter, info = self.model.transcribe(
            audio_input,
            language=language,
            task="translate" if translate else "transcribe",
            beam_size=5,                # wider beam = more accurate
            word_timestamps=True,       # get per-word timing
            vad_filter=True,            # Voice Activity Detection: skip silence
            vad_parameters={"min_silence_duration_ms": 500},
        )

        segments: List[Segment] = []
        full_text_parts: List[str] = []

        for seg in segments_iter:
            # avg_logprob is a log-probability; exp() turns it back into 0-1
            confidence = math.exp(seg.avg_logprob) if seg.avg_logprob else 0.0

            words = []
            if seg.words:
                for w in seg.words:
                    words.append(Word(
                        text=w.word.strip(),
                        start=round(w.start, 3),
                        end=round(w.end, 3),
                        probability=round(w.probability, 3),
                    ))

            segments.append(Segment(
                text=seg.text.strip(),
                start=round(seg.start, 3),
                end=round(seg.end, 3),
                confidence=round(confidence, 3),
                words=words,
            ))
            full_text_parts.append(seg.text.strip())

        return Transcript(
            text=" ".join(full_text_parts),
            language=info.language,
            duration=round(info.duration, 2),
            segments=segments,
        )

    # ───────────────────────────────────────────────────────────────────────
    # 3. SPEAKER DIARIZATION ("who spoke when")
    # ───────────────────────────────────────────────────────────────────────
    def add_speakers(self, transcript: Transcript, audio_path: str,
                     hf_token: Optional[str] = None) -> Transcript:
        """
        Label each segment with a speaker (SPEAKER_00, SPEAKER_01, ...).

        Uses pyannote.audio. This needs a (free) Hugging Face token because
        the model is gated:
          1. Create a token at huggingface.co/settings/tokens
          2. Accept the terms at huggingface.co/pyannote/speaker-diarization-3.1
          3. Pass the token here or set env var HF_TOKEN

        If pyannote isn't installed or no token is given, this is skipped
        gracefully and the transcript is returned unchanged.
        """
        try:
            from pyannote.audio import Pipeline
        except ImportError:
            print("[diarization] pyannote.audio not installed — skipping.")
            return transcript

        token = hf_token or os.environ.get("HF_TOKEN")
        if not token:
            print("[diarization] No HF token provided — skipping.")
            return transcript

        print("[diarization] Running speaker diarization...")
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=token,
        )
        diarization = pipeline(audio_path)

        # Build a list of (start, end, speaker) turns
        turns = [(turn.start, turn.end, speaker)
                 for turn, _, speaker in diarization.itertracks(yield_label=True)]

        # For each transcript segment, find which speaker overlaps it most
        for seg in transcript.segments:
            best_speaker, best_overlap = None, 0.0
            for t_start, t_end, speaker in turns:
                overlap = min(seg.end, t_end) - max(seg.start, t_start)
                if overlap > best_overlap:
                    best_overlap, best_speaker = overlap, speaker
            seg.speaker = best_speaker

        return transcript

    # ───────────────────────────────────────────────────────────────────────
    # 4. MULTI-FORMAT EXPORT
    # ───────────────────────────────────────────────────────────────────────
    def export(self, transcript: Transcript, basename: str,
               formats: List[str] = ("txt", "srt", "json")) -> List[str]:
        """
        Save the transcript in several formats at once.
        Returns the list of files written.

        Formats: txt, srt, vtt, json, csv
        """
        written = []
        for fmt in formats:
            out = f"{basename}.{fmt}"
            writer = getattr(self, f"_write_{fmt}", None)
            if writer is None:
                print(f"[export] Unknown format '{fmt}' — skipping.")
                continue
            writer(transcript, out)
            written.append(out)
            print(f"[export] Wrote {out}")
        return written

    # --- individual format writers ---

    @staticmethod
    def _fmt_time(seconds: float, sep: str = ",") -> str:
        """Seconds → HH:MM:SS,mmm  (SRT uses ',', VTT uses '.')"""
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"

    def _write_txt(self, t: Transcript, path: str):
        with open(path, "w", encoding="utf-8") as f:
            for seg in t.segments:
                prefix = f"[{seg.speaker}] " if seg.speaker else ""
                f.write(f"{prefix}{seg.text}\n")

    def _write_srt(self, t: Transcript, path: str):
        with open(path, "w", encoding="utf-8") as f:
            for i, seg in enumerate(t.segments, 1):
                prefix = f"{seg.speaker}: " if seg.speaker else ""
                f.write(f"{i}\n")
                f.write(f"{self._fmt_time(seg.start)} --> "
                        f"{self._fmt_time(seg.end)}\n")
                f.write(f"{prefix}{seg.text}\n\n")

    def _write_vtt(self, t: Transcript, path: str):
        with open(path, "w", encoding="utf-8") as f:
            f.write("WEBVTT\n\n")
            for seg in t.segments:
                prefix = f"{seg.speaker}: " if seg.speaker else ""
                f.write(f"{self._fmt_time(seg.start, '.')} --> "
                        f"{self._fmt_time(seg.end, '.')}\n")
                f.write(f"{prefix}{seg.text}\n\n")

    def _write_json(self, t: Transcript, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(t), f, ensure_ascii=False, indent=2)

    def _write_csv(self, t: Transcript, path: str):
        import pandas as pd
        rows = [{
            "start": seg.start,
            "end": seg.end,
            "speaker": seg.speaker or "",
            "confidence": seg.confidence,
            "text": seg.text,
        } for seg in t.segments]
        pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8")


# ═══════════════════════════════════════════════════════════════════════════
# CLI DEMO
# ═══════════════════════════════════════════════════════════════════════════

def run_cli():
    """Command-line demo: transcribe a file and export everything."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Advanced speech-to-text transcription.")
    parser.add_argument("audio", help="path to audio/video file")
    parser.add_argument("--model", default="base",
                        help="tiny|base|small|medium|large-v3|turbo")
    parser.add_argument("--language", default=None,
                        help="force a language, e.g. 'en' or 'hi'")
    parser.add_argument("--translate", action="store_true",
                        help="translate to English")
    parser.add_argument("--speakers", action="store_true",
                        help="run speaker diarization (needs HF_TOKEN)")
    parser.add_argument("--formats", default="txt,srt,json",
                        help="comma-separated: txt,srt,vtt,json,csv")
    args = parser.parse_args()

    transcriber = AdvancedTranscriber(model_size=args.model)
    result = transcriber.transcribe(
        args.audio, language=args.language, translate=args.translate)

    if args.speakers:
        result = transcriber.add_speakers(result, args.audio)

    print("\n" + "=" * 60)
    print(f"LANGUAGE : {result.language}   DURATION: {result.duration}s")
    print("=" * 60)
    for seg in result.segments:
        tag = f"[{seg.speaker}] " if seg.speaker else ""
        print(f"  {seg.start:6.1f}s {tag}{seg.text}  "
              f"(conf {seg.confidence:.2f})")

    basename = os.path.splitext(os.path.basename(args.audio))[0]
    transcriber.export(result, basename, formats=args.formats.split(","))


# ═══════════════════════════════════════════════════════════════════════════
# STREAMLIT WEB APP
# ═══════════════════════════════════════════════════════════════════════════
# Run with:  streamlit run speech_to_text_advanced.py
# This gives you a drag-and-drop web UI — perfect for a portfolio demo / GIF.

def run_streamlit():
    import streamlit as st
    import tempfile

    st.set_page_config(page_title="Advanced Speech-to-Text", page_icon="🎙️")
    st.title("🎙️ Advanced Speech-to-Text")
    st.caption("faster-whisper · word timestamps · confidence · diarization")

    col1, col2 = st.columns(2)
    with col1:
        model_size = st.selectbox(
            "Model", ["tiny", "base", "small", "medium", "large-v3"], index=1)
    with col2:
        language = st.selectbox(
            "Language", ["auto", "en", "hi", "es", "fr", "de"], index=0)

    translate = st.checkbox("Translate to English")
    uploaded = st.file_uploader(
        "Upload audio", type=["mp3", "wav", "m4a", "flac", "ogg", "webm"])

    if uploaded and st.button("Transcribe", type="primary"):
        # Save the upload to a temp file the model can read
        suffix = os.path.splitext(uploaded.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        with st.spinner(f"Loading '{model_size}' and transcribing..."):
            # Cache the model so it isn't reloaded on every rerun
            @st.cache_resource
            def get_model(size):
                return AdvancedTranscriber(model_size=size)

            transcriber = get_model(model_size)
            result = transcriber.transcribe(
                tmp_path,
                language=None if language == "auto" else language,
                translate=translate,
            )

        st.success(f"Detected: {result.language} · {result.duration}s")
        st.subheader("Full transcript")
        st.write(result.text)

        st.subheader("Segments")
        for seg in result.segments:
            st.markdown(
                f"`{seg.start:6.1f}s` **conf {seg.confidence:.2f}** — {seg.text}")

        # Download buttons for each format
        st.subheader("Download")
        base = os.path.splitext(uploaded.name)[0]
        for fmt in ["txt", "srt", "vtt", "json", "csv"]:
            transcriber.export(result, os.path.join(tempfile.gettempdir(), base),
                               formats=[fmt])
            fpath = os.path.join(tempfile.gettempdir(), f"{base}.{fmt}")
            with open(fpath, "rb") as f:
                st.download_button(f"Download .{fmt}", f,
                                   file_name=f"{base}.{fmt}", key=fmt)

        os.unlink(tmp_path)


# ═══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════
# Detect whether we're running under `streamlit run` or plain `python`.

def _is_streamlit():
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


if _is_streamlit():
    run_streamlit()
elif __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        run_cli()
    else:
        print("Usage:")
        print("  python speech_to_text_advanced.py <audio_file> [--model base]")
        print("  streamlit run speech_to_text_advanced.py")
