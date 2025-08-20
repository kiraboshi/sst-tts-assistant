## Silero Voice Activation Sample

This app is a microphone-driven demo that uses Silero VAD to detect speech segments and Whisper (via Hugging Face Transformers) to transcribe them. It supports a simple voice UX with a wake phrase to enter "listen" mode and a stop phrase to exit listen mode. Optionally, it can call an OpenAI-compatible assistant to generate brief responses.

### Core flow
- Audio frames are captured from the default microphone using `sounddevice`.
- `SileroVoiceActivation` (ONNX) runs streaming VAD to detect speech boundaries.
- When a speech segment ends, the buffered audio is sent to a Whisper ASR pipeline.
- Transcribed text controls an internal state machine:
  - Wake phrase (default: "hello computer") switches to listen mode; a beep plays.
  - Say "reset" at any time in listen mode to clear the transcript and assistant history.
  - Stop phrase (default: "thank you") exits listen mode; a beep plays.
- If an API key is configured, `OpenAIAssistant` can optionally answer user utterances in one sentence.
  - New: if enabled, assistant replies are spoken via Coqui TTS (XTTS v2 voice cloning).

Files of interest:
- `run.py`: minimal operational loop that orchestrates services and handles I/O.
- `voice_activation.py`: Silero VAD wrapper for streaming speech detection.
- `openai_client.py`: thin wrapper around the OpenAI (and OpenRouter) Chat Completions API.
- `asr_service.py`: lazy-loading Whisper ASR service (Hugging Face Transformers).
- `tts_service.py`: lazy-loading Coqui TTS (XTTS v2) service and playback helper.
- `controller.py`: `ConversationController` managing wake/listen/reset/stop flow and assistant calls.
- `audio_utils.py`: audio helpers (beep generation, simple linear resampling).
- `regex_utils.py`: robust phrase-matching regex compiler for wake/stop/reset.
- `warmup.py`: model warm-up orchestrator for TTS and ASR.

## Requirements
- Python 3.10+
- A working microphone
- OS: Windows, macOS, or Linux

Install dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Windows (PowerShell):

```powershell
py -3.10 -m venv venv
./venv/Scripts/Activate.ps1
pip install -r requirements.txt
```

GPU (optional): install a CUDA-enabled PyTorch matching your system from the official instructions first, then install the rest. See the PyTorch install guide: [Get Started with PyTorch](https://pytorch.org/get-started/locally/).

## Running

Basic run with defaults:

```bash
python run.py
```

You should see:
- A "ready" beep at start.
- Console logs like "[VAD] speech_start" / "[VAD] speech_end" and the transcribed text.
- Prompts explaining the wake/stop phrases.

### Text-to-Speech (optional)

- Enable TTS playback of assistant replies using Coqui TTS (XTTS v2 voice cloning).
- Provide a short reference voice clip (e.g., `test.wav` in the repo) to clone a voice.

Examples:

```bash
# Use default test.wav if present
python run.py --tts

# Explicit speaker reference and language
python run.py --tts --tts-speaker-wav path/to/voice.wav --tts-language en

# Choose a specific model id
python run.py --tts --tts-model-id tts_models/multilingual/multi-dataset/xtts_v2
```

Notes:
- TTS loads lazily on first use; GPU is used if available, else CPU.
- Audio plays through the system default output device via `sounddevice`.
- If you hear feedback loops, use headphones.

### Voice commands
- Wake phrase: "hello computer" (configurable)
- Stop phrase: "thank you" (configurable)
- Reset keyword: "reset" (clears ongoing transcript and assistant history while staying in listen mode)

## CLI options

All flags are provided by `run.py`.

- `--start-words`, `--start`:
  - Set the wake phrase.
  - Example: `--start "ok system"`

- `--stop-words`, `--stop`:
  - Set the stop phrase.
  - Example: `--stop "that is all"`

- `--asr-model`:
  - Hugging Face model ID for ASR. Defaults depend on device: a lighter model on CPU, a larger model on GPU.
  - Examples: `openai/whisper-tiny`, `openai/whisper-small`, `openai/whisper-large-v3-turbo`

- `--asr-local-only`:
  - Use only locally cached models (no network download). Useful for offline usage.

- `--eager-asr`:
  - Load the ASR pipeline immediately at startup (original behavior). Without this, ASR loads lazily the first time it’s needed.

- `--asr-warmup`:
  - Start loading the ASR pipeline in a background thread. This makes startup instant while the model warms up.

Examples:

```bash
# Instant startup; ASR loads in background
python run.py --asr-warmup

# Force small model for faster load on CPU
python run.py --asr-model openai/whisper-tiny

# Use local cache only (no downloads)
python run.py --asr-local-only
```

## Environment variables

- Assistant/LLM (optional):
  - `OPENAI_API_KEY`: enables OpenAI API usage (or set `OPENROUTER_API_KEY` to route via OpenRouter).
  - Optional: `OPENAI_BASE_URL`, `OPENAI_MODEL_ID`, `OPENAI_MAX_TOKENS`, `OPENAI_TEMPERATURE`.
  - `ASSISTANT_LLM_CHECKS=1`: enables fuzzy wake/stop checks via the assistant for borderline cases.

- ASR model selection:
  - `ASR_MODEL_ID`: explicit model ID (same as `--asr-model`).
  - `ASR_MODEL_ID_CPU`: default model on CPU (default: `openai/whisper-small`).
  - `ASR_MODEL_ID_GPU`: default model on GPU (default: `openai/whisper-large-v3-turbo`).
  - `ASR_LOCAL_ONLY=1`: only use locally cached models (same as `--asr-local-only`).

- TTS (optional):
  - `TTS_ENABLED=1`: enable TTS playback (same as `--tts`).
  - `TTS_SPEAKER_WAV`: path to the reference speaker audio (default: `test.wav` if present).
  - `TTS_LANGUAGE`: language code, e.g., `en`.
  - `TTS_MODEL_ID`: Coqui TTS model id (default: `tts_models/multilingual/multi-dataset/xtts_v2`).

## How it works (brief)

1. `sounddevice` captures mono float32 frames at 16 kHz (default frame: 32 ms).
2. `SileroVoiceActivation` processes frames via `silero_vad.VADIterator`, exposing `speech_start` and `speech_end`.
3. On `speech_end`, the buffered audio segment is transcribed by `ASRService` (Whisper transformers pipeline).
4. `ConversationController` manages modes:
   - Idle: waits for the wake phrase (regex-based via `regex_utils`). If enabled, the assistant does fuzzy checks.
   - Listen: accumulates text until the stop phrase (or assistant fuzzy stop), then prints and beeps.
5. `OpenAIAssistant` optionally answers each listen-mode utterance. If TTS is enabled, `TTSService` speaks it.

Timing logs like `[ASR] Loaded in …` and `[VAD] Initialized in …` help diagnose startup time.

## Troubleshooting

- Slow startup / first run:
  - Use `--asr-warmup` (background load) or choose a smaller model (e.g., `openai/whisper-tiny`).
  - First run may download models; use `--asr-local-only` to skip downloads if you already cached them.

- No audio input or beeps:
  - Ensure mic permissions and the default input device are correct.
  - On Windows, `sounddevice` may require the latest Microsoft Visual C++ Redistributable.

- GPU not used:
  - Install the correct CUDA-enabled PyTorch. See the official guide: [Get Started with PyTorch](https://pytorch.org/get-started/locally/).

- OpenAI request errors:
  - Verify `OPENAI_API_KEY` (or `OPENROUTER_API_KEY`) and network/proxy settings.
  - Set `OPENAI_ENABLED=0` (or unset keys) to disable the assistant if you don’t need it.

## Credits

- Silero VAD: lightweight, streaming voice activity detection.
- Whisper (Hugging Face Transformers).
- OpenAI / OpenRouter SDKs for Chat Completions (optional).


