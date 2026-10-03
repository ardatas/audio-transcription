---
title: Audio to transcript
emoji: 🎙️
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 6.28.0
python_version: "3.12"
app_file: app.py
models:
  - openai/whisper-tiny
---

# Audio to transcript

Upload a recording, choose its language, and get a transcript using
[Whisper Tiny](https://huggingface.co/openai/whisper-tiny). The first version
accepts one recording at a time, up to 30 seconds and 20 MiB. It supports
automatic language detection, English, and German. Transcripts stay in the
recording's language.

## Run on your Mac

Install Python 3.12 and FFmpeg. With Homebrew and uv installed:

```bash
brew install ffmpeg
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
python app.py
```

Open the local address printed in the terminal, usually http://127.0.0.1:7860.
The first valid transcription downloads Whisper Tiny from Hugging Face. Later
requests reuse the model in memory. Subsequent app starts reuse downloaded
files from the Hugging Face cache. The app does not call a hosted transcription API.

Device selection is automatic: NVIDIA CUDA, Apple MPS, then CPU. To use the CPU
explicitly, including when a GPU operation is unsupported:

```bash
TRANSCRIPTION_DEVICE=cpu python app.py
```

You can also transcribe a file without opening a browser:

```bash
python transcribe.py recordings/hello.wav --language English
```

## How it works

1. Gradio passes the uploaded file's path to `Transcriber.transcribe`.
2. FFmpeg decodes a bounded portion into mono audio at 16 kHz. The app rejects
   recordings over the duration or file-size limit, unreadable files, and silence.
3. Transformers loads `openai/whisper-tiny` on the first valid request.
4. The same pipeline processes later recordings, one at a time.
5. Gradio displays the text with a copy button, or an error explaining what to fix.

`transcription.py` contains the processing logic, `app.py` supplies the interface,
and `transcribe.py` is the command-line entry point. `requirements.in` lists direct
dependencies; `requirements.txt` preserves the exact versions from the original
development environment. `packages.txt` installs FFmpeg on Spaces.

## Verify it

Run offline checks with FFmpeg installed:

```bash
python -m unittest discover -s tests -v
```

These tests decode generated WAV and MP3 files, exercise upload limits and errors,
check model reuse with a fake pipeline, and load the Gradio page. They do not
measure Whisper's transcription quality. The real-model test is skipped by default.

For a real-model check, record a short English sentence and provide a phrase
you expect in its transcript:

```bash
RUN_MODEL_TESTS=1 TEST_AUDIO=recordings/hello.wav \
  EXPECTED_PHRASE="hello world" python -m unittest discover -s tests -v
```

This check needs the model cached locally or internet access to download it.
Also try a noisy recording and compare the result manually with what was spoken.
Whisper can omit words or invent text; the silence check is a simple amplitude
threshold, not a speech detector. The app is intended for short uploaded clips,
not live streaming or speaker identification.

## Deploy to Hugging Face

Create a Gradio Space with a plan and hardware option supported by your account.
Copy the tracked project files into its repository, preserving the README metadata.
CPU hardware is sufficient to try Whisper Tiny; the app selects the Space's
available hardware independently of your Mac. This project does not include the
extra integration required for ZeroGPU.

Spaces installs `requirements.txt` and `packages.txt`, then starts `app.py`.
Confirm that a fresh audio upload returns a transcript at the Space URL.
See the [Gradio Spaces guide](https://huggingface.co/docs/hub/spaces-sdks-gradio).

## Audio storage

With a local server, recordings are processed on your computer. On a Space,
recordings are uploaded to that Space's server. No audio is sent to a model API.
Gradio stores temporary uploads in `.cache/gradio` and checks every five minutes
for files older than five minutes. While the server is running, cleanup normally
takes five to ten minutes. Clicking Clear resets the interface; it does not
immediately delete files. The application does not save a transcript history.
Keep personal recordings in the ignored `recordings/` directory.
