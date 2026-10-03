"""Transcribe a local recording without starting a web server."""

import argparse
import sys

from transcription import LANGUAGES, Transcriber, TranscriptionError


def main() -> int:
    parser = argparse.ArgumentParser(description="Transcribe up to 30 seconds with Whisper Tiny.")
    parser.add_argument("audio", help="Path to an audio recording, such as a WAV or MP3 file")
    parser.add_argument("--language", choices=LANGUAGES, default="Auto-detect")
    args = parser.parse_args()
    try:
        print(Transcriber().transcribe(args.audio, args.language))
    except TranscriptionError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
