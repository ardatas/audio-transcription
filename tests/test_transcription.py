"""Offline tests use real audio decoding and a fake model. No weights are downloaded."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch
import wave

import numpy as np

from transcription import MAX_BYTES, SAMPLE_RATE, Transcriber, TranscriptionError, decode_audio


def write_wav(path: Path, seconds=0.25, rate=22_050, channels=2, silent=False):
    time = np.arange(round(seconds * rate)) / rate
    samples = np.zeros(time.size) if silent else 0.2 * np.sin(2 * np.pi * 440 * time)
    pcm = (samples * 32767).astype("<i2")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(channels)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(np.repeat(pcm[:, None], channels, axis=1).tobytes())


@unittest.skipUnless(shutil.which("ffmpeg"), "Install FFmpeg to run the decoder tests")
class AudioTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.audio = Path(self.directory.name) / "sample.wav"
        write_wav(self.audio)

    def test_resamples_and_downmixes_stereo_audio(self):
        samples = decode_audio(self.audio)
        self.assertEqual(samples.ndim, 1)
        self.assertEqual(samples.dtype, np.float32)
        self.assertAlmostEqual(samples.size / SAMPLE_RATE, 0.25, places=3)
        self.assertGreater(np.max(np.abs(samples)), 0.1)

    def test_decodes_mp3(self):
        mp3 = self.audio.with_suffix(".mp3")
        subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(self.audio), str(mp3)], check=True,
        )
        self.assertGreater(decode_audio(mp3).size, 0)

    def test_rejects_missing_upload(self):
        for value in [None, "", self.audio.with_name("missing.wav")]:
            with self.subTest(value=value), self.assertRaises(TranscriptionError):
                decode_audio(value)

    def test_rejects_empty_and_corrupt_files(self):
        for data in [b"", b"this is not an audio recording"]:
            with self.subTest(data=data):
                self.audio.write_bytes(data)
                with self.assertRaises(TranscriptionError):
                    decode_audio(self.audio)

    def test_rejects_silence(self):
        write_wav(self.audio, silent=True)
        with self.assertRaisesRegex(TranscriptionError, "silent"):
            decode_audio(self.audio)

    def test_accepts_exactly_thirty_seconds(self):
        write_wav(self.audio, seconds=30, rate=SAMPLE_RATE, channels=1)
        self.assertEqual(decode_audio(self.audio).size, SAMPLE_RATE * 30)

    def test_rejects_long_audio_instead_of_truncating(self):
        write_wav(self.audio, seconds=31)
        with self.assertRaisesRegex(TranscriptionError, "longer than 30"):
            decode_audio(self.audio)

    def test_rejects_large_file_before_decoding(self):
        with self.audio.open("wb") as output:
            output.truncate(MAX_BYTES + 1)
        with patch("transcription.subprocess.run") as run:
            with self.assertRaisesRegex(TranscriptionError, "20 MiB"):
                decode_audio(self.audio)
            run.assert_not_called()

    def test_missing_decoder_has_actionable_error(self):
        with patch("transcription.subprocess.run", side_effect=FileNotFoundError):
            with self.assertRaisesRegex(TranscriptionError, "FFmpeg is missing"):
                decode_audio(self.audio)

    def test_decoder_timeout_has_actionable_error(self):
        with patch("transcription.subprocess.run", side_effect=subprocess.TimeoutExpired("ffmpeg", 20)):
            with self.assertRaisesRegex(TranscriptionError, "too long"):
                decode_audio(self.audio)

    def test_reuses_model_and_resets_language_to_auto(self):
        model = Mock(return_value={"text": "  Example transcript.  "})
        with patch("transcription.create_pipeline", return_value=model) as load:
            service = Transcriber()
            self.assertEqual(service.transcribe(self.audio, "German"), "Example transcript.")
            service.transcribe(self.audio)
            load.assert_called_once()
        first, second = model.call_args_list
        self.assertEqual(first.kwargs["generate_kwargs"], {"task": "transcribe", "language": "german"})
        self.assertEqual(second.kwargs["generate_kwargs"], {"task": "transcribe", "language": None})
        self.assertEqual(first.args[0]["sampling_rate"], SAMPLE_RATE)

    def test_invalid_input_does_not_load_model(self):
        with patch("transcription.create_pipeline") as load:
            for audio, language in [(None, "English"), (self.audio, "invalid")]:
                with self.assertRaises(TranscriptionError):
                    Transcriber().transcribe(audio, language)
            load.assert_not_called()

    def test_failed_model_download_can_be_retried(self):
        model = Mock(return_value={"text": "Success"})
        with patch("transcription.create_pipeline", side_effect=[OSError("offline"), model]):
            service = Transcriber()
            with self.assertRaisesRegex(TranscriptionError, "Could not load"):
                service.transcribe(self.audio)
            self.assertEqual(service.transcribe(self.audio), "Success")

    def test_inference_failure_and_empty_output_are_reported(self):
        for model in [Mock(side_effect=RuntimeError("out of memory")), Mock(return_value={"text": " "})]:
            with patch("transcription.create_pipeline", return_value=model):
                with self.assertRaises(TranscriptionError):
                    Transcriber().transcribe(self.audio)

    def test_cli_missing_file_returns_error_status(self):
        import sys

        result = subprocess.run(
            [sys.executable, "transcribe.py", str(self.audio.with_name("missing.wav"))],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("recording is missing", result.stderr)
        self.assertEqual(result.stdout, "")


class InterfaceTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("ffmpeg"), "Install FFmpeg to run the upload test")
    def test_upload_and_transcribe_through_http_api(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        import gradio as gr
        from app import build_app

        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "speech.wav"
            write_wav(audio)
            model = Mock(return_value={"text": "Example transcript"})
            demo = build_app()
            mounted = gr.mount_gradio_app(FastAPI(), demo, path="/", max_file_size=MAX_BYTES)
            with patch("transcription.create_pipeline", return_value=model), TestClient(mounted) as client:
                with audio.open("rb") as recording:
                    upload = client.post("/gradio_api/upload", files={"files": (audio.name, recording, "audio/wav")})
                self.assertEqual(upload.status_code, 200)
                request = client.post("/gradio_api/call/transcribe", json={
                    "data": [{"path": upload.json()[0], "meta": {"_type": "gradio.FileData"}}, "English"],
                })
                self.assertEqual(request.status_code, 200)
                event_id = request.json()["event_id"]
                result = client.get(f"/gradio_api/call/transcribe/{event_id}")
                self.assertEqual(result.status_code, 200)
                self.assertIn("event: complete", result.text)
                self.assertIn("Example transcript", result.text)
                model.assert_called_once()

    def test_browser_page_loads_and_errors_clear_old_transcript(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        import gradio as gr
        from app import build_app

        service = Mock()
        service.transcribe.side_effect = ["Example transcript", TranscriptionError("Upload audio first.")]
        demo = build_app(service)
        mounted = gr.mount_gradio_app(FastAPI(), demo, path="/", max_file_size=MAX_BYTES)
        with TestClient(mounted) as client:
            self.assertEqual(client.get("/").status_code, 200)
            config = client.get("/config").json()
            self.assertEqual(config["title"], "Audio to transcript")
        callback = next(fn.fn for fn in demo.fns.values() if fn.api_name == "transcribe")
        success = list(callback("sample.wav", "English"))
        failure = list(callback(None, "English"))
        self.assertEqual(success[-1][0], "Example transcript")
        self.assertEqual(failure[0][0], "")
        self.assertEqual(failure[-1], ("", "Upload audio first."))


@unittest.skipUnless(os.environ.get("RUN_MODEL_TESTS") == "1", "Opt-in: downloads Whisper Tiny")
class RealModelTests(unittest.TestCase):
    def test_transcribes_known_speech(self):
        # Supply your own short recording and a phrase it contains. The test does
        # not send audio anywhere; only model files are fetched when not cached.
        audio = os.environ.get("TEST_AUDIO")
        phrase = os.environ.get("EXPECTED_PHRASE")
        self.assertTrue(audio and phrase, "Set TEST_AUDIO and EXPECTED_PHRASE")
        text = Transcriber().transcribe(audio, "English")
        self.assertIn(phrase.casefold(), text.casefold())


if __name__ == "__main__":
    unittest.main()
