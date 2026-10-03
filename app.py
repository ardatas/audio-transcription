"""Run the Gradio audio-to-text app: python app.py."""

import os

# Keep Gradio telemetry and upload caches local to this project by default.
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("GRADIO_TEMP_DIR", ".cache/gradio")

import gradio as gr

from transcription import LANGUAGES, MAX_BYTES, Transcriber, TranscriptionError


def build_app(transcriber: Transcriber | None = None) -> gr.Blocks:
    service = transcriber if transcriber is not None else Transcriber()

    def transcribe(audio_file: str | None, language: str):
        # Clear a previous transcript before validating a replacement recording.
        yield "", "Transcribing… The first run may need to download the model."
        try:
            text = service.transcribe(audio_file, language)
        except TranscriptionError as exc:
            yield "", str(exc)
            return
        yield text, "Done. Review the transcript for mistakes."

    with gr.Blocks(
        title="Audio to transcript", analytics_enabled=False, delete_cache=(300, 300)
    ) as demo:
        gr.Markdown(
            "# Audio to transcript\n"
            "Upload a short recording and turn speech into text. "
            "**Up to 30 seconds · 20 MiB per file.**"
        )
        audio = gr.Audio(sources=["upload"], type="filepath", label="Your recording")
        language = gr.Dropdown(list(LANGUAGES), value="Auto-detect", label="Spoken language")
        with gr.Row():
            submit = gr.Button("Transcribe", variant="primary")
            clear = gr.Button("Clear")
        status = gr.Textbox(label="Status", value="Ready for a recording.", interactive=False)
        transcript = gr.Textbox(label="Transcript", lines=8, interactive=False, buttons=["copy"])
        gr.Markdown(
            "Whisper Tiny can mishear speech or invent words, especially in noise. "
            "Review the result. Audio is processed on the machine running this app. "
            "Uploaded files are temporary and normally cleaned up within five to ten minutes."
        )
        event = submit.click(
            transcribe, inputs=[audio, language], outputs=[transcript, status],
            api_name="transcribe", concurrency_limit=1,
        )
        clear.click(
            lambda: (None, "Auto-detect", "", "Ready for a recording."),
            outputs=[audio, language, transcript, status], cancels=[event], queue=False,
        )
        audio.change(lambda: ("", "Ready for a recording."), outputs=[transcript, status], queue=False)
    return demo.queue(max_size=8, default_concurrency_limit=1)


if __name__ == "__main__":
    build_app().launch(share=False, max_file_size=MAX_BYTES)
