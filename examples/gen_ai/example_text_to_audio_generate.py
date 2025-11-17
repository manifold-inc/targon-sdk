import io
import targon

image = targon.Image.debian_slim(python_version="3.12").pip_install(
    "chatterbox-tts==0.1.1", "fastapi[standard]"
)

with image.imports():
    import torchaudio as ta
    from chatterbox.tts import ChatterboxTTS
    from fastapi.responses import StreamingResponse
    
app = targon.App("example-chatterbox-tts", image=image)


@app.function(resource="h200-medium",timeout=10 * 60)
@targon.fastapi_endpoint(docs=True, method="POST")
def generate_audio(input: dict):
    if "prompt" in input:
        prompt = input["prompt"]
        model = ChatterboxTTS.from_pretrained(device="cuda")

        wav = model.generate(prompt)

        buffer = io.BytesIO()

        ta.save(buffer, wav, model.sr, format="wav")

        buffer.seek(0)

        return StreamingResponse(
            io.BytesIO(buffer.read()),
            media_type="audio/wav",
        )
