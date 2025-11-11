import targon

vllm_image = (
    targon.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04",
        add_python="3.12"
    )
    .pip_install(
        "vllm==0.10.2",
        "torch==2.8.0",
        "huggingface_hub==0.35.0",
        "fastapi[standard]",
        "hf_transfer"
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)
# hf-transfer
app = targon.App("vllm-inference", image=vllm_image)


@app.function(resource="h200-small",max_replicas=1)
@targon.web_server(port=8080)
def serve():
    import subprocess
    MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    MODEL_REVISION = "main"

    # Fast boot & small GPU use
    FAST_BOOT = True
    N_GPU = 1

    cmd = [
        "vllm",
        "serve",
        "--uvicorn-log-level=info",
        MODEL_NAME,
        "--revision",
        MODEL_REVISION,
        "--served-model-name",
        MODEL_NAME,
        "llm",
        "--host",
        "0.0.0.0",
        "--port",
        "8080",
    ]

    cmd += ["--enforce-eager" if FAST_BOOT else "--no-enforce-eager"]

    cmd += ["--tensor-parallel-size", str(N_GPU)]

    subprocess.Popen(" ".join(cmd), shell=True)
