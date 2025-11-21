import targon

app = targon.App("getting-started", image=targon.Image.debian_slim("3.12"))

@app.function()
def hello(message:str)->dict[str,str]:
    return {"message": message, "status": "success"}

@app.local_entrypoint()
def main(message:str):
    """Simple entrypoint that prints a message."""
    return hello.remote(message)