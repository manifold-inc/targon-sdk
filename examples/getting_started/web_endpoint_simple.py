import targon

# Create an image with FastAPI installed
# Note: Runtime is automatically injected during deployment!
image = targon.Image.debian_slim("3.13").pip_install("fastapi[standard]")


# Create an app
app = targon.App("web-endpoint-simple", image=image)


@app.function()
@targon.fastapi_endpoint(method="GET", docs=True)
def greet(name: str = "World"):
    """GET endpoint with query parameters and docs enabled.
    
    Try visiting:
    - /greet?name=Alice
    - /greet/docs (for interactive documentation)
    """
    return {"greeting": f"Hello, {name}!"}

