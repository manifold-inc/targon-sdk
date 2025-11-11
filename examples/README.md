# Targon SDK Examples

This directory contains examples demonstrating various features of the Targon SDK.

## Web Endpoints (FastAPI)

### `basic_web.py` - FastAPI Endpoint Basics
Comprehensive example showing all FastAPI endpoint features:
- Simple GET endpoint (hello world)
- Query parameters with type hints
- POST endpoints with request body
- Multiple query parameters
- Protected endpoints (authentication required)
- Complex JSON responses

**Deploy:**
```bash
targon deploy examples/basic_web.py
```

**Test locally:**
```bash
python examples/basic_web.py
```

### `web_lifecycle.py` - Class-Based Endpoints with Lifecycle Management
Shows how to use `@app.cls()` with `@targon.enter()` for expensive initialization:
- Container lifecycle management with `@enter()`
- Loading ML models once per container
- Database connection pooling
- Multiple endpoints per class
- Shared state across requests

**Use cases:**
- ML model serving (load once, predict many times)
- Database connection management
- Cache initialization
- Any expensive one-time setup

**Deploy:**
```bash
targon deploy examples/web_lifecycle.py
```

### `simple_web.py` - Custom Web Server
Example using `@targon.web_server()` for custom web servers (not FastAPI).

## Key Features Demonstrated

### 1. FastAPI Endpoints
```python
@app.function()
@targon.fastapi_endpoint(docs=True)
def hello():
    return "Hello world!"
```

### 2. Query Parameters
```python
@app.function()
@targon.fastapi_endpoint(docs=True)
def greet(user: str) -> str:
    return f"Hello {user}!"
```

### 3. POST with Request Body
```python
@app.function()
@targon.fastapi_endpoint(method="POST", docs=True)
def process(data: dict) -> dict:
    return {"result": data}
```

### 4. Protected Endpoints
```python
@app.function(gpu="h100")
@targon.fastapi_endpoint(requires_proxy_auth=True)
def secret():
    return "Protected data"
```

### 5. Class-Based with Lifecycle
```python
@app.cls()
class Service:
    @targon.enter()
    def startup(self):
        self.model = load_model()  # Expensive, once per container
    
    @targon.fastapi_endpoint(docs=True)
    def predict(self, data: dict):
        return self.model.predict(data)
```

## Quick Start

1. **Install the SDK:**
   ```bash
   pip install -e /path/to/targon-sdk
   ```

2. **Run an example locally:**
   ```bash
   python examples/basic_web.py
   ```

3. **Deploy to Targon:**
   ```bash
   targon deploy examples/basic_web.py
   ```

4. **Access your endpoints:**
   After deployment, you'll receive URLs for each function.
   Add `/docs` to any URL to see interactive API documentation.

## Documentation

For comprehensive documentation on FastAPI endpoints, see:
- [FASTAPI_ENDPOINTS.md](../FASTAPI_ENDPOINTS.md) - Full FastAPI endpoint guide
- [README.md](../README.md) - Main SDK documentation

## Tips

- **Enable docs during development**: `@targon.fastapi_endpoint(docs=True)`
- **Use type hints**: They provide automatic validation and better documentation
- **Use `@app.cls()` for expensive startup**: Load models once, not per request
- **Protect expensive endpoints**: `requires_proxy_auth=True` for GPU-heavy functions
- **Test locally first**: Run examples with Python before deploying

## Getting Help

- Check the [main documentation](../README.md)
- See [FASTAPI_ENDPOINTS.md](../FASTAPI_ENDPOINTS.md) for detailed endpoint docs
- Open an issue on GitHub
