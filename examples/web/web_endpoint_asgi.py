"""Full ASGI application example with multiple routes."""
import targon

# Create an image with FastAPI installed
# Note: Runtime is automatically injected during deployment!
image = targon.Image.debian_slim("3.13").pip_install("fastapi[standard]")

# Create an app
app = targon.App("web-endpoint-asgi", image=image)


@app.function()
@targon.asgi_app(label="api")
def hello():
    """Create a full FastAPI application with multiple routes.
    
    This demonstrates:
    - Multiple endpoints in one app
    - Different HTTP methods
    - Path parameters
    - Query parameters
    - Full control over FastAPI configuration
    """
    from fastapi import FastAPI, HTTPException, Query
    from pydantic import BaseModel
    from typing import List, Optional
    
    api = FastAPI(
        title="Targon Demo API",
        description="A sample API demonstrating full ASGI app deployment",
        version="1.0.0",
    )
    
    # In-memory storage for demo
    items_db = {}
    
    class Item(BaseModel):
        name: str
        description: Optional[str] = None
        price: float
        
    @api.get("/")
    def root():
        """Root endpoint."""
        return {
            "message": "Welcome to Targon ASGI Demo API",
            "endpoints": [
                "GET / - This message",
                "GET /health - Health check",
                "GET /items - List all items",
                "GET /items/{item_id} - Get specific item",
                "POST /items - Create new item",
                "DELETE /items/{item_id} - Delete item",
            ]
        }
    
    @api.get("/health")
    def health_check():
        """Health check endpoint."""
        return {"status": "healthy", "items_count": len(items_db)}
    
    @api.get("/items")
    def list_items(
        skip: int = Query(0, ge=0),
        limit: int = Query(10, ge=1, le=100)
    ):
        """List items with pagination."""
        item_list = list(items_db.items())[skip:skip + limit]
        return {
            "items": [{"id": k, **v} for k, v in item_list],
            "total": len(items_db),
            "skip": skip,
            "limit": limit,
        }
    
    @api.get("/items/{item_id}")
    def get_item(item_id: str):
        """Get a specific item by ID."""
        if item_id not in items_db:
            raise HTTPException(status_code=404, detail="Item not found")
        return {"id": item_id, **items_db[item_id]}
    
    @api.post("/items")
    def create_item(item: Item):
        """Create a new item."""
        import uuid
        item_id = str(uuid.uuid4())[:8]
        items_db[item_id] = item.dict()
        return {"id": item_id, **items_db[item_id]}
    
    @api.delete("/items/{item_id}")
    def delete_item(item_id: str):
        """Delete an item."""
        if item_id not in items_db:
            raise HTTPException(status_code=404, detail="Item not found")
        deleted_item = items_db.pop(item_id)
        return {"deleted": True, "item": deleted_item}
    
    return api


