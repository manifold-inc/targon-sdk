# Targon Universal Runtime asgi
import asyncio
from typing import Any, Callable, Coroutine, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class LifespanManager:
    """Manages ASGI lifespan events for web endpoints."""

    def __init__(self) -> None:
        self._startup_complete = asyncio.Event()
        self._shutdown = asyncio.Event()

    async def lifespan_startup(self) -> None:
        """Called when the ASGI app starts up."""
        self._startup_complete.set()

    async def lifespan_shutdown(self) -> None:
        """Called when the ASGI app shuts down."""
        self._shutdown.set()
        await self._shutdown.wait()


def asgi_app_wrapper(
    asgi_app: Callable[..., Any],
) -> Tuple[Callable[..., Coroutine[Any, Any, None]], LifespanManager]:
    """Wrap an ASGI app to handle execution context and lifespan events."""
    state: Dict[str, Any] = {}
    lifespan_manager = LifespanManager()

    async def wrapped_app(
        scope: Dict[str, Any], receive: Callable, send: Callable
    ) -> None:
        if "state" not in scope:
            scope["state"] = state

        if scope["type"] == "lifespan":
            message = await receive()
            if message["type"] == "lifespan.startup":
                await lifespan_manager.lifespan_startup()
                await send({"type": "lifespan.startup.complete"})
            elif message["type"] == "lifespan.shutdown":
                await lifespan_manager.lifespan_shutdown()
                await send({"type": "lifespan.shutdown.complete"})
        else:
            await asgi_app(scope, receive, send)

    return wrapped_app, lifespan_manager


def wsgi_app_wrapper(
    wsgi_app: Callable[..., Any],
) -> Tuple[Callable[..., Coroutine[Any, Any, None]], LifespanManager]:
    """Wrap a WSGI app by converting it to ASGI."""
    try:
        from a2wsgi import WSGIMiddleware  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "WSGI app support requires the 'a2wsgi' package. "
            "Install it with: pip install a2wsgi"
        ) from exc

    asgi_app = WSGIMiddleware(wsgi_app)
    return asgi_app_wrapper(asgi_app)


def fastapi_app(fn: Callable[..., Any], method: str, docs: bool) -> Any:
    """Create a FastAPI app that wraps a single function."""
    try:
        from fastapi import FastAPI  # type: ignore
        from fastapi.middleware.cors import CORSMiddleware  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "FastAPI endpoints require the 'fastapi' package. "
            "Install it with: pip install 'fastapi[standard]'"
        ) from exc

    app = FastAPI(
        openapi_url="/openapi.json" if docs else None,
        docs_url="/docs" if docs else None,
        redoc_url="/redoc" if docs else None,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.add_api_route("/", fn, methods=[method.upper()])
    return app
