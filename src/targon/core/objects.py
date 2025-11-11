import functools
import asyncio
import aiohttp
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional, Callable, Self, TYPE_CHECKING, List

from targon.core.resolver import Resolver
from targon.core.exceptions import APIError, ValidationError
from collections.abc import Awaitable, Hashable, Sequence

if TYPE_CHECKING:
    from targon.core.function import _Function
    from targon.core.image import _Image
    from targon.client.client import Client


class _Object:
    """Base class for all Targon objects that can be loaded and hydrated."""

    # Class-level type registry
    _type_prefix: Optional[str] = None
    _prefix_to_type: dict[str, type["_Object"]] = {}

    # Instance attributes set during construction
    _local_uuid: str
    _load: Optional[Callable[["_Object", "Resolver", Optional[str]], Awaitable[None]]]
    _rep: str
    _deps: Optional[Callable[..., Sequence["_Object"]]]
    _deduplication_key: Optional[Callable[[], Awaitable[Hashable]]]
    _name: Optional[str]

    # Instance attributes set during hydration
    _object_id: Optional[str]
    _client: Optional["Client"]
    _is_hydrated: bool

    @classmethod
    def __init_subclass__(cls, type_prefix: Optional[str] = None):
        """Register subclasses with their type prefix for deserialization."""
        super().__init_subclass__()
        if type_prefix is not None:
            cls._type_prefix = type_prefix
            cls._prefix_to_type[type_prefix] = cls

    def __init__(self):
        """Prevent direct instantiation."""
        raise RuntimeError(
            f"Class {type(self).__name__} cannot be instantiated directly. "
            f"Use class constructor methods instead (e.g., {type(self).__name__}.from_loader())"
        )

    def _init(
        self,
        rep: str,  # representation of the object
        load: Optional[
            Callable[["_Object", "Resolver", Optional[str]], Awaitable[None]]
        ],  # loader function
        deps: Optional[
            Callable[..., Sequence["_Object"]]
        ] = None,  # dependencies function
        deduplication_key: Optional[
            Callable[[], Awaitable[Hashable]]
        ] = None,  # deduplication key function
        name: Optional[str] = None,  # name of the object
    ):
        """Initialize object with loader function and dependencies."""
        self._local_uuid = str(uuid.uuid4())
        self._load = load
        self._rep = rep
        self._deps = deps
        self._deduplication_key = deduplication_key

        self._object_id = None
        self._client = None
        self._is_hydrated = False

        self._name = name

        self._initialize_from_empty()

    def _initialize_from_empty(self):
        """Override in subclasses for custom initialization."""
        pass

    def _hydrate(
        self, object_id: str, client: "Client", metadata: Optional[dict] = None
    ):
        """Hydrate object with server-side data."""
        assert isinstance(object_id, str) and self._type_prefix is not None
        if not object_id.startswith(self._type_prefix):
            raise RuntimeError(
                f"Cannot hydrate {type(self)}: "
                f"it has type prefix {self._type_prefix} "
                f"but the object_id starts with {object_id[:3]}"
            )
        self._object_id = object_id
        self._client = client
        self._hydrate_metadata(metadata)
        self._is_hydrated = True

    def _hydrate_metadata(self, metadata: Optional[dict[str, Any]]):
        """Override in subclasses that need additional metadata."""
        pass

    def _get_metadata(self) -> Optional[dict[str, Any]]:
        """
        Return metadata needed to re-hydrate this object in another context.
        Override in subclasses that need to preserve state across
        serialization boundaries.
        """
        return None

    def _validate_is_hydrated(self):
        """Ensure object is hydrated before use."""
        if not self._is_hydrated:
            object_type = self.__class__.__name__.strip("_")
            raise RuntimeError(
                f"{object_type} has not been hydrated with the metadata it needs to run on Targon."
            )

    @classmethod
    def _from_loader(
        cls,
        load: Callable[["_Object", "Resolver", Optional[str]], Awaitable[None]],
        rep: str,
        deps: Optional[Callable[..., Sequence["_Object"]]] = None,
        deduplication_key: Optional[Callable[[], Awaitable[Hashable]]] = None,
        name: Optional[str] = None,
    ):
        """Create object from loader function."""
        obj = _Object.__new__(cls)
        obj._init(rep, load, deps, deduplication_key, name)
        return obj

    @staticmethod
    def _get_type_from_id(object_id: str) -> type["_Object"]:
        parts = object_id.split("-")
        if len(parts) != 2:
            raise ValidationError(
                f"Invalid object_id format: '{object_id}'. "
                f"Expected format: 'prefix-uuid'"
            )
        prefix = parts[0]
        if prefix not in _Object._prefix_to_type:
            raise ValidationError(
                f"Unknown object prefix: '{prefix}'. "
                f"Available prefixes: {", ".join(sorted(_Object._prefix_to_type.keys()))}"
            )
        return _Object._prefix_to_type[prefix]

    @classmethod
    def _is_id_type(cls, object_id: str) -> bool:
        return cls._get_type_from_id(object_id) == cls

    @property
    def local_uuid(self) -> str:
        """Unique identifier for this object instance."""
        return self._local_uuid

    @property
    def object_id(self) -> str:
        """Server-side object ID."""
        if self._object_id is None:
            raise AttributeError(f"Attempting to get object_id of unhydrated {self}")
        return self._object_id

    @property
    def client(self) -> "Client":
        """HTTP client for this object."""
        if self._client is None:
            raise AttributeError(
                f"Cannot access object_id of unhydrated {self}. "
                f"Call .hydrate() first."
            )
        return self._client

    @property
    def is_hydrated(self) -> bool:
        """Whether this object has been hydrated."""
        return self._is_hydrated

    @property
    def deps(self) -> Callable[..., Sequence["_Object"]]:
        """Get dependencies function."""

        def default_deps() -> Sequence["_Object"]:
            return []

        return self._deps if self._deps is not None else default_deps

    async def hydrate(self, client: Optional["Client"] = None) -> Self:
        """Hydrate this object with server-side data."""
        if self._is_hydrated:
            return self

        if not self._load:
            self._validate_is_hydrated()
            return self
        else:
            from targon.core.resolver import Resolver
            from targon.client.client import Client

            c = self.client or (client if client is not None else Client.from_env())
            resolver = Resolver(c)
            await resolver.load(self)
        return self

    def __repr__(self) -> str:
        return self._rep


def live_method(method):
    """Decorator that ensures an object is hydrated before calling a method."""

    if not asyncio.iscoroutinefunction(method):
        raise TypeError(
            f"@live_method can only decorate async methods, "
            f"but {method.__name__} is not a coroutine function"
        )

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        async def _async_call():
            # await self.hydrate()
            return await method(self, *args, **kwargs)

        coro = _async_call()

        # Check if we're already inside a running event loop
        try:
            asyncio.current_task()
            # If we get here, we're in a running event loop
            return coro
        except RuntimeError:
            # No running event loop, so we need to run one
            return asyncio.run(coro)

    return wrapper


class _LocalEntrypoint:
    _raw_f: Callable
    _app: "BaseApp"

    def __init__(self, raw_f: Callable, app: "BaseApp") -> None:
        self._raw_f = raw_f
        self._app = app

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the local entrypoint function."""
        return self._raw_f(*args, **kwargs)

    @property
    def raw_f(self) -> Callable:
        """Return the raw Python function."""
        return self._raw_f

    @property
    def app(self) -> "BaseApp":
        """Return the app this entrypoint belongs to."""
        return self._app


class BaseApp:
    _name: str
    _description: Optional[str]
    _tags: Dict[str, str]
    _functions: Dict[str, "_Function"]
    _classes: Dict[str, Any]
    _image: "_Image"
    _local_entrypoints: Dict[str, _LocalEntrypoint]
    _web_endpoints: List[str]
    _app_id: str
    _project_name: str

    def __init__(self, name: str, image: "_Image", project_name: str) -> None:
        self._name = name.strip()
        self._image = image
        self._project_name = project_name.strip()

    @property
    def name(self) -> Optional[str]:
        return self._name

    @property
    def description(self) -> Optional[str]:
        return self._description

    @property
    def functions(self) -> dict[str, "_Function"]:
        return self._functions

    @property
    def classes(self) -> dict[str, "_Function"]:
        return self._classes

    @property
    def app_id(self) -> Optional[str]:
        return self._app_id

    @property
    def registered_web_endpoints(self) -> List[str]:
        return self._web_endpoints

    @property
    def registered_functions(self) -> Dict[str, "_Function"]:
        return self._functions

    @property
    def project_name(self) -> str:
        return self._project_name

    def function(self):
        pass

    def local_entrypoint(self):
        pass

    def deploy(self):
        pass

    @asynccontextmanager
    async def run(self):
        yield


class AsyncBaseHTTPClient:
    def __init__(self, client: "Client"):
        self.client = client
        self.session = client.async_session
        self.base_url = client.config.base_url.rstrip("/")

    @classmethod
    async def from_env(cls) -> "AsyncBaseHTTPClient":
        from targon.client.client import Client

        client = Client.from_env()
        return cls(client)

    async def _async_get(self, path: str, **kwargs: Any):
        async with self.session.get(f"{self.base_url}{path}", **kwargs) as res:
            return await self._handle_async_response(res)

    async def _async_post(self, path: str, **kwargs: Any):
        async with self.session.post(f"{self.base_url}{path}", **kwargs) as res:
            return await self._handle_async_response(res)

    async def _async_patch(self, path: str, **kwargs: Any):
        async with self.session.patch(f"{self.base_url}{path}", **kwargs) as res:
            return await self._handle_async_response(res)

    async def _async_delete(self, path: str, **kwargs: Any):
        async with self.session.delete(f"{self.base_url}{path}", **kwargs) as res:
            return await self._handle_async_response(res)

    async def _handle_async_response(self, res: aiohttp.ClientResponse):
        if res.status >= 400:
            text = await res.text()
            raise APIError(res.status, text)
        try:
            return (
                await res.json()
                if res.content_type == 'application/json'
                else await res.text()
            )
        except (ValueError, aiohttp.ContentTypeError):
            return await res.text()
