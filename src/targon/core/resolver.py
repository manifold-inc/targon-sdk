import asyncio
import os
import tempfile
from asyncio import Future
from collections.abc import Hashable
from typing import TYPE_CHECKING, Optional
from pathlib import Path

from targon.core.exceptions import ValidationError

if TYPE_CHECKING:
    from targon.core.objects import _Object
    from targon.client.client import Client


class Resolver:
    """Resolver for loading and hydrating Targon objects."""

    _local_uuid_to_future: dict[str, Future["_Object"]]
    _project_name: Optional[str]
    _app_id: Optional[str]
    _deduplication_cache: dict[Hashable, Future["_Object"]]
    _client: "Client"
    _build_start: float
    _console_instance: Optional[object]
    _app_file: Optional[Path]
    _app_module_name: Optional[str]

    def __init__(
        self,
        client: "Client",
        *,
        project_name: Optional[str] = None,
        app_id: Optional[str] = None,
        console_instance: Optional[object] = None,
        app_file: Optional[Path] = None,
        app_module_name: Optional[str] = None,
    ):
        self._local_uuid_to_future = {}
        self._client = client
        self._app_id = app_id
        self._project_name = project_name
        self._deduplication_cache = {}
        self._console_instance = console_instance
        self._app_file = app_file
        self._app_module_name = app_module_name

        with tempfile.TemporaryFile() as temp_file:
            # Use file mtime to track build start time
            self._build_start = os.fstat(temp_file.fileno()).st_mtime

    @property
    def app_id(self) -> Optional[str]:
        return self._app_id

    @property
    def client(self) -> "Client":
        return self._client

    @property
    def project_name(self) -> Optional[str]:
        return self._project_name

    @property
    def build_start(self) -> float:
        return self._build_start

    @property
    def console_instance(self) -> Optional[object]:
        return self._console_instance

    async def load(
        self, obj: "_Object", existing_object_id: Optional[str] = None
    ) -> "_Object":
        """Load and hydrate an object, handling dependencies and deduplication."""
        if getattr(obj, "_is_hydrated", False):
            if obj.local_uuid not in self._local_uuid_to_future:
                fut: Future["_Object"] = Future()
                fut.set_result(obj)
                self._local_uuid_to_future[obj.local_uuid] = fut
            return obj

        # Handle deduplication
        deduplication_key: Optional[Hashable] = None
        if obj._deduplication_key:
            deduplication_key = await obj._deduplication_key()

        cached_future = self._local_uuid_to_future.get(obj.local_uuid)

        # Check deduplication cache
        # check if the object is already in the deduplication cache
        # which means that this object has been loaded before other app
        if not cached_future and deduplication_key is not None:
            cached_future = self._deduplication_cache.get(deduplication_key)
            if cached_future:
                hydrated_object = await cached_future
                obj._hydrate(
                    hydrated_object.object_id,
                    self._client,
                    hydrated_object._get_metadata(),
                )
                return obj

        if not cached_future:

            async def loader() -> "_Object":
                deps = obj.deps()
                if deps:
                    await asyncio.gather(*[self.load(dep) for dep in deps])

                if not obj._load:
                    raise ValidationError(
                        f"Object {obj} has no loader function",
                        field="obj._load",
                    )

                await obj._load(obj, self, existing_object_id)

                if (
                    existing_object_id is not None
                    and existing_object_id.startswith("fu-")
                    and obj.object_id != existing_object_id
                ):
                    raise ValidationError(
                        f"Object ID mismatch: expected {existing_object_id}, got {obj.object_id}",
                        field="object_id",
                        value=obj.object_id,
                    )

                return obj

            cached_future = asyncio.create_task(loader())
            self._local_uuid_to_future[obj.local_uuid] = cached_future

            if deduplication_key is not None:
                self._deduplication_cache[deduplication_key] = cached_future

        return await cached_future

    def objects(self) -> list["_Object"]:
        """Get all loaded objects."""
        unique_objects: dict[str, "_Object"] = {}
        for fut in self._local_uuid_to_future.values():
            if not fut.done():
                raise ValidationError(
                    "Cannot retrieve objects: not all objects have been resolved yet"
                )
            obj = fut.result()
            unique_objects.setdefault(obj.object_id, obj)
        return list(unique_objects.values())

    def clear_cache(self):
        self._deduplication_cache.clear()

    def get_object_count(self) -> int:
        return len(self._local_uuid_to_future)

    def is_loading(self, obj: "_Object") -> bool:
        return obj.local_uuid in self._local_uuid_to_future

    async def load_all(self, objects: list["_Object"]) -> list["_Object"]:
        return await asyncio.gather(*[self.load(obj) for obj in objects])

    def get_loading_status(self) -> dict[str, str]:
        status: dict[str, str] = {}
        for uuid, future in self._local_uuid_to_future.items():
            if future.done():
                try:
                    obj = future.result()
                    status[uuid] = f"loaded: {obj.object_id}"
                except Exception as e:
                    status[uuid] = f"error: {type(e).__name__}: {e}"
            else:
                status[uuid] = "loading"
        return status

    def get_app_file(self) -> Optional[Path]:
        return self._app_file

    def get_app_module_name(self) -> Optional[str]:
        return self._app_module_name
