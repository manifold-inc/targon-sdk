# import asyncio
# import json
# import logging
# import os
# import random
# from pathlib import Path
# from typing import Any, Dict, List, Optional

# import aiohttp
# import fcntl
# from aiobotocore.session import get_session
# from botocore.config import Config

# # R2 Storage Configuration
# FOLDER = os.getenv("R2_FOLDER", "affine")
# BUCKET = os.getenv("R2_BUCKET_ID", "00523074f51300584834607253cae0fa")
# ACCESS = os.getenv("R2_WRITE_ACCESS_KEY_ID", "")
# SECRET = os.getenv("R2_WRITE_SECRET_ACCESS_KEY", "")
# ENDPOINT = f"https://{BUCKET}.r2.cloudflarestorage.com"

# # Public read configuration
# PUBLIC_READ = os.getenv("R2_PUBLIC_READ", "true").lower() == "true"
# R2_PUBLIC_BASE = os.getenv(
#     "R2_PUBLIC_BASE", "https://pub-bf429ea7a5694b99adaf3d444cbbe64d.r2.dev"
# )

# # Cache configuration
# CACHE_DIR = os.getenv("DATASET_CACHE_DIR", "/tmp/dataset_cache")
# DOWNLOAD_CONCURRENCY = int(os.getenv("DOWNLOAD_CONCURRENCY", "1"))

# # Logger
# logger = logging.getLogger("affine")

# # Shared HTTP client session
# _http_client: Optional[aiohttp.ClientSession] = None


# async def _get_http_client() -> aiohttp.ClientSession:
#     """Get or create shared HTTP client session."""
#     global _http_client
#     if _http_client is None or _http_client.closed:
#         _http_client = aiohttp.ClientSession()
#     return _http_client


# class R2Dataset:
#     """
#     R2 dataset with background download and local-only sampling.

#     Features:
#     - Background download of all files to local cache
#     - Sampling only from local cache (no real-time downloads)
#     - Automatic retry on download failures
#     """

#     def __init__(
#         self,
#         dataset_name: str,
#         seed: Optional[int] = None,
#         cache_dir: Optional[str] = None,
#         download_concurrency: int = DOWNLOAD_CONCURRENCY,
#     ):
#         """
#         Initialize R2 dataset with background downloading.

#         Args:
#             dataset_name: Name of the dataset (e.g., \"satpalsr/rl-python\")
#             seed: Random seed for reproducibility (optional)
#             cache_dir: Directory for local cache (default: /tmp/dataset_cache)
#             download_concurrency: Number of concurrent downloads (default: 10)
#         """
#         self.dataset_name = dataset_name
#         self._rng = random.Random(seed)
#         self._download_concurrency = download_concurrency

#         # Cache configuration
#         self._cache_dir = Path(cache_dir or CACHE_DIR) / dataset_name.replace("/", "_")
#         self._cache_dir.mkdir(parents=True, exist_ok=True)

#         # Build dataset paths
#         self._dataset_folder = f"affine/datasets/{dataset_name}/"
#         self._index_key = self._dataset_folder + "index.json"

#         # R2 credentials
#         self._endpoint_url = ENDPOINT
#         self._access_key = ACCESS
#         self._secret_key = SECRET
#         self._public_read = PUBLIC_READ
#         self._public_base = R2_PUBLIC_BASE

#         # Dataset metadata
#         self._index: Optional[Dict[str, Any]] = None
#         self._files: List[Dict[str, Any]] = []
#         self.total_size: int = 0

#         # Start background initialization
#         self._init_task = asyncio.create_task(self._async_init())
#         logger.info(
#             "Started background initialization for dataset '%s'", self.dataset_name
#         )

#     def _get_s3_client(self):
#         """Create S3 client context for private R2 access."""
#         if not self._endpoint_url:
#             raise RuntimeError("R2 endpoint is not configured (missing R2_BUCKET_ID)")

#         session = get_session()
#         return session.create_client(
#             "s3",
#             endpoint_url=self._endpoint_url,
#             aws_access_key_id=self._access_key,
#             aws_secret_access_key=self._secret_key,
#             config=Config(max_pool_connections=256),
#         )

#     async def _async_init(self) -> None:
#         """Background initialization: load index and download all files."""
#         try:
#             await self._load_index()
#             await self._download_all_files()
#             logger.info(
#                 "Dataset '%s' initialization complete", self.dataset_name
#             )
#         except Exception as exc:
#             logger.error("Background initialization failed: %s", exc)

#     async def _load_index(self) -> None:
#         """Load dataset index from R2."""
#         if self._index is not None:
#             return

#         try:
#             if self._public_read:
#                 url = f"{self._public_base}/{self._index_key}"
#                 logger.debug("Loading public R2 index: %s", url)

#                 client = await _get_http_client()
#                 async with client.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
#                     resp.raise_for_status()
#                     self._index = await resp.json()
#             else:
#                 logger.debug("Loading R2 index: s3://%s/%s", FOLDER, self._index_key)

#                 async with self._get_s3_client() as s3:
#                     resp = await s3.get_object(Bucket=FOLDER, Key=self._index_key)
#                     body = await resp["Body"].read()
#                     self._index = json.loads(body.decode())

#             # Extract file list and metadata
#             self._files = list(self._index.get("files", []))
#             self.total_size = int(self._index.get("total_rows", 0))

#             if not self._files:
#                 raise RuntimeError(
#                     f"Dataset '{self.dataset_name}' contains no files"
#                 )

#             logger.info(
#                 "Loaded dataset '%s': %d files, %d total samples",
#                 self.dataset_name,
#                 len(self._files),
#                 self.total_size,
#             )

#         except Exception as exc:
#             logger.error("Failed to load dataset index: %s", exc)
#             raise RuntimeError(
#                 f"Failed to load dataset '{self.dataset_name}': {exc}"
#             ) from exc

#     async def _download_all_files(self) -> None:
#         """Background task to download all files to local cache in random order."""
#         logger.info(
#             "Starting download of all %d files for dataset '%s'",
#             len(self._files),
#             self.dataset_name,
#         )

#         # Shuffle files for random download order
#         shuffled_files = self._files.copy()
#         self._rng.shuffle(shuffled_files)

#         # Download files in batches with concurrency control
#         semaphore = asyncio.Semaphore(self._download_concurrency)

#         async def download_with_semaphore(file_info):
#             async with semaphore:
#                 return await self._download_and_cache_file(file_info)

#         tasks = [download_with_semaphore(file_info) for file_info in shuffled_files]
#         results = await asyncio.gather(*tasks, return_exceptions=True)

#         # Count successful downloads
#         success_count = sum(1 for result in results if result and not isinstance(result, Exception))

#         logger.info(
#             "Download completed: %d/%d files cached for dataset '%s'",
#             success_count,
#             len(self._files),
#             self.dataset_name,
#         )

#     async def _download_and_cache_file(self, file_info: Dict[str, Any]) -> bool:
#         """Download a file and save it to local cache with file locking."""
#         key = file_info.get("key") or (self._dataset_folder + file_info.get("filename", ""))
#         if not key:
#             return False

#         cache_path = self._get_cache_path(key)
#         lock_path = cache_path.with_suffix(".lock")

#         # Skip if already cached
#         if cache_path.exists():
#             logger.debug("File already cached: %s", key)
#             return True

#         # Use file lock to prevent concurrent downloads across processes
#         try:
#             lock_path.parent.mkdir(parents=True, exist_ok=True)
#             lock_file = open(lock_path, "w")

#             try:
#                 # Try to acquire exclusive lock (non-blocking)
#                 fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

#                 # Double-check file doesn't exist (race condition)
#                 if cache_path.exists():
#                     logger.debug("File already cached (after lock): %s", key)
#                     return True

#                 # Download file
#                 if self._public_read:
#                     url = f"{self._public_base}/{key}"
#                     client = await _get_http_client()
#                     async with client.get(url, timeout=aiohttp.ClientTimeout(total=120)) as resp:
#                         resp.raise_for_status()
#                         body = await resp.read()
#                 else:
#                     async with self._get_s3_client() as s3:
#                         resp = await s3.get_object(Bucket=FOLDER, Key=key)
#                         body = await resp["Body"].read()

#                 # Validate JSON
#                 data = json.loads(body.decode())
#                 if not isinstance(data, list) or not data:
#                     logger.warning("Skipping invalid/empty file: %s", key)
#                     return False

#                 # Write to cache atomically
#                 temp_path = cache_path.with_suffix(".tmp")
#                 temp_path.write_bytes(body)
#                 temp_path.rename(cache_path)

#                 logger.debug("Cached file: %s (%d samples)", key, len(data))
#                 return True

#             except BlockingIOError:
#                 # Another process is downloading this file, wait for it
#                 logger.debug("File %s is being downloaded by another process, waiting...", key)
#                 fcntl.flock(lock_file.fileno(), fcntl.LOCK_SH)

#                 # Check if download succeeded
#                 if cache_path.exists():
#                     logger.debug("File %s downloaded by another process", key)
#                     return True
#                 logger.warning("File %s download failed in another process", key)
#                 return False

#             finally:
#                 fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
#                 lock_file.close()

#                 # Clean up lock file
#                 try:
#                     lock_path.unlink()
#                 except Exception:
#                     pass

#         except Exception as exc:
#             logger.warning(
#                 "Failed to download file %s: %s: %s", key, type(exc).__name__, exc
#             )
#             return False

#     def _get_cache_path(self, key: str) -> Path:
#         """Get local cache path for a file key."""
#         filename = key.split("/")[-1]
#         return self._cache_dir / filename

#     def _get_cached_files(self) -> List[str]:
#         """Get list of cached file paths."""
#         if not self._cache_dir.exists():
#             return []
#         return [str(path) for path in self._cache_dir.glob("*.json")]

#     async def _read_from_cache(self, cache_path: Path) -> Optional[List[Any]]:
#         """Read file from local cache."""
#         if not cache_path.exists():
#             return None

#         try:
#             body = cache_path.read_bytes()
#             data = json.loads(body.decode())

#             if not isinstance(data, list):
#                 return None

#             return data

#         except Exception as exc:
#             logger.warning("Failed to read from cache %s: %s", cache_path, exc)
#             return None

#     async def get(self) -> Any:
#         """
#         Get a random sample from the local cache.

#         Returns:
#             A randomly selected sample from cached files

#         Raises:
#             RuntimeError: If no cached files available after retries
#         """
#         # Retry mechanism: check for cached files, sleep 5s and retry up to 3 times
#         max_retries = 3
#         retry_delay = 5

#         for attempt in range(max_retries):
#             cached_files = self._get_cached_files()

#             if cached_files:
#                 break

#             if attempt < max_retries - 1:
#                 logger.warning(
#                     "No cached files found for dataset '%s' (attempt %d/%d), waiting %ds...",
#                     self.dataset_name,
#                     attempt + 1,
#                     max_retries,
#                     retry_delay,
#                 )
#                 await asyncio.sleep(retry_delay)
#             else:
#                 raise RuntimeError(
#                     f"No cached files available for dataset '{self.dataset_name}' after {max_retries} retries"
#                 )

#         # Randomly select a cached file
#         selected_file = Path(self._rng.choice(cached_files))

#         # Load file contents
#         samples = await self._read_from_cache(selected_file)

#         if not samples:
#             raise RuntimeError(f"Failed to read cached file: {selected_file}")

#         # Randomly select a sample from the file
#         sample = self._rng.choice(samples)

#         logger.debug(
#             "Retrieved random sample from dataset '%s' (file: %s)",
#             self.dataset_name,
#             selected_file.name,
#         )
#         return sample

#     def __aiter__(self):
#         """Support async iteration."""
#         return self

#     async def __anext__(self) -> Any:
#         """Get next random sample."""
#         return await self.get()

