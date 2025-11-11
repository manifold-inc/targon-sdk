# Targon Universal Runtime
import grpc
from concurrent import futures
import asyncio
import os
import sys
import importlib
import cloudpickle
import time
import traceback as tb
import uuid
import logging
import json
import threading
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from enum import Enum

from targon.proto import function_execution_pb2
from targon.proto import function_execution_pb2_grpc

from targon.runtime import asgi

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s]  | %(message)s"
)
logger = logging.getLogger(__name__)

try:
    import uvicorn  # type: ignore
except ImportError as e:
    logger.error(f"Missing dependency for web endpoints: {e}")
    logger.error("Install with: pip install 'fastapi[standard]' uvicorn")
    raise ImportError(
        "Web endpoint dependencies not installed. "
        "Run: pip install 'fastapi[standard]' uvicorn"
    ) from e


class RuntimeMode(str, Enum):
    GRPC = "grpc"
    WEB_ENDPOINT = "web_endpoint"
    WEB_SERVER = "web_server"


@dataclass
class WebhookConfig:
    type: str
    method: str = "GET"
    docs: bool = False
    label: str = ""
    requires_auth: bool = False
    port: Optional[int] = None
    startup_timeout: int = 30

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> Optional['WebhookConfig']:
        if not data:
            return None

        return cls(
            type=data.get("type", ""),
            method=data.get("method", "GET"),
            docs=data.get("docs", False),
            label=data.get("label", ""),
            requires_auth=data.get("requires_auth", False),
            port=data.get("port"),
            startup_timeout=data.get("startup_timeout", 30),
        )

    def is_valid(self) -> bool:
        return bool(self.type and self.type not in ["", "unspecified"])


@dataclass
class RuntimeConfig:
    app_module: str
    target_function: str

    function_id: str = "unknown"
    port: int = 50051

    grpc_max_workers: int = 10
    grpc_max_message_length: int = 500 * 1024 * 1024  # 100MB

    default_timeout_ms: int = 300000  # 5 minutes

    webhook_config: Optional[WebhookConfig] = None

    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_environment(cls) -> 'RuntimeConfig':
        app_module = os.environ.get("APP_MODULE", "app")
        target_function = os.environ.get("TARGET_FUNCTION")
        if not target_function:
            raise ValueError(
                "TARGET_FUNCTION environment variable is required but not set. "
                "This should specify the name of the function to execute."
            )
        function_id = os.environ.get("FUNCTION_ID", "unknown")
        port = int(os.environ.get("FUNCTION_PORT", "50051"))
        if not (1 <= port <= 65535):
            raise ValueError(f"Port must be between 1 and 65535, got {port}")

        grpc_max_workers = int(os.environ.get("GRPC_MAX_WORKERS", "10"))
        if grpc_max_workers < 1:
            raise ValueError("GRPC_MAX_WORKERS must be positive")

        grpc_max_message_length = int(
            os.environ.get("GRPC_MAX_MESSAGE_LENGTH", str(500 * 1024 * 1024))
        )
        if grpc_max_message_length > 500 * 1024 * 1024:
            raise ValueError("GRPC_MAX_MESSAGE_LENGTH must be less then 100MB")

        default_timeout_ms = int(os.environ.get("DEFAULT_TIMEOUT_MS", "300000"))
        if default_timeout_ms < 0:
            raise ValueError("DEFAULT_TIMEOUT_MS must be non-negative")

        # Parse function metadata
        metadata = {}
        function_metadata = os.environ.get("FUNCTION_METADATA", "{}")
        if function_metadata:
            try:
                metadata = json.loads(function_metadata)
                if not isinstance(metadata, dict):
                    logger.warning(
                        f"FUNCTION_METADATA is not a JSON object, got {type(metadata).__name__}"
                    )
                    metadata = {}
            except json.JSONDecodeError as e:
                logger.warning(
                    f"Failed to parse FUNCTION_METADATA as JSON: {e}. "
                    "No metadata will be loaded."
                )
                metadata = {}

        # Parse webhook configuration
        webhook_config = WebhookConfig.from_dict(metadata.get("webhook_config"))

        return cls(
            app_module=app_module,
            target_function=target_function,
            function_id=function_id,
            port=port,
            grpc_max_workers=grpc_max_workers,
            grpc_max_message_length=grpc_max_message_length,
            default_timeout_ms=default_timeout_ms,
            webhook_config=webhook_config,
            metadata=metadata,
        )

    def get_runtime_mode(self) -> RuntimeMode:
        if self.webhook_config and self.webhook_config.is_valid():
            if self.webhook_config.type == "web_server":
                return RuntimeMode.WEB_SERVER
            return RuntimeMode.WEB_ENDPOINT
        return RuntimeMode.GRPC

    def log_configuration(self) -> None:
        logger.info("=" * 40)
        logger.info("Targon Universal Runtime - Starting")
        logger.info("=" * 40)
        logger.info(f"App Module:        {self.app_module}")
        logger.info(f"Target Function:   {self.target_function}")
        logger.info(f"Function ID:       {self.function_id}")
        logger.info(f"Port:              {self.port}")
        logger.info(f"Runtime Mode:      {self.get_runtime_mode().value}")
        logger.info(f"gRPC Workers:      {self.grpc_max_workers}")
        logger.info(f"Default Timeout:   {self.default_timeout_ms}ms")

        if self.webhook_config and self.webhook_config.is_valid():
            logger.info(f"Webhook Type:      {self.webhook_config.type}")
            logger.info(f"Webhook Method:    {self.webhook_config.method}")
            logger.info(f"Webhook Auth:      {self.webhook_config.requires_auth}")

        logger.info("=" * 40)


class FunctionLoader:
    """Handles loading and unwrapping user functions from modules."""

    def __init__(self, config: RuntimeConfig):
        self.config = config
        self.user_function = None
        self.actual_function = None

    def load(self) -> Any:
        # Add /app to Python path for imports
        if "/app" not in sys.path:
            sys.path.insert(0, "/app")

        try:
            app_module = importlib.import_module(self.config.app_module)
        except ImportError as e:
            available_modules = [
                name for name in sys.modules.keys() if not name.startswith('_')
            ]
            logger.error(f"Failed to import module '{self.config.app_module}': {e}")
            logger.error(f"Python path: {sys.path}")
            logger.error(f"Available modules [:20]: {available_modules[:20]}")
            raise ImportError(
                f"Cannot import module '{self.config.app_module}'. "
                f"Ensure the module exists in /app and has no import errors."
            ) from e

        # Get the function from the module
        self.user_function = getattr(app_module, self.config.target_function, None)

        if self.user_function is None:
            available_attrs = [x for x in dir(app_module) if not x.startswith('_')]
            logger.error(
                f"Function '{self.config.target_function}' not found in '{self.config.app_module}'"
            )
            logger.error(f"Available functions/objects: {available_attrs}")
            raise AttributeError(
                f"Function '{self.config.target_function}' not found in module '{self.config.app_module}'. "
                f"Available: {', '.join(available_attrs[:10])}"
            )

        self._merge_decorator_config()

        # Unwrap the function to get the actual callable
        self.actual_function = self._unwrap_function()

        # logger.info(f"Actual callable type: {type(self.actual_function).__name__}")
        # logger.info("Function loading complete")

        return self.actual_function

    def _merge_decorator_config(self) -> None:
        """Merge webhook configuration from decorator with environment config."""
        if not hasattr(self.user_function, '_webhook_config'):
            return

        # Function decorated with _PartialFunction no need to do anything
        decorator_config = self.user_function._webhook_config  # type: ignore
        if not decorator_config:
            return

        # If no environment config exists, use decorator config entirely
        if not self.config.webhook_config or not self.config.webhook_config.is_valid():
            self.config.webhook_config = WebhookConfig(
                type=decorator_config.type,
                method=getattr(decorator_config, 'method', 'GET'),
                docs=getattr(decorator_config, 'docs', False),
                label=getattr(decorator_config, 'label', ''),
                requires_auth=getattr(decorator_config, 'requires_auth', False),
                port=decorator_config.port or self.config.port,
                startup_timeout=getattr(decorator_config, 'startup_timeout', 30),
            )
        else:
            # Merge decorator config with environment config
            # Decorator takes precedence for port
            if hasattr(decorator_config, 'port') and decorator_config.port:
                if self.config.webhook_config.port != decorator_config.port:
                    logger.warning(
                        f"Port mismatch: environment={self.config.webhook_config.port}, "
                        f"decorator={decorator_config.port}. Using decorator port."
                    )
                    self.config.webhook_config.port = decorator_config.port

    def _unwrap_function(self) -> Any:
        """
        Unwrap decorated function to get the actual callable.

        Supports multiple decoration patterns:
        - _raw_f attribute (standard Targon functions)
        - local() method (for remote-enabled functions)
        - Direct callable (undecorated functions)
        """
        if hasattr(self.user_function, '_raw_f'):
            logger.info("Unwrapping via _raw_f attribute")
            return self.user_function._raw_f  # type: ignore
        elif hasattr(self.user_function, 'local'):
            logger.info("Unwrapping via local() method")
            return self.user_function.local  # type: ignore
        else:
            return self.user_function


class TimeoutExecutor:
    """Handles function execution with timeout using threading."""

    @staticmethod
    def run_with_timeout(func: Any, args: tuple, kwargs: dict, timeout_seconds: float):
        """
        Execute a function with a timeout using threading.

        Note: We use threading instead of signals (SIGALRM) because:
        1. gRPC handles requests in worker threads from ThreadPoolExecutor
        2. signal.signal() only works in the main thread
        3. Threading-based timeout works in any thread

        Args:
            func: The function to execute
            args: Positional arguments
            kwargs: Keyword arguments
            timeout_seconds: Timeout in seconds (0 or negative means no timeout)

        Returns:
            tuple: (result, error, timed_out)
        """
        result: list[Any] = [None]
        error: list[Optional[Exception]] = [None]
        timed_out: list[bool] = [False]

        def target():
            try:
                result[0] = func(*args, **kwargs)
            except Exception as e:
                error[0] = e

        thread = threading.Thread(target=target, name="FunctionExecutor")
        thread.daemon = True
        thread.start()

        # Wait for completion or timeout
        timeout = timeout_seconds if timeout_seconds > 0 else None
        thread.join(timeout=timeout)

        if thread.is_alive():
            timed_out[0] = True
            logger.warning("Function execution timed out, thread may still be running")

        return result[0], error[0], timed_out[0]


class HealthServicer(function_execution_pb2_grpc.HealthServicer):
    """gRPC health checks for Kubernetes/Knative readiness and liveness probes."""

    def Check(self, request, context):
        """Handle health check requests."""
        return function_execution_pb2.HealthCheckResponse(
            status=function_execution_pb2.HealthCheckResponse.SERVING
        )

    def Watch(self, request, context):
        """Stream health status updates."""
        try:
            while True:
                yield function_execution_pb2.HealthCheckResponse(
                    status=function_execution_pb2.HealthCheckResponse.SERVING
                )
                time.sleep(5)
        except Exception as e:
            logger.error(f"Health watch error: {e}")


class FunctionExecutorServicer(function_execution_pb2_grpc.FunctionExecutorServicer):
    """gRPC service for executing user functions."""

    def __init__(self, config: RuntimeConfig, function: Any):
        self.config = config
        self.function = function
        self.executor = TimeoutExecutor()

    def Execute(self, request, context):
        """Execute the pre-loaded function with provided arguments."""

        # Extract request metadata
        request_id = (
            request.request_id
            if hasattr(request, 'request_id') and request.request_id
            else str(uuid.uuid4())
        )
        timeout_ms = (
            request.timeout_ms
            if hasattr(request, 'timeout_ms') and request.timeout_ms > 0
            else self.config.default_timeout_ms
        )
        timeout_seconds = timeout_ms / 1000.0

        logger.info(
            f"[{request_id}] Executing {self.config.target_function} "
            f"(timeout: {timeout_seconds}s)"
        )
        start_time = time.time()

        try:
            # Deserialize arguments
            args_result = self._deserialize_arguments(request, request_id, start_time)
            if (
                args_result is None
            ):  # Deserialization failed, method already returned error
                return None  # This should not happen, but handle defensively

            if isinstance(args_result, function_execution_pb2.ExecuteResponse):
                # Deserialization error occurred, return the error response
                return args_result

            args, kwargs = args_result

            # Execute the function with timeout
            result, exec_error, timed_out = self.executor.run_with_timeout(
                self.function, args, kwargs, timeout_seconds
            )

            # Handle timeout
            if timed_out:
                return self._create_timeout_response(
                    request_id, timeout_seconds, start_time
                )

            # Handle execution error
            if exec_error is not None:
                return self._create_error_response(request_id, exec_error, start_time)

            # Serialize and return successful result
            return self._create_success_response(request_id, result, start_time)

        except Exception as e:
            # Catch-all for unexpected errors
            return self._create_unexpected_error_response(request_id, e, start_time)

    def _deserialize_arguments(self, request, request_id: str, start_time: float):
        """
        Deserialize function arguments from request.

        Returns:
            tuple: (args, kwargs) on success
            ExecuteResponse: Error response on failure
        """
        args = ()
        kwargs = {}

        if request.args:
            try:
                args = cloudpickle.loads(request.args)
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"[{request_id}] Failed to deserialize args: {e}")
                return function_execution_pb2.ExecuteResponse(
                    request_id=request_id,
                    status=function_execution_pb2.ExecuteResponse.ERROR,
                    error=f"Failed to deserialize arguments: {type(e).__name__}: {e}",
                    traceback=tb.format_exc(),
                    execution_time_ms=int(execution_time * 1000),
                )

        if request.kwargs:
            try:
                kwargs = cloudpickle.loads(request.kwargs)
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"[{request_id}] Failed to deserialize kwargs: {e}")
                return function_execution_pb2.ExecuteResponse(
                    request_id=request_id,
                    status=function_execution_pb2.ExecuteResponse.ERROR,
                    error=f"Failed to deserialize keyword arguments: {type(e).__name__}: {e}",
                    traceback=tb.format_exc(),
                    execution_time_ms=int(execution_time * 1000),
                )

        return args, kwargs

    def _create_timeout_response(
        self, request_id: str, timeout_seconds: float, start_time: float
    ):
        """Create response for timeout error."""
        execution_time = time.time() - start_time
        error_msg = f"Function execution exceeded timeout of {timeout_seconds}s"
        logger.error(f"[{request_id}] {error_msg}")

        return function_execution_pb2.ExecuteResponse(
            request_id=request_id,
            status=function_execution_pb2.ExecuteResponse.TIMEOUT,
            error=error_msg,
            execution_time_ms=int(execution_time * 1000),
        )

    def _create_error_response(
        self, request_id: str, error: Exception, start_time: float
    ):
        """Create response for function execution error."""
        execution_time = time.time() - start_time
        error_msg = f"{type(error).__name__}: {error}"
        error_trace = ''.join(
            tb.format_exception(type(error), error, error.__traceback__)
        )

        logger.error(f"[{request_id}] Function execution failed: {error_msg}")

        return function_execution_pb2.ExecuteResponse(
            request_id=request_id,
            status=function_execution_pb2.ExecuteResponse.ERROR,
            error=error_msg,
            traceback=error_trace,
            execution_time_ms=int(execution_time * 1000),
        )

    def _create_success_response(self, request_id: str, result: Any, start_time: float):
        """Create response for successful execution."""
        # Serialize result
        try:
            result_bytes = cloudpickle.dumps(result)
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"[{request_id}] Failed to serialize result: {e}")

            return function_execution_pb2.ExecuteResponse(
                request_id=request_id,
                status=function_execution_pb2.ExecuteResponse.ERROR,
                error=f"Failed to serialize result: {type(e).__name__}: {e}",
                traceback=tb.format_exc(),
                execution_time_ms=int(execution_time * 1000),
            )

        execution_time = time.time() - start_time
        logger.info(
            f"[{request_id}] Execution completed successfully in {int(execution_time * 1000)}ms"
        )

        return function_execution_pb2.ExecuteResponse(
            request_id=request_id,
            status=function_execution_pb2.ExecuteResponse.SUCCESS,
            result=result_bytes,
            execution_time_ms=int(execution_time * 1000),
        )

    def _create_unexpected_error_response(
        self, request_id: str, error: Exception, start_time: float
    ):
        """Create response for unexpected errors."""
        execution_time = time.time() - start_time
        error_msg = f"Unexpected error: {type(error).__name__}: {error}"
        error_trace = tb.format_exc()

        logger.error(f"[{request_id}] {error_msg}")

        return function_execution_pb2.ExecuteResponse(
            request_id=request_id,
            status=function_execution_pb2.ExecuteResponse.ERROR,
            error=error_msg,
            traceback=error_trace,
            execution_time_ms=int(execution_time * 1000),
        )


class GrpcServer:
    """Manages gRPC server lifecycle."""

    def __init__(self, config: RuntimeConfig, function: Any):
        self.config = config
        self.function = function
        self.server = None

    def start(self) -> None:
        """Start the gRPC server and block until termination."""
        # Create server with thread pool
        self.server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=self.config.grpc_max_workers),
            options=[
                ('grpc.max_send_message_length', self.config.grpc_max_message_length),
                (
                    'grpc.max_receive_message_length',
                    self.config.grpc_max_message_length,
                ),
            ],
        )

        # Add servicers
        function_execution_pb2_grpc.add_HealthServicer_to_server(
            HealthServicer(), self.server
        )
        function_execution_pb2_grpc.add_FunctionExecutorServicer_to_server(
            FunctionExecutorServicer(self.config, self.function), self.server
        )

        # Bind to port
        server_address = f'[::]:{self.config.port}'
        self.server.add_insecure_port(server_address)

        # Start server
        logger.info("=" * 40)
        logger.info("gRPC Server Starting")
        logger.info("=" * 40)
        logger.info(f"Address:           {server_address}")
        logger.info(f"Function:          {self.config.target_function}")
        logger.info(f"Function ID:       {self.config.function_id}")
        logger.info(f"Worker Threads:    {self.config.grpc_max_workers}")
        logger.info(
            f"Max Message Size:  {self.config.grpc_max_message_length // (1024*1024)}MB"
        )
        logger.info("=" * 40)

        self.server.start()
        logger.info("Server ready - accepting requests")

        # Block until termination
        try:
            self.server.wait_for_termination()
        except KeyboardInterrupt:
            logger.info("Server stopped by user")
            self.stop()

    def stop(self, grace_period: float = 5.0) -> None:
        """Stop the gRPC server gracefully."""
        if self.server:
            logger.info(f"Stopping server (grace period: {grace_period}s)")
            self.server.stop(grace_period)
            logger.info("Server stopped")


class WebEndpointServer:
    """Manages web endpoint server lifecycle."""

    def __init__(self, config: RuntimeConfig, function: Any, runtime_mode: RuntimeMode):
        self.config = config
        self.function = function
        self.web_callable = None
        self.lifespan_manager = None
        self.runtime_mode = runtime_mode

    def start(self) -> None:
        """Start the web endpoint server."""

        # Convert webhook config to dict for construct_webhook_callable
        # Note: webhook_config is guaranteed to be non-None here by runtime_mode check
        self.webhook_cfg = self.config.webhook_config
        assert (
            self.webhook_cfg is not None
        ), "webhook_config must be set for web endpoints"

        self._build_asgi_app()

        if self.runtime_mode == RuntimeMode.WEB_SERVER:
            self._run_web_server_mode()
        else:
            self._run_web_endpoint_mode()

    def _build_asgi_app(self) -> None:

        if self.webhook_cfg is None:
            raise RuntimeError("webhook_cfg must be set before building ASGI app")

        webhook_cfg = self.webhook_cfg  # Type narrowing for type checker
        webhook_type = webhook_cfg.type

        if webhook_type == "function":
            # Simple function endpoint - wrap in FastAPI
            logger.info(f"Creating FastAPI endpoint: {webhook_cfg.method} /")
            fastapi_app = asgi.fastapi_app(
                self.function,
                webhook_cfg.method,
                webhook_cfg.docs,
            )
            self.web_callable, self.lifespan_manager = asgi.asgi_app_wrapper(
                fastapi_app
            )

        elif webhook_type == "asgi_app":
            # User function returns an ASGI app
            logger.info("Creating ASGI app endpoint")
            logger.info(f"Function type: {type(self.function)}")

            # Call the function to get the ASGI app
            user_asgi_app = self.function()

            if not callable(user_asgi_app):
                raise TypeError(
                    f"Expected ASGI app (callable), got {type(user_asgi_app)}: {user_asgi_app}"
                )

            self.web_callable, self.lifespan_manager = asgi.asgi_app_wrapper(
                user_asgi_app
            )

        elif webhook_type == "wsgi_app":
            # User function returns a WSGI app
            logger.info("Creating WSGI app endpoint")
            user_wsgi_app = self.function()
            self.web_callable, self.lifespan_manager = asgi.wsgi_app_wrapper(
                user_wsgi_app
            )

        elif webhook_type == "web_server":
            # User function starts a web server - handled separately
            logger.info(f"Web server mode: function will start its own server")
            logger.info(f"Expected port: {webhook_cfg.port or self.config.port}")
            # Don't build ASGI app here, handled in _run_web_server_mode
            return

        else:
            raise ValueError(
                f"Unrecognized webhook type: {webhook_type}. "
                f"Supported types: function, asgi_app, wsgi_app, web_server"
            )

        logger.info("Web endpoint wrapper created successfully")

    def _run_web_server_mode(self) -> None:
        logger.info(f"Starting web server mode on port {self.config.port}")
        logger.info("Calling user function to start web server...")

        self.function()

        logger.info("Web server started, runtime process will keep running")
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            logger.info("Server stopped by user")

    def _run_web_endpoint_mode(self) -> None:
        """
        Run in web endpoint mode (function with @app.function(web=True)).
        Runtime manages the ASGI server.
        """
        logger.info(f"Starting ASGI server on port {self.config.port}")

        async def run_asgi_app():
            """Run the ASGI app with lifespan support."""
            try:
                # Run lifespan startup if available
                if self.lifespan_manager:
                    logger.info("Running lifespan startup")
                    await self.lifespan_manager.lifespan_startup()

                # Configure and start uvicorn server
                uvicorn_config = uvicorn.Config(  # type: ignore
                    app=self.web_callable,
                    host="0.0.0.0",
                    port=self.config.port,
                    log_level="info",
                    access_log=True,
                )
                server = uvicorn.Server(uvicorn_config)  # type: ignore

                logger.info("ASGI server starting...")
                await server.serve()

            except Exception as e:
                logger.error(f"ASGI server error: {e}")
                raise
            finally:
                # Run lifespan shutdown if available
                if self.lifespan_manager:
                    logger.info("Running lifespan shutdown")
                    try:
                        await self.lifespan_manager.lifespan_shutdown()
                    except Exception as e:
                        logger.error(f"Lifespan shutdown error: {e}")

        try:
            asyncio.run(run_asgi_app())
        except KeyboardInterrupt:
            logger.info("Server stopped by user")


if __name__ == "__main__":
    config = RuntimeConfig.from_environment()
    config.log_configuration()

    loader = FunctionLoader(config)
    actual_function = loader.load()

    if config.get_runtime_mode() == RuntimeMode.GRPC:
        # Start gRPC server
        try:
            grpc_server = GrpcServer(config, actual_function)
            grpc_server.start()
        except Exception as e:
            logger.error(f"gRPC server failed: {e}")
            tb.print_exc()
            sys.exit(1)

    else:
        # start web server
        try:
            web_server = WebEndpointServer(
                config, actual_function, config.get_runtime_mode()
            )
            web_server.start()
        except Exception as e:
            logger.error(f"Web server failed to start: {e}")
            tb.print_exc()
            sys.exit(1)
