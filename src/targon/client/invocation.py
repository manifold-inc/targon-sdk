import grpc.aio
import cloudpickle
import uuid
import time
import logging
import asyncio
from dataclasses import dataclass
from typing import Any, Optional, Dict, Tuple
from enum import Enum
from collections import OrderedDict

from targon.proto import function_execution_pb2
from targon.proto import function_execution_pb2_grpc
from targon.core.exceptions import TargonError

logger = logging.getLogger(__name__)


class FunctionInvocationError(TargonError):
    """Raised when function invocation fails."""

    def __init__(self, message: str, traceback: Optional[str] = None):
        super().__init__(message)
        self.traceback = traceback


class FunctionTimeoutError(TargonError):
    """Raised when function execution times out."""

    pass


class CircuitBreakerState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Circuit breaker pattern implementation for fault tolerance."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        success_threshold: int = 2,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold

        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        self._lock = asyncio.Lock()

    async def call(self, func, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        async with self._lock:
            if self.state == CircuitBreakerState.OPEN:
                if (
                    self.last_failure_time
                    and (time.time() - self.last_failure_time) >= self.recovery_timeout
                ):
                    logger.info(
                        "Circuit breaker transitioning to HALF_OPEN (testing recovery)"
                    )
                    self.state = CircuitBreakerState.HALF_OPEN
                    self.success_count = 0
                else:
                    raise FunctionInvocationError(
                        f"Circuit breaker is OPEN (too many failures). "
                        f"Retry after {self.recovery_timeout}s recovery period."
                    )

        try:
            result = await func(*args, **kwargs)
            await self._on_success()
            return result
        except Exception as e:
            await self._on_failure()
            raise

    async def _on_success(self) -> None:
        """Handle successful execution."""
        async with self._lock:
            self.failure_count = 0

            if self.state == CircuitBreakerState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.success_threshold:
                    logger.info("Circuit breaker transitioning to CLOSED (recovered)")
                    self.state = CircuitBreakerState.CLOSED
                    self.success_count = 0

    async def _on_failure(self) -> None:
        """Handle failed execution."""
        async with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.state == CircuitBreakerState.HALF_OPEN:
                logger.warning(
                    "Circuit breaker transitioning to OPEN (test request failed)"
                )
                self.state = CircuitBreakerState.OPEN
                self.success_count = 0
            elif self.state == CircuitBreakerState.CLOSED:
                if self.failure_count >= self.failure_threshold:
                    logger.warning(
                        f"Circuit breaker transitioning to OPEN "
                        f"({self.failure_count} consecutive failures)"
                    )
                    self.state = CircuitBreakerState.OPEN


class GRPCConnectionPool:
    """Connection pool for gRPC channels with LRU eviction."""

    def __init__(self, max_channels: int = 100) -> None:
        self.max_channels = max_channels
        self._channels: OrderedDict[Tuple[str, int], grpc.aio.Channel] = OrderedDict()
        self._stubs: Dict[
            Tuple[str, int], function_execution_pb2_grpc.FunctionExecutorStub
        ] = {}
        self._circuit_breakers: Dict[Tuple[str, int], CircuitBreaker] = {}
        self._lock = asyncio.Lock()

    async def get_stub(
        self,
        host: str,
        port: int,
        credentials: Optional[grpc.ChannelCredentials] = None,
    ) -> Tuple[function_execution_pb2_grpc.FunctionExecutorStub, CircuitBreaker]:
        """Get or create a secure gRPC stub for the given endpoint."""
        key = (host, port)

        async with self._lock:
            if key in self._channels:
                self._channels.move_to_end(key)
                return self._stubs[key], self._circuit_breakers[key]

            address = f"{host}:{port}"
            options = [
                ('grpc.max_send_message_length', 100 * 1024 * 1024),
                ('grpc.max_receive_message_length', 100 * 1024 * 1024),
                ('grpc.keepalive_time_ms', 30000),
                ('grpc.keepalive_timeout_ms', 10000),
                ('grpc.http2.max_pings_without_data', 0),
                ('grpc.keepalive_permit_without_calls', 1),
            ]

            creds = credentials or grpc.ssl_channel_credentials()
            channel = grpc.aio.secure_channel(address, creds, options=options)
            logger.debug(f"Created secure channel to {address}")

            stub = function_execution_pb2_grpc.FunctionExecutorStub(channel)
            circuit_breaker = CircuitBreaker()

            self._channels[key] = channel
            self._stubs[key] = stub
            self._circuit_breakers[key] = circuit_breaker

            if len(self._channels) > self.max_channels:
                oldest_key = next(iter(self._channels))
                old_channel = self._channels.pop(oldest_key)
                self._stubs.pop(oldest_key)
                self._circuit_breakers.pop(oldest_key)
                await old_channel.close()
                logger.debug(f"Evicted channel from pool: {oldest_key}")

            logger.debug(f"Created new channel in pool: {key}")
            return stub, circuit_breaker

    async def close_all(self) -> None:
        """Close all channels in the pool."""
        async with self._lock:
            for channel in self._channels.values():
                await channel.close()
            self._channels.clear()
            self._stubs.clear()
            self._circuit_breakers.clear()


_connection_pool = GRPCConnectionPool()


class GRPCFunctionClient:
    """Async gRPC client for invoking functions over secure TLS/SSL connections."""

    def __init__(
        self,
        host: str,
        port: int = 50051,
        timeout: int = 300,
        use_connection_pool: bool = True,
        credentials: Optional[grpc.ChannelCredentials] = None,
    ) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout
        self.use_connection_pool = use_connection_pool
        self.credentials = credentials

    async def invoke(
        self,
        *args,
        timeout: Optional[int] = None,
        request_id: Optional[str] = None,
        **kwargs,
    ) -> Any:
        """Invoke the function asynchronously with the given arguments."""
        if request_id is None:
            request_id = str(uuid.uuid4())

        timeout_seconds = timeout or self.timeout
        timeout_ms = int(timeout_seconds * 1000)

        try:
            args_bytes = cloudpickle.dumps(args) if args else b""
            kwargs_bytes = cloudpickle.dumps(kwargs) if kwargs else b""
        except Exception as e:
            logger.error(f"[{request_id}] Failed to serialize arguments: {e}")
            raise FunctionInvocationError(f"Failed to serialize arguments: {e}")

        request = function_execution_pb2.ExecuteRequest(
            request_id=request_id,
            args=args_bytes,
            kwargs=kwargs_bytes,
            timeout_ms=timeout_ms,
        )

        if self.use_connection_pool:
            stub, circuit_breaker = await _connection_pool.get_stub(
                self.host, self.port, self.credentials
            )
        else:
            stub, circuit_breaker = await self._create_direct_stub()

        try:
            result = await circuit_breaker.call(
                self._execute_request, stub, request, request_id, timeout_seconds
            )
            return result
        except (FunctionInvocationError, FunctionTimeoutError):
            raise
        except Exception as e:
            logger.error(
                f"[{request_id}] Unexpected error during invocation: {e}", exc_info=True
            )
            raise FunctionInvocationError(f"Unexpected error: {e}")

    async def _execute_request(
        self,
        stub: function_execution_pb2_grpc.FunctionExecutorStub,
        request: function_execution_pb2.ExecuteRequest,
        request_id: str,
        timeout_seconds: float,
    ) -> Any:
        """Execute the gRPC request."""
        start_time = time.time()

        try:
            logger.debug(
                f"[{request_id}] Invoking function via gRPC at {self.host}:{self.port}"
            )
            response = await stub.Execute(request, timeout=timeout_seconds + 5)

        except grpc.RpcError as e:
            elapsed = time.time() - start_time
            logger.error(
                f"[{request_id}] gRPC error after {elapsed:.2f}s: "
                f"code={e.code()}, details={e.details()}"
            )

            if e.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
                raise FunctionTimeoutError(
                    f"Function call timed out after {elapsed:.2f}s (timeout: {timeout_seconds}s)"
                )
            raise FunctionInvocationError(f"gRPC error: {e.details()}")

        elapsed_ms = int((time.time() - start_time) * 1000)
        logger.debug(f"[{request_id}] Response received in {elapsed_ms}ms")

        if response.status == function_execution_pb2.ExecuteResponse.SUCCESS:
            try:
                result = cloudpickle.loads(response.result)
                logger.info(
                    f"[{request_id}] ✅ Success "
                    f"(execution: {response.execution_time_ms}ms, total: {elapsed_ms}ms)"
                )
                return result
            except Exception as e:
                logger.error(f"[{request_id}] Failed to deserialize result: {e}")
                raise FunctionInvocationError(f"Failed to deserialize result: {e}")

        elif response.status == function_execution_pb2.ExecuteResponse.TIMEOUT:
            logger.error(f"[{request_id}] Function execution timeout: {response.error}")
            raise FunctionTimeoutError(response.error)

        elif response.status == function_execution_pb2.ExecuteResponse.ERROR:
            error_msg = response.error
            if response.traceback:
                error_msg += f"\n\nRemote traceback:\n{response.traceback}"
            logger.error(f"[{request_id}] Function execution error: {error_msg}")
            raise FunctionInvocationError(error_msg, traceback=response.traceback)

        else:
            logger.error(f"[{request_id}] Unknown response status: {response.status}")
            raise FunctionInvocationError(f"Unknown status: {response.status}")

    async def _create_direct_stub(
        self,
    ) -> Tuple[function_execution_pb2_grpc.FunctionExecutorStub, CircuitBreaker]:
        """Create a direct stub without using the connection pool."""
        address = f"{self.host}:{self.port}"
        options = [
            ('grpc.max_send_message_length', 100 * 1024 * 1024),
            ('grpc.max_receive_message_length', 100 * 1024 * 1024),
        ]

        creds = self.credentials or grpc.ssl_channel_credentials()
        channel = grpc.aio.secure_channel(address, creds, options=options)

        stub = function_execution_pb2_grpc.FunctionExecutorStub(channel)
        circuit_breaker = CircuitBreaker()
        return stub, circuit_breaker

    async def health_check(self) -> bool:
        """Check if the function server is healthy."""
        try:
            stub, _ = await _connection_pool.get_stub(
                self.host, self.port, self.credentials
            )

            request = function_execution_pb2.HealthCheckRequest()

            address = f"{self.host}:{self.port}"
            creds = self.credentials or grpc.ssl_channel_credentials()
            channel = grpc.aio.secure_channel(address, creds)

            health_stub = function_execution_pb2_grpc.HealthStub(channel)

            response = await health_stub.Check(request, timeout=5)
            await channel.close()

            return response.status == function_execution_pb2.HealthCheckResponse.SERVING
        except Exception as e:
            logger.debug(f"Health check failed: {e}")
            return False

    async def __aenter__(self) -> "GRPCFunctionClient":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        pass
