"""openYuanrong remote sandbox command execution.

This sandbox infra is developed by the OpenYuanrong & Ant Akernel team.

Wraps remote sandbox lifecycle (create, run commands, cleanup and etc.)"""

from __future__ import annotations

import asyncio
import logging
import os
import shlex
import sys
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .base import ExecResult, Sandbox, _to_str
from .registry import register_sandbox

if TYPE_CHECKING:
    from .base import SandboxConfig

logger = logging.getLogger(__name__)

_sdk_initialized = False


def _resolve_sandbox_name() -> str | None:
    """Return ``{prefix}{random}`` when ``SANDBOX_NAME_PREFIX`` env is set."""
    prefix = os.getenv("SANDBOX_NAME_PREFIX")
    if not prefix:
        return None
    return f"{prefix}{uuid.uuid4().hex[:8]}"


def _load_sandbox_module() -> Any:
    """Configure env and select sandbox SDK via ``sys.modules`` injection."""
    global _sdk_initialized
    if _sdk_initialized:
        return sys.modules["openyuanrong_sandbox_sdk"]

    server = os.getenv("OPENYUANRONG_SERVER_ADDRESS")
    token = os.getenv("OPENYUANRONG_TOKEN")
    if not server or not token:
        raise ValueError(
            "OPENYUANRONG_SERVER_ADDRESS and OPENYUANRONG_TOKEN environment variables must be set for sandbox"
        )
    os.environ["TUNNEL_SSL_VERIFY"] = os.getenv("OPENYUANRONG_TUNNEL_SSL_VERIFY", "0")

    if os.getenv("USE_OPENYUANRONG_SDK", "0") == "1":
        try:
            import openyuanrong_sandbox_sdk as mod
        except ImportError as exc:
            raise ImportError(
                "USE_OPENYUANRONG_SDK=1 but openyuanrong_sandbox_sdk is not installed. "
                "Please install openyuanrong_sandbox_sdk or set USE_OPENYUANRONG_SDK=0."
            ) from exc
    else:
        os.environ["AKERNEL_SERVER_ADDRESS"] = server
        os.environ["AKERNEL_TOKEN"] = token
        try:
            import akernel_sdk as mod
        except ImportError as exc:
            raise ImportError(
                "USE_OPENYUANRONG_SDK=0 but the fallback SDK is not installed. "
                "Please install it or set USE_OPENYUANRONG_SDK=1."
            ) from exc
    sys.modules["openyuanrong_sandbox_sdk"] = mod
    _sdk_initialized = True
    return mod


class _OpenyuanrongShell:
    """Adapt openyuanrong SDK shell to the uni-agent sandbox shell handle.

    Converts the provider shell protocol (``shell.run`` / ``shell.kill``) into
    uni-agent's ``open_shell()`` contract: ``run`` → :class:`ExecResult`,
    ``close`` to release the session. Not killed between ``run`` calls.
    """

    def __init__(self, shell: Any) -> None:
        self._shell = shell

    async def run(self, command: str, *, timeout: float | None = None) -> ExecResult:
        result = await self._shell.run(command, timeout=int(timeout) if timeout else 60)
        return ExecResult(
            exit_code=getattr(result, "exit_code", -1),
            stdout=getattr(result, "stdout", "") or "",
            stderr=getattr(result, "stderr", "") or "",
        )

    async def close(self) -> None:
        try:
            await self._shell.kill()
        except Exception:
            pass


def _to_openyuanrong_image(image: str) -> str:
    """Map the provider-agnostic SWE-bench image ref to the openyuanrong registry.

    ``preprocess`` writes the canonical open-source ref
    ``swebench/sweb.eval.x86_64.<id>``; openyuanrong serves the same instance
    under a fully-qualified registry path with a ``:v2`` tag. Unknown prefixes
    (``python:3.12``, or an already fully-qualified openyuanrong image) are
    passed through unchanged -- the SDK resolves short names itself.
    """
    if image.startswith("swebench/"):
        return image.replace("swebench/", "swr.cn-east-3.myhuaweicloud.com/openyuanrong/swe-bench-verified/") + ":v2"
    return image

@register_sandbox("openyuanrong")
class OpenyuanrongSandbox(Sandbox):
    """Command execution via remote sandbox."""

    supports_shell = True

    def __init__(
        self,
        *,
        image: str,
        runtime_timeout: float = 3600.0,
        cpu: int = 1000,
        memory: int = 2048,
        cpu_limit: int = 4000,
        mem_limit: int = 8192,
        idle_timeout: int = 7200,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        name: str | None = None,
        mounts: list[Any] | None = None,
        upstream: str | None = None,
        proxy_port: int | None = None,
        port_forwardings: list[int] | None = None,
        **extra_kwargs: Any,
    ) -> None:
        self.image = image
        self.runtime_timeout = runtime_timeout
        self.cpu = cpu
        self.memory = memory
        self.cpu_limit = cpu_limit
        self.mem_limit = mem_limit
        self.idle_timeout = idle_timeout
        self.env = env
        self.cwd = cwd
        self.name = name
        self.mounts = mounts or []
        self.upstream = upstream
        self.proxy_port = proxy_port
        self.port_forwardings = port_forwardings or []
        self.extra_kwargs = extra_kwargs
        self._sandbox: Any = None

    @classmethod
    def from_config(cls, config: SandboxConfig) -> OpenyuanrongSandbox:
        return cls(
            image=_to_openyuanrong_image(config.image), runtime_timeout=config.runtime_timeout, **config.sandbox_kwargs
        )

    # ----- public: control plane -----
    async def start(self) -> None:
        if self._sandbox is not None:
            return
        sdk = _load_sandbox_module()
        sb_kwargs: dict[str, Any] = {
            "image": self.image,
            "cpu": self.cpu,
            "memory": self.memory,
            "cpu_limit": self.cpu_limit,
            "mem_limit": self.mem_limit,
            "idle_timeout": self.idle_timeout,
        }
        if self.mounts:
            sb_kwargs["mounts"] = [self._coerce_mount(m, sdk.Mount) for m in self.mounts]
        if self.env:
            sb_kwargs["env"] = self.env
        if self.cwd:
            sb_kwargs["cwd"] = self.cwd
        if self.upstream:
            sb_kwargs["upstream"] = self.upstream
        if self.proxy_port:
            sb_kwargs["proxy_port"] = self.proxy_port
        if self.port_forwardings:
            sb_kwargs["port_forwardings"] = list(self.port_forwardings)
        name = _resolve_sandbox_name()
        if name is not None:
            sb_kwargs["name"] = name
        sb_kwargs.update(self.extra_kwargs)
        self._sandbox = await asyncio.to_thread(lambda: sdk.Sandbox(**sb_kwargs))

    async def stop(self) -> None:
        """Kill the sandbox if still running."""
        if self._sandbox is not None:
            sid = getattr(self._sandbox, "sandbox_id", "?")
            try:
                await asyncio.to_thread(self._sandbox.kill)
                logger.info("openyuanrong sandbox %s killed", sid)
            except Exception as e:
                logger.warning("Failed to kill openyuanrong sandbox %s: %s", sid, e)
            self._sandbox = None

    async def is_alive(self) -> bool:
        sb = self._sandbox
        if sb is None:
            return False
        try:
            return bool(await asyncio.to_thread(sb.is_running))
        except Exception:
            return False

    _CONDA_ACTIVATE_INIT = (
        "source /opt/conda/etc/profile.d/conda.sh 2>/dev/null"
        " || source /opt/miniconda3/etc/profile.d/conda.sh 2>/dev/null;"
        " conda activate testbed 2>/dev/null || true"
    )

    async def open_shell(
        self,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> _OpenyuanrongShell:
        """Return a long-lived SDK shell (cwd/env persist across ``run`` calls)."""
        sb = self._require()
        shell = await sb.shells.create(cwd=cwd, envs=env)
        handle = _OpenyuanrongShell(shell)
        await handle.run(self._CONDA_ACTIVATE_INIT, timeout=30)
        return handle

    # ----- public: data plane (commands / files / ports) -----
    async def exec(
        self,
        argv: list[str],
        *,
        timeout: float | None = None,
        workdir: str | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecResult:
        """Fully overrides base ``exec``; does not use the base error-policy wrapper.

        Joins ``argv`` and delegates to :meth:`exec_shell` (Yuanrong
        ``commands.run``). Does not call ``super().exec`` or base ``_exec``.
        """
        return await self.exec_shell(shlex.join(argv), timeout=timeout, workdir=workdir, env=env)

    async def exec_shell(
        self,
        script: str,
        *,
        timeout: float | None = None,
        workdir: str | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecResult:
        """Fully overrides base ``exec_shell``; does not use ``bash -lc`` via base.

        Runs ``script`` directly through Yuanrong ``commands.run``. Does not call
        ``super().exec_shell`` / ``exec(["bash", "-lc", script])``. Owns its own
        timeout / alive error policy (mirrors base ``exec``, not base ``exec_shell``).
        """
        try:
            return await self._run_command(script, timeout=timeout, workdir=workdir, env=env)
        except Exception as exc:
            if self._is_timeout_error(exc):
                return ExecResult(exit_code=-1, stdout="", stderr=f"exec timed out after {timeout}s: {exc}")
            if not await self.is_alive():
                raise
            return ExecResult(exit_code=127, stdout="", stderr=str(exc))

    async def read_file(self, path: str) -> bytes:
        """Read via SDK ``files.read(..., format='bytes')``."""
        data = await asyncio.to_thread(lambda: self._require().files.read(path, format="bytes"))
        return data if isinstance(data, bytes) else bytes(data)

    async def write_file(self, path: str, content: bytes | str) -> None:
        """Write via SDK ``files.write``."""
        data: bytes | str = content.encode("utf-8") if isinstance(content, str) else content
        await asyncio.to_thread(self._require().files.write, path, data)

    async def upload(self, local_path: Path | str, remote_path: str) -> None:
        """Upload file or directory via SDK ``files.copy_from_local``."""
        await asyncio.to_thread(self._require().files.copy_from_local, str(local_path), str(remote_path))

    async def download(self, remote_path: str, local_path: Path | str) -> None:
        """Download file or directory via SDK ``files.copy_to_local``."""
        await asyncio.to_thread(self._require().files.copy_to_local, str(remote_path), str(local_path))

    async def expose_port(self, port: int) -> str:
        """Return gateway URL for a port declared in ``port_forwardings``."""
        return await asyncio.to_thread(self._require().get_port_url, port)

    def get_port_url(self, port: int) -> str:
        return self._require().get_port_url(port)

    def get_tunnel_url(self) -> str:
        return self._require().get_tunnel_url()

    # ----- private helpers -----
    def _require(self) -> Any:
        if self._sandbox is None:
            raise RuntimeError("OpenyuanrongSandbox not started; call start() first")
        return self._sandbox

    @staticmethod
    def _coerce_mount(m: Any, MountCls: type) -> Any:
        """Accept a ``Mount`` instance or a dict (``target`` + ``image_url``/``s3_config``)."""
        if isinstance(m, dict):
            return MountCls(**m)
        return m

    def _is_timeout_error(self, exc: BaseException) -> bool:
        return type(exc).__name__ == "CommandTimeoutError" or super()._is_timeout_error(exc)

    async def _run_command(
        self,
        cmd: str,
        *,
        timeout: float | None = None,
        workdir: str | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecResult:
        """Run a shell command string once via ``commands.run``."""
        sb = self._require()
        timeout_i = int(timeout) if timeout else 60
        result = await asyncio.to_thread(lambda: sb.commands.run(cmd, envs=env, cwd=workdir, timeout=timeout_i))
        return ExecResult(
            exit_code=int(getattr(result, "exit_code", -99)),
            stdout=_to_str(getattr(result, "stdout", "")),
            stderr=_to_str(getattr(result, "stderr", "")),
        )

    async def _exec(
        self,
        argv: list[str],
        *,
        timeout: float | None = None,
        workdir: str | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecResult:
        """Intentionally unimplemented.

        Base marks ``_exec`` abstract, so this stub exists only to instantiate the
        class. Public :meth:`exec` is fully overridden and never calls this (or
        ``super().exec``).
        """
        raise NotImplementedError("OpenyuanrongSandbox overrides exec(); _exec is unused")
