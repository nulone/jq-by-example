"""
Safe execution of jq binary with resource limits.

This module provides the JQExecutor class for running jq filters in a controlled
subprocess environment with timeout and output size limits to prevent denial of
service attacks and resource exhaustion.
"""

import json
import logging
import shutil
import subprocess
from typing import Any

from src.domain import ExecutionResult

__all__ = ["ExecutionResult", "JQExecutor"]

logger = logging.getLogger(__name__)


class JQExecutor:
    """
    Executes jq filters safely with resource limits.

    This class wraps the jq binary and provides controlled execution with:
    - Timeout limits to prevent infinite loops
    - Output size limits to prevent memory exhaustion
    - Proper error handling for various failure modes

    Attributes:
        jq_path: Resolved path to the jq binary.
        timeout_sec: Maximum execution time in seconds.
        max_output_bytes: Maximum output size in bytes.
    """

    def __init__(
        self,
        jq_path: str = "jq",
        timeout_sec: float = 1.0,
        max_output_bytes: int = 1_000_000,
    ) -> None:
        """
        Initialize the JQ executor.

        Args:
            jq_path: Path to the jq binary. Defaults to 'jq' (uses PATH lookup).
            timeout_sec: Maximum execution time in seconds. Defaults to 1.0.
            max_output_bytes: Maximum output size in bytes. Defaults to 1MB.

        Raises:
            RuntimeError: If the jq binary is not found at the specified path.
        """
        resolved_path = shutil.which(jq_path)
        if resolved_path is None:
            raise RuntimeError(
                f"jq binary not found: '{jq_path}'. "
                "Please install jq (https://stedolan.github.io/jq/)"
            )

        self.jq_path = resolved_path
        self.timeout_sec = timeout_sec
        self.max_output_bytes = max_output_bytes

        logger.debug(
            "JQExecutor initialized: jq_path=%s, timeout_sec=%s, max_output_bytes=%s",
            self.jq_path,
            self.timeout_sec,
            self.max_output_bytes,
        )

    def run(self, filter_code: str, input_data: Any) -> ExecutionResult:
        """
        Execute a jq filter on the given input data.

        Args:
            filter_code: The jq filter expression to execute.
            input_data: The JSON-serializable input data to process.

        Returns:
            ExecutionResult containing stdout, stderr, exit code, and timeout status.
            Special exit codes:
            - 124: Execution timed out
            - 137: Output exceeded size limit (truncated)
        """
        # Serialize input data to JSON
        try:
            input_json = json.dumps(input_data)
        except (TypeError, ValueError) as e:
            logger.warning("Failed to serialize input data: %s", e)
            return ExecutionResult(
                stdout="",
                stderr=f"Failed to serialize input data: {e}",
                exit_code=1,
                is_timeout=False,
            )

        # SECURITY: Build command as list to prevent shell injection
        # filter_code is passed as an argument, NOT through shell
        cmd = [self.jq_path, "-M", "-c", filter_code]

        logger.debug(
            "Executing jq: filter='%s', input_size=%d bytes",
            filter_code,
            len(input_json),
        )

        try:
            # SECURITY: subprocess.run with cmd as list (not shell=True)
            # prevents command injection even with malicious filter_code
            result = subprocess.run(
                cmd,
                input=input_json,
                capture_output=True,
                text=True,
                timeout=self.timeout_sec,
                check=False,  # Don't raise on non-zero exit
                # shell=False is default - explicitly avoiding shell injection
            )

            stdout = result.stdout
            stderr = result.stderr
            exit_code = result.returncode

            # Check output size limit
            stdout_bytes = stdout.encode("utf-8")
            if len(stdout_bytes) > self.max_output_bytes:
                logger.warning(
                    "Output exceeded size limit: %d > %d bytes",
                    len(stdout_bytes),
                    self.max_output_bytes,
                )
                # Truncate at byte boundary, handling potential mid-character cuts
                truncated_bytes = stdout_bytes[: self.max_output_bytes]
                truncated_stdout = truncated_bytes.decode("utf-8", errors="ignore")
                return ExecutionResult(
                    stdout=truncated_stdout,
                    stderr="Output too large",
                    exit_code=137,
                    is_timeout=False,
                )

            # Strip trailing newlines for cleaner output comparison
            stdout = stdout.rstrip("\n")
            stderr = stderr.rstrip("\n")

            logger.debug(
                "jq execution completed: exit_code=%d, stdout_len=%d, stderr_len=%d",
                exit_code,
                len(stdout),
                len(stderr),
            )

            return ExecutionResult(
                stdout=stdout,
                stderr=stderr,
                exit_code=exit_code,
                is_timeout=False,
            )

        except subprocess.TimeoutExpired:
            logger.warning(
                "jq execution timed out after %s seconds",
                self.timeout_sec,
            )
            return ExecutionResult(
                stdout="",
                stderr=f"Execution timed out after {self.timeout_sec} seconds",
                exit_code=124,
                is_timeout=True,
            )
