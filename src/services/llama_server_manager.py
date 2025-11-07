"""llama-server process manager for GPU-accelerated inference with KV cache reuse.

This module manages llama-server processes to enable:
- Full GPU utilization (matches llama.cpp WebUI performance)
- Persistent KV cache across conversation turns
- 40+ tokens/s performance for follow-up messages

References:
- https://github.com/ggml-org/llama.cpp/discussions/16938
- https://github.com/abetlen/llama-cpp-python/tree/main/llama_cpp/server
"""

import subprocess
import sys
import time
import requests
from pathlib import Path
from typing import Optional, Dict, Any
from loguru import logger
import psutil


class LlamaServerManager:
    """Manages llama-server process lifecycle and provides HTTP API client."""

    def __init__(self, model_path: str, host: str = "127.0.0.1", port: int = 8080):
        """Initialize server manager.

        Args:
            model_path: Path to GGUF model file
            host: Server host (default: 127.0.0.1)
            port: Server port (default: 8080)
        """
        self.model_path = Path(model_path)
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self.process: Optional[subprocess.Popen] = None
        self.slot_id = 0  # Use slot 0 for conversation (enables KV cache reuse)

    def start(
        self,
        n_gpu_layers: int = -1,
        n_ctx: int = 4096,
        n_batch: int = 2048,
        n_ubatch: int = 512,
        n_threads: int = 8,
        n_threads_batch: Optional[int] = None,
        mmproj_path: Optional[str] = None,
    ) -> bool:
        """Start llama-server process with optimized parameters.

        Args:
            n_gpu_layers: GPU layers (-1 = all, 0 = CPU only)
            n_ctx: Context window size
            n_batch: Logical batch size
            n_ubatch: Physical batch size
            n_threads: Generation threads
            n_threads_batch: Batch processing threads
            mmproj_path: Path to mmproj file for VLM support (2025 feature)

        Returns:
            True if server started successfully
        """
        if self.is_running():
            logger.info(f"llama-server already running on {self.base_url}")
            return True

        # Build command using native llama.cpp binary (not Python wrapper)
        # This matches the llama.cpp WebUI command format
        from pathlib import Path
        import os

        # Find llama-server binary
        possible_paths = [
            Path("llama.cpp/build/bin/llama-server"),  # Relative to cwd
            Path(__file__).parent.parent.parent / "llama.cpp/build/bin/llama-server",  # Relative to this file
        ]

        llama_server_bin = None
        for path in possible_paths:
            if path.exists():
                llama_server_bin = str(path)
                break

        if not llama_server_bin:
            raise RuntimeError(
                "llama-server binary not found. Please compile llama.cpp:\n"
                "  git clone https://github.com/ggerganov/llama.cpp\n"
                "  cd llama.cpp && mkdir build && cd build\n"
                "  cmake .. -DGGML_CUDA=ON && cmake --build . --config Release"
            )

        # Use native binary with parameters matching working WebUI command
        cmd = [
            llama_server_bin,
            "-m", str(self.model_path),  # Model path
            "-c", str(n_ctx),             # Context size
            "--host", self.host,
            "--port", str(self.port),
            "-ngl", str(n_gpu_layers if n_gpu_layers >= 0 else 99),  # GPU layers (99 = all)
            "-b", str(n_batch),           # Batch size
            "-ub", str(n_ubatch),         # Micro batch size
            "-t", str(n_threads),         # Threads
        ]

        if n_threads_batch:
            cmd.extend(["-tb", str(n_threads_batch)])

        # Add mmproj for VLM support (2025 feature)
        if mmproj_path:
            cmd.extend(["--mmproj", str(mmproj_path)])
            logger.info(f"VLM mode enabled with mmproj: {Path(mmproj_path).name}")

        # Enable continuous batching for multi-turn conversations
        cmd.append("--cont-batching")

        logger.info(f"Starting llama-server: {' '.join(cmd)}")

        try:
            # Start server process with unbuffered output
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Redirect stderr to stdout for combined output
                text=True,
                bufsize=1,  # Line buffered
            )

            # Wait for server to be ready
            max_wait = 60  # 60 seconds timeout (model loading can take time)
            start_time = time.time()

            while time.time() - start_time < max_wait:
                # Check if process crashed
                if self.process.poll() is not None:
                    # Process ended prematurely - read output
                    stdout, _ = self.process.communicate()
                    logger.error(f"llama-server process crashed during startup!")
                    logger.error(f"Process output:\n{stdout}")
                    return False

                # Check if server is responding
                if self.is_running():
                    logger.info(f"✓ llama-server started successfully on {self.base_url}")
                    return True

                time.sleep(0.5)

            # Timeout - read any available output
            logger.error(f"llama-server failed to start within {max_wait} seconds")

            # Try to read partial output (non-blocking)
            if self.process and self.process.poll() is None:
                # Process still running - try to get some output
                try:
                    import select
                    import sys
                    if sys.platform != 'win32':
                        # Unix-like systems
                        ready, _, _ = select.select([self.process.stdout], [], [], 0)
                        if ready:
                            output = self.process.stdout.read()
                            logger.error(f"Partial server output:\n{output}")
                except Exception:
                    pass

            self.stop()
            return False

        except Exception as e:
            logger.error(f"Failed to start llama-server: {e}")
            if self.process:
                try:
                    stdout, _ = self.process.communicate(timeout=1)
                    logger.error(f"Process output:\n{stdout}")
                except Exception:
                    pass
            self.stop()
            return False

    def is_running(self) -> bool:
        """Check if server is running and responding.

        Returns:
            True if server is healthy
        """
        try:
            response = requests.get(f"{self.base_url}/health", timeout=2)
            return response.status_code == 200
        except:
            return False

    def stop(self):
        """Stop llama-server process."""
        if self.process:
            try:
                # Terminate gracefully
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    # Force kill if doesn't terminate
                    self.process.kill()
                    self.process.wait()
                logger.info("llama-server stopped")
            except Exception as e:
                logger.warning(f"Error stopping llama-server: {e}")
            finally:
                self.process = None

    def chat_completion(
        self,
        messages: list[Dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[list[str]] = None,
        stream: bool = True,
    ) -> Any:
        """Send chat completion request to llama-server.

        Uses slot-based caching for KV cache reuse across turns.

        Args:
            messages: List of message dicts with 'role' and 'content'
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            stop: Stop sequences
            stream: Enable streaming response

        Returns:
            Response object (streaming iterator or dict)
        """
        if not self.is_running():
            raise RuntimeError("llama-server is not running. Call start() first.")

        url = f"{self.base_url}/v1/chat/completions"

        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream,
        }

        if stop:
            payload["stop"] = stop

        # Use slot ID for KV cache reuse
        # Same slot = reuse KV cache from previous turns
        payload["cache_prompt"] = True  # Enable prompt caching
        payload["slot_id"] = self.slot_id

        if stream:
            # Return streaming response
            response = requests.post(url, json=payload, stream=True, timeout=120)
            response.raise_for_status()
            return self._parse_stream(response)
        else:
            # Return full response
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            return response.json()

    def _parse_stream(self, response):
        """Parse SSE (Server-Sent Events) streaming response.

        Yields:
            Parsed JSON chunks from stream
        """
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data = line[6:]  # Remove 'data: ' prefix
                    if data.strip() == '[DONE]':
                        break
                    try:
                        import json
                        yield json.loads(data)
                    except json.JSONDecodeError:
                        continue

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - stop server."""
        self.stop()

    def __del__(self):
        """Destructor - ensure server is stopped."""
        self.stop()
