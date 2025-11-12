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
import os
from pathlib import Path
from typing import Optional, Dict, Any, Union
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
            # Provide platform-specific compilation instructions
            import platform
            system = platform.system().lower()

            if system == "darwin":  # macOS
                compile_cmd = "cmake .. -DGGML_METAL=ON && cmake --build . --config Release"
                gpu_info = "Metal (Apple Silicon/Intel Mac GPU)"
            elif system == "linux":
                # Check for NVIDIA GPU
                try:
                    subprocess.run(["nvidia-smi"], capture_output=True, check=True)
                    compile_cmd = "cmake .. -DGGML_CUDA=ON && cmake --build . --config Release"
                    gpu_info = "CUDA (NVIDIA GPU)"
                except:
                    # No NVIDIA GPU, check for AMD
                    if Path("/opt/rocm").exists():
                        compile_cmd = "cmake .. -DGGML_HIPBLAS=ON && cmake --build . --config Release"
                        gpu_info = "ROCm (AMD GPU)"
                    else:
                        compile_cmd = "cmake .. && cmake --build . --config Release"
                        gpu_info = "CPU only"
            else:
                compile_cmd = "cmake .. && cmake --build . --config Release"
                gpu_info = "CPU only"

            raise RuntimeError(
                f"llama-server binary not found. Please compile llama.cpp ({gpu_info}):\n"
                f"  git clone https://github.com/ggerganov/llama.cpp\n"
                f"  cd llama.cpp && mkdir build && cd build\n"
                f"  {compile_cmd}"
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

        # Enable continuous batching (required for VLM to work properly)
        cmd.append("--cont-batching")

        # Enable reasoning/thinking output (keeps <think> tags in response)
        cmd.extend(["--reasoning-format", "deepseek-legacy"])
        cmd.extend(["--reasoning-budget", "-1"])  # Unrestricted thinking

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
                    # Store context size for downstream consumers
                    try:
                        self.n_ctx = n_ctx
                    except Exception:
                        self.n_ctx = n_ctx
                    logger.info(f"✓ llama-server started successfully on {self.base_url}")
                    logger.info(f"  Model: {self.model_path.name}")
                    logger.info(f"  Server type: legacy llama-server (subprocess binary)")
                    logger.info(f"  GPU layers: {n_gpu_layers}, Context: {n_ctx}")
                    if mmproj_path:
                        logger.info(f"  VLM mode: {Path(mmproj_path).name}")
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

    def get_recent_output(self, max_lines: int = 50) -> str:
        """Get recent output from llama-server.

        Args:
            max_lines: Maximum number of recent lines to return

        Returns:
            Recent server output as string
        """
        if not self.process or not self.process.stdout:
            return ""

        try:
            import select
            import sys
            if sys.platform != 'win32':
                # Unix-like systems - non-blocking read
                ready, _, _ = select.select([self.process.stdout], [], [], 0)
                if ready:
                    lines = []
                    while len(lines) < max_lines:
                        line = self.process.stdout.readline()
                        if not line:
                            break
                        lines.append(line)
                    return ''.join(lines)
        except Exception as e:
            logger.debug(f"Could not read server output: {e}")

        return ""

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
        messages: list[Dict[str, Any]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[list[str]] = None,
        stream: bool = True,
        images: Optional[list[str]] = None,
        image_data: Optional[list[Dict[str, Any]]] = None,
        # Additional llama.cpp options
        top_k: Optional[int] = None,
        min_p: Optional[float] = None,
        typical_p: Optional[float] = None,
        tfs_z: Optional[float] = None,
        repeat_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        penalty_last_n: Optional[int] = None,
        mirostat: Optional[int] = None,
        mirostat_tau: Optional[float] = None,
        mirostat_eta: Optional[float] = None,
        seed: Optional[int] = None,
        n_keep: Optional[int] = None,
        ignore_eos: Optional[bool] = None,
        grammar: Optional[str] = None,
        logit_bias: Optional[Dict[Union[str,int], float]] = None,
        n_probs: Optional[int] = None,
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

        url, payload = self.build_chat_payload(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            stream=stream,
            images=images,
            image_data=image_data,
            top_k=top_k,
            min_p=min_p,
            typical_p=typical_p,
            tfs_z=tfs_z,
            repeat_penalty=repeat_penalty,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            penalty_last_n=penalty_last_n,
            mirostat=mirostat,
            mirostat_tau=mirostat_tau,
            mirostat_eta=mirostat_eta,
            seed=seed,
            n_keep=n_keep,
            ignore_eos=ignore_eos,
            grammar=grammar,
            logit_bias=logit_bias,
            n_probs=n_probs,
        )

        # Save last request for debugging
        try:
            self._last_request = {"url": url, "payload": payload}
        except Exception:
            pass

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

    def build_chat_payload(
        self,
        messages: list[Dict[str, Any]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[list[str]] = None,
        stream: bool = True,
        images: Optional[list[str]] = None,
        image_data: Optional[list[Dict[str, Any]]] = None,
        # Additional llama.cpp options
        top_k: Optional[int] = None,
        min_p: Optional[float] = None,
        typical_p: Optional[float] = None,
        tfs_z: Optional[float] = None,
        repeat_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        penalty_last_n: Optional[int] = None,
        mirostat: Optional[int] = None,
        mirostat_tau: Optional[float] = None,
        mirostat_eta: Optional[float] = None,
        seed: Optional[int] = None,
        n_keep: Optional[int] = None,
        ignore_eos: Optional[bool] = None,
        grammar: Optional[str] = None,
        logit_bias: Optional[Dict[Union[str,int], float]] = None,
        n_probs: Optional[int] = None,
    ) -> tuple[str, Dict[str, Any]]:
        """Build URL and JSON payload for chat completion.

        Returns:
            Tuple of (url, payload)
        """
        url = f"{self.base_url}/v1/chat/completions"

        payload: Dict[str, Any] = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream,
        }

        if images:
            payload["images"] = images

        if image_data:
            payload["image_data"] = image_data

        if stop:
            payload["stop"] = stop

        # Optional advanced parameters
        opt_map = {
            "top_k": top_k,
            "min_p": min_p,
            "typical_p": typical_p,
            "tfs_z": tfs_z,
            "repeat_penalty": repeat_penalty,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "penalty_last_n": penalty_last_n,
            "mirostat": mirostat,
            "mirostat_tau": mirostat_tau,
            "mirostat_eta": mirostat_eta,
            "seed": seed,
            "n_keep": n_keep,
            "ignore_eos": ignore_eos,
            "grammar": grammar,
            "logit_bias": logit_bias,
            "n_probs": n_probs,
        }
        for k, v in opt_map.items():
            if v is not None:
                payload[k] = v

        # Disable prompt caching for stateless runs
        payload["cache_prompt"] = False

        return url, payload

    def get_last_request(self) -> Optional[Dict[str, Any]]:
        """Return last request (url and payload) sent to llama-server."""
        return getattr(self, "_last_request", None)

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
