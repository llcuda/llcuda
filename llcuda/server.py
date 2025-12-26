"""
llcuda.server - Server Management for llama-server

This module provides automatic management of llama-server lifecycle,
including finding, starting, and stopping the server process.
"""

from typing import Optional, Dict, Any
import os
import subprocess
import time
import signal
from pathlib import Path
import requests


class ServerManager:
    """
    Manages llama-server lifecycle automatically.

    This class handles:
    - Finding llama-server executable in common locations
    - Starting llama-server with appropriate parameters
    - Health checking and waiting for server readiness
    - Stopping server gracefully

    Examples:
        >>> manager = ServerManager()
        >>> manager.start_server(
        ...     model_path="/path/to/model.gguf",
        ...     gpu_layers=99
        ... )
        >>> # Server is now running
        >>> manager.stop_server()
    """

    def __init__(self, server_url: str = "http://127.0.0.1:8090"):
        """
        Initialize the server manager.

        Args:
            server_url: URL where server will be accessible
        """
        self.server_url = server_url
        self.server_process: Optional[subprocess.Popen] = None
        self._server_path: Optional[Path] = None

    def find_llama_server(self) -> Optional[Path]:
        """
        Find llama-server executable in common locations.

        Searches in the following order:
        1. LLAMA_SERVER_PATH environment variable
        2. LLAMA_CPP_DIR environment variable + /bin/llama-server
        3. User's project directory (llama-cpp-cuda)
        4. System PATH locations
        5. ~/.llcuda/bin/llama-server

        Returns:
            Path to llama-server executable, or None if not found
        """
        # Priority 1: Explicit path from environment
        env_path = os.getenv('LLAMA_SERVER_PATH')
        if env_path:
            path = Path(env_path)
            if path.exists() and path.is_file():
                return path

        # Priority 2: LLAMA_CPP_DIR
        llama_cpp_dir = os.getenv('LLAMA_CPP_DIR')
        if llama_cpp_dir:
            path = Path(llama_cpp_dir) / 'bin' / 'llama-server'
            if path.exists():
                return path

        # Priority 3: Project-specific location (llama-cpp-cuda)
        project_paths = [
            Path('/media/waqasm86/External1/Project-Nvidia/llama-cpp-cuda/bin/llama-server'),
            Path.home() / 'llama-cpp-cuda' / 'bin' / 'llama-server',
            Path.cwd() / 'llama-cpp-cuda' / 'bin' / 'llama-server',
        ]

        for path in project_paths:
            if path.exists():
                return path

        # Priority 4: System locations
        system_paths = [
            '/usr/local/bin/llama-server',
            '/usr/bin/llama-server',
            '/opt/llama.cpp/bin/llama-server',
            '/opt/homebrew/bin/llama-server',  # macOS Homebrew
        ]

        for path_str in system_paths:
            path = Path(path_str)
            if path.exists():
                return path

        # Priority 5: User's home directory
        home_path = Path.home() / '.llcuda' / 'bin' / 'llama-server'
        if home_path.exists():
            return home_path

        return None

    def check_server_health(self, timeout: float = 2.0) -> bool:
        """
        Check if llama-server is responding to health checks.

        Args:
            timeout: Request timeout in seconds

        Returns:
            True if server is healthy, False otherwise
        """
        try:
            response = requests.get(
                f"{self.server_url}/health",
                timeout=timeout
            )
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def start_server(
        self,
        model_path: str,
        port: int = 8090,
        host: str = "127.0.0.1",
        gpu_layers: int = 99,
        ctx_size: int = 2048,
        n_parallel: int = 1,
        timeout: int = 60,
        verbose: bool = True,
        **kwargs
    ) -> bool:
        """
        Start llama-server with specified configuration.

        Args:
            model_path: Path to GGUF model file
            port: Server port (default: 8090)
            host: Server host (default: 127.0.0.1)
            gpu_layers: Number of layers to offload to GPU (default: 99)
            ctx_size: Context size (default: 2048)
            n_parallel: Number of parallel sequences (default: 1)
            timeout: Max seconds to wait for server startup (default: 60)
            verbose: Print status messages (default: True)
            **kwargs: Additional server arguments

        Returns:
            True if server started successfully, False otherwise

        Raises:
            FileNotFoundError: If llama-server executable not found
            RuntimeError: If server fails to start
        """
        # Check if server is already running
        if self.check_server_health(timeout=1.0):
            if verbose:
                print(f"✓ llama-server already running at {self.server_url}")
            return True

        # Find llama-server executable
        if self._server_path is None:
            self._server_path = self.find_llama_server()

        if self._server_path is None:
            raise FileNotFoundError(
                "llama-server not found. Please install llama.cpp or set LLAMA_SERVER_PATH.\n"
                "See: https://github.com/ggerganov/llama.cpp for installation instructions."
            )

        # Verify model exists
        model_path_obj = Path(model_path)
        if not model_path_obj.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Build command
        cmd = [
            str(self._server_path),
            '-m', str(model_path_obj.absolute()),
            '--host', host,
            '--port', str(port),
            '-ngl', str(gpu_layers),
            '-c', str(ctx_size),
            '--parallel', str(n_parallel),
        ]

        # Add additional arguments
        for key, value in kwargs.items():
            if key.startswith('-'):
                cmd.extend([key, str(value)])
            else:
                cmd.extend([f'--{key.replace("_", "-")}', str(value)])

        if verbose:
            print(f"Starting llama-server...")
            print(f"  Executable: {self._server_path}")
            print(f"  Model: {model_path_obj.name}")
            print(f"  GPU Layers: {gpu_layers}")
            print(f"  Context Size: {ctx_size}")
            print(f"  Server URL: {self.server_url}")

        # Start server process
        try:
            self.server_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True
            )
        except Exception as e:
            raise RuntimeError(f"Failed to start llama-server: {e}")

        # Wait for server to be ready
        if verbose:
            print(f"Waiting for server to be ready...", end='', flush=True)

        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.check_server_health(timeout=1.0):
                if verbose:
                    elapsed = time.time() - start_time
                    print(f" ✓ Ready in {elapsed:.1f}s")
                return True

            # Check if process died
            if self.server_process.poll() is not None:
                stderr = self.server_process.stderr.read().decode('utf-8', errors='ignore')
                raise RuntimeError(
                    f"llama-server process died unexpectedly.\n"
                    f"Error output:\n{stderr}"
                )

            if verbose:
                print('.', end='', flush=True)
            time.sleep(1)

        # Timeout reached
        if verbose:
            print(" ✗ Timeout")
        self.stop_server()
        raise RuntimeError(f"Server failed to start within {timeout} seconds")

    def stop_server(self, timeout: float = 10.0) -> bool:
        """
        Stop the running llama-server gracefully.

        Args:
            timeout: Max seconds to wait for graceful shutdown

        Returns:
            True if server stopped successfully, False otherwise
        """
        if self.server_process is None:
            return True

        if self.server_process.poll() is not None:
            # Already stopped
            self.server_process = None
            return True

        try:
            # Try graceful shutdown first (SIGTERM)
            self.server_process.terminate()

            # Wait for graceful shutdown
            try:
                self.server_process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                # Force kill if graceful shutdown failed
                self.server_process.kill()
                self.server_process.wait(timeout=5.0)

            self.server_process = None
            return True

        except Exception as e:
            print(f"Error stopping server: {e}")
            return False

    def get_server_info(self) -> Dict[str, Any]:
        """
        Get information about the running server.

        Returns:
            Dictionary with server information
        """
        info = {
            'running': False,
            'url': self.server_url,
            'process_id': None,
            'executable': str(self._server_path) if self._server_path else None,
        }

        if self.server_process and self.server_process.poll() is None:
            info['running'] = True
            info['process_id'] = self.server_process.pid

        return info

    def restart_server(self, model_path: str, **kwargs) -> bool:
        """
        Restart the server with new configuration.

        Args:
            model_path: Path to GGUF model file
            **kwargs: Server configuration parameters

        Returns:
            True if restart successful, False otherwise
        """
        self.stop_server()
        time.sleep(1)  # Brief pause before restart
        return self.start_server(model_path, **kwargs)

    def __del__(self):
        """Cleanup: stop server when manager is destroyed."""
        if self.server_process is not None:
            self.stop_server()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit: stop server."""
        self.stop_server()
        return False
