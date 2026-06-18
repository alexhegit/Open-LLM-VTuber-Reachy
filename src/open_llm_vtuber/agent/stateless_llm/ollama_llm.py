import atexit

import httpx
import requests
from loguru import logger

from .openai_compatible_llm import AsyncLLM


class OllamaLLM(AsyncLLM):
    """Ollama backend using OpenAI-compatible API; HTTP client ignores proxy env for localhost."""

    def __init__(
        self,
        model: str,
        base_url: str,
        llm_api_key: str = "z",
        organization_id: str = "z",
        project_id: str = "z",
        temperature: float = 1.0,
        keep_alive: float = -1,
        unload_at_exit: bool = True,
    ):
        self.keep_alive = keep_alive
        self.unload_at_exit = unload_at_exit
        self.cleaned = False
        self._ollama_http_client = httpx.AsyncClient(trust_env=False)
        self._ollama_requests_session = requests.Session()
        self._ollama_requests_session.trust_env = False
        super().__init__(
            model=model,
            base_url=base_url,
            llm_api_key=llm_api_key,
            organization_id=organization_id,
            project_id=project_id,
            temperature=temperature,
            http_client=self._ollama_http_client,
        )
        try:
            logger.info("Preloading model for Ollama")
            logger.debug(
                self._ollama_requests_session.post(
                    base_url.replace("/v1", "") + "/api/chat",
                    json={
                        "model": model,
                        "keep_alive": keep_alive,
                    },
                )
            )
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Failed to preload model: {e}")
            logger.critical(
                "Fail to connect to Ollama backend. Is Ollama server running? Try running `ollama list` to start the server and try again.\nThe AI will repeat 'Error connecting chat endpoint' until the server is running."
            )
        except Exception as e:
            logger.error(f"Failed to preload model: {e}")
        if unload_at_exit:
            atexit.register(self.cleanup)

    def __del__(self) -> None:
        """Destructor to unload the model."""
        self.cleanup()

    def cleanup(self) -> None:
        """Unload the model when exiting."""
        if not self.cleaned and self.unload_at_exit:
            logger.info(f"Ollama: Unloading model: {self.model}")
            logger.debug(
                self._ollama_requests_session.post(
                    self.base_url.replace("/v1", "") + "/api/chat",
                    json={
                        "model": self.model,
                        "keep_alive": 0,
                    },
                )
            )
            self.cleaned = True
        try:
            self._ollama_requests_session.close()
        except Exception:
            pass
