from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

    # Application settings
    LOG_LEVEL: str = "INFO"

    # vLLM Backend settings
    VLLM_BASE_URL: str = "http://localhost:10001"  # Base URL of the vLLM OpenAI-compatible server
    VLLM_CHAT_COMPLETIONS_ENDPOINT: str = "/v1/chat/completions" # Specific endpoint for chat
    VLLM_REQUEST_TIMEOUT: int = 1200 # seconds

    # Default Qwen non-thinking mode parameters
    DEFAULT_TEMPERATURE: float = 0.7
    DEFAULT_TOP_P: float = 0.8
    DEFAULT_TOP_K: int = 20
    DEFAULT_PRESENCE_PENALTY: float = 1.5
    DEFAULT_MIN_P: float = 0.0 # Qwen docs suggest MinP=0 for non-thinking

@lru_cache()
def get_settings() -> Settings:
    return Settings()

settings = get_settings()