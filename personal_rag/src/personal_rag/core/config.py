"""Pydantic Settings and centralized configuration management with graceful fallback."""

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional, List, Any, Dict
from pydantic import Field, model_validator

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
    _HAS_SETTINGS = True
except ImportError:
    from pydantic import BaseModel as BaseSettings
    SettingsConfigDict = None
    _HAS_SETTINGS = False

from personal_rag.core.hardware import HardwareManager


_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _resolve_path(env_var: str, default_rel: str) -> Path:
    val = os.getenv(env_var)
    if val:
        p = Path(val)
        return p if p.is_absolute() else (_PROJECT_ROOT / p).resolve()
    return (_PROJECT_ROOT / default_rel).resolve()


class Settings(BaseSettings):
    """Application-wide configuration parameters with strict type safety.

    Accepts both lowercase (``cpu_only``) and uppercase (``CPU_ONLY``) kwargs
    so the class works naturally whether populated from env-vars or test code.
    """

    if _HAS_SETTINGS and SettingsConfigDict is not None:
        model_config = SettingsConfigDict(
            env_file=".env",
            env_file_encoding="utf-8",
            extra="ignore",
        )

    @model_validator(mode="before")
    @classmethod
    def _normalise_case(cls, values: Any) -> Any:
        """Fold any UPPERCASE_KEY into its lowercase_key equivalent."""
        if not isinstance(values, dict):
            return values
        normalised: Dict[str, Any] = {}
        for k, v in values.items():
            normalised[k.lower()] = v
        return normalised

    # ── Non-Negotiable Hardware Policy ───────────────────────────────────────
    cpu_only: bool = Field(
        default_factory=lambda: os.getenv("CPU_ONLY", "true").lower() in ("true", "1", "yes")
    )
    gpu_execution_approved: bool = Field(
        default_factory=lambda: os.getenv("GPU_EXECUTION_APPROVED", "false").lower() in ("true", "1", "yes")
    )
    cloud_instance_autostart: bool = Field(
        default_factory=lambda: os.getenv("CLOUD_INSTANCE_AUTOSTART", "false").lower() in ("true", "1", "yes")
    )
    model_download_autostart: bool = Field(
        default_factory=lambda: os.getenv("MODEL_DOWNLOAD_AUTOSTART", "false").lower() in ("true", "1", "yes")
    )
    device: str = Field(
        default_factory=lambda: os.getenv("DEVICE", "cpu")
    )

    # ── Storage Directories ──────────────────────────────────────────────────
    sqlite_db_path: Path = Field(
        default_factory=lambda: _resolve_path("SQLITE_DB_PATH", "data/knowledge_base.db")
    )
    qdrant_storage_path: Path = Field(
        default_factory=lambda: _resolve_path("QDRANT_STORAGE_PATH", "data/qdrant")
    )
    qdrant_collection_name: str = Field(
        default_factory=lambda: os.getenv("QDRANT_COLLECTION_NAME", "personal_knowledge_base")
    )
    documents_dir: Path = Field(
        default_factory=lambda: _resolve_path("DOCUMENTS_DIR", "data/documents")
    )
    cache_dir: Path = Field(
        default_factory=lambda: _resolve_path("CACHE_DIR", "cache")
    )

    # ── Embeddings Configuration ─────────────────────────────────────────────
    embedding_provider: str = Field(
        default_factory=lambda: os.getenv("EMBEDDING_PROVIDER", "sentence_transformers")
    )
    embedding_model_name: str = Field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5")
    )
    embedding_dimension: int = Field(
        default_factory=lambda: int(os.getenv("EMBEDDING_DIMENSION", "384"))
    )
    embedding_batch_size: int = Field(
        default_factory=lambda: int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
    )

    # ── Chunking Settings ────────────────────────────────────────────────────
    default_chunk_size: int = Field(
        default_factory=lambda: int(os.getenv("DEFAULT_CHUNK_SIZE", "500"))
    )
    default_chunk_overlap: int = Field(
        default_factory=lambda: int(os.getenv("DEFAULT_CHUNK_OVERLAP", "100"))
    )
    parent_chunk_size: int = Field(
        default_factory=lambda: int(os.getenv("PARENT_CHUNK_SIZE", "1500"))
    )

    # ── Retrieval & RRF Fusion ───────────────────────────────────────────────
    rrf_k: int = Field(
        default_factory=lambda: int(os.getenv("RRF_K", "60"))
    )
    dense_search_weight: float = Field(
        default_factory=lambda: float(os.getenv("DENSE_SEARCH_WEIGHT", "0.65"))
    )
    lexical_search_weight: float = Field(
        default_factory=lambda: float(os.getenv("LEXICAL_SEARCH_WEIGHT", "0.35"))
    )
    retrieval_top_k: int = Field(
        default_factory=lambda: int(os.getenv("RETRIEVAL_TOP_K", "10"))
    )
    abstain_similarity_threshold: float = Field(
        default_factory=lambda: float(os.getenv("ABSTAIN_SIMILARITY_THRESHOLD", "0.25"))
    )

    # ── Reranking ────────────────────────────────────────────────────────────
    reranker_enabled: bool = Field(
        default_factory=lambda: os.getenv("RERANKER_ENABLED", "false").lower() in ("true", "1", "yes")
    )
    reranker_model_name: str = Field(
        default_factory=lambda: os.getenv("RERANKER_MODEL_NAME", "BAAI/bge-reranker-small")
    )

    # ── LLM / Generation ─────────────────────────────────────────────────────
    llm_provider: str = Field(
        default_factory=lambda: os.getenv("LLM_PROVIDER", "mock")
    )
    llm_model_name: str = Field(
        default_factory=lambda: os.getenv("LLM_MODEL_NAME", "mock-model")
    )
    llm_temperature: float = Field(
        default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", "0.0"))
    )
    llm_max_tokens: int = Field(
        default_factory=lambda: int(os.getenv("LLM_MAX_TOKENS", "512"))
    )
    local_llm_endpoint: str = Field(
        default_factory=lambda: os.getenv("LOCAL_LLM_ENDPOINT", "http://127.0.0.1:11434/v1")
    )
    local_llm_api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("LOCAL_LLM_API_KEY", None)
    )

    # ── Host & Security ──────────────────────────────────────────────────────
    api_host: str = Field(default_factory=lambda: os.getenv("API_HOST", "127.0.0.1"))
    api_port: int = Field(default_factory=lambda: int(os.getenv("API_PORT", "8000")))
    ui_host: str = Field(default_factory=lambda: os.getenv("UI_HOST", "127.0.0.1"))
    ui_port: int = Field(default_factory=lambda: int(os.getenv("UI_PORT", "8501")))
    log_level: str = Field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))

    def get_hardware_manager(self) -> HardwareManager:
        """Create a HardwareManager instance adhering to this configuration."""
        return HardwareManager(
            cpu_only=self.cpu_only,
            gpu_execution_approved=self.gpu_execution_approved,
            preferred_device=self.device
        )

    def ensure_directories(self) -> None:
        """Create local data and cache directories if they do not exist."""
        self.sqlite_db_path.parent.mkdir(parents=True, exist_ok=True)
        self.qdrant_storage_path.mkdir(parents=True, exist_ok=True)
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)


@lru_cache()
def get_settings() -> Settings:
    """Retrieve cached application settings instance."""
    return Settings()


def reset_settings_cache() -> None:
    """Clear cached settings instance (crucial for test isolation)."""
    get_settings.cache_clear()
