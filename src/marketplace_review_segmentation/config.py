from __future__ import annotations

from pathlib import Path
from typing import Iterable

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    dataset_root: Path = Field(default=Path("data/source/dataset/full"))
    dataset_roots_raw: str | None = Field(default=None)
    duckdb_path: Path = Field(default=Path(r"F:\diploma\warehouse\review_segmentation.duckdb"))
    artifacts_dir: Path = Field(default=Path(r"F:\diploma\warehouse\artifacts"))
    duckdb_temp_dir: Path = Field(default=Path(r"F:\diploma\warehouse\tmp"))
    default_text_model: str = Field(
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    random_state: int = Field(default=42)
    dataset_epoch_start: str = Field(default="2020-01-01 00:00:00")

    model_config = SettingsConfigDict(
        env_prefix="MRS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def ensure_directories(self) -> None:
        self.duckdb_path.parent.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.duckdb_temp_dir.mkdir(parents=True, exist_ok=True)

    @property
    def dataset_roots(self) -> list[Path]:
        if self.dataset_roots_raw:
            roots = [Path(part.strip()) for part in self.dataset_roots_raw.split(";") if part.strip()]
            if roots:
                return roots
        return [self.dataset_root]


settings = Settings()
