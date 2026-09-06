"""Central configuration.

All tunables live here and are overridable through environment variables or a
`.env` file (see `.env.example`). Paths are resolved relative to the repository
root so the same settings object works from a notebook, a pytest run, the
Streamlit app, or inside the Docker container.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repo root = two levels up from src/utils/config.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),
    )

    # ---------------- Anthropic ----------------
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    sql_model: str = Field(default="claude-sonnet-5", alias="SQL_MODEL")
    summary_model: str = Field(default="claude-haiku-4-5", alias="SUMMARY_MODEL")
    max_tokens: int = Field(default=2048, alias="MAX_TOKENS")
    # No temperature setting: the Messages API in anthropic 1.x does not accept
    # one. Determinism comes from the prompt, not from a sampling parameter.
    enable_prompt_caching: bool = Field(default=True, alias="ENABLE_PROMPT_CACHING")

    # ---------------- Paths ----------------
    data_dir: Path = Field(default=Path("data"), alias="DATA_DIR")
    raw_dir: Path = Field(default=Path("data/raw"), alias="RAW_DIR")
    processed_dir: Path = Field(default=Path("data/processed"), alias="PROCESSED_DIR")
    model_dir: Path = Field(default=Path("models"), alias="MODEL_DIR")
    report_dir: Path = Field(default=Path("reports"), alias="REPORT_DIR")
    sqlite_path: Path = Field(default=Path("data/credit_risk.db"), alias="SQLITE_PATH")

    # ---------------- Talk-to-Data guardrails ----------------
    sql_row_limit: int = Field(default=200, alias="SQL_ROW_LIMIT")
    sql_max_retries: int = Field(default=2, alias="SQL_MAX_RETRIES")
    sql_timeout_seconds: int = Field(default=30, alias="SQL_TIMEOUT_SECONDS")

    # ---------------- ML ----------------
    random_seed: int = Field(default=42, alias="RANDOM_SEED")
    holdout_fraction: float = Field(default=0.2, alias="HOLDOUT_FRACTION")
    valid_fraction: float = Field(default=0.1, alias="VALID_FRACTION")
    calib_fraction: float = Field(default=0.1, alias="CALIB_FRACTION")
    # Cost matrix, expressed as a fraction of the applicant's AMT_CREDIT.
    # LGD  = loss given default   -> cost of a false negative (bad loan approved)
    # MARGIN = lifetime margin    -> cost of a false positive (good loan rejected)
    lgd_rate: float = Field(default=0.45, alias="LGD_RATE")
    margin_rate: float = Field(default=0.08, alias="MARGIN_RATE")

    # ---------------- Logging ----------------
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    @field_validator(
        "data_dir", "raw_dir", "processed_dir", "model_dir", "report_dir", "sqlite_path",
        mode="after",
    )
    @classmethod
    def _absolutise(cls, value: Path) -> Path:
        """Relative paths are interpreted against the repo root, not the CWD."""
        return value if value.is_absolute() else (PROJECT_ROOT / value).resolve()

    # ---------------- Derived paths ----------------
    @property
    def figures_dir(self) -> Path:
        return self.report_dir / "figures"

    @property
    def model_path(self) -> Path:
        return self.model_dir / "lgbm_credit_risk.joblib"

    @property
    def calibrator_path(self) -> Path:
        return self.model_dir / "calibrator.joblib"

    @property
    def preprocessor_path(self) -> Path:
        return self.model_dir / "preprocessor.joblib"

    @property
    def metrics_path(self) -> Path:
        return self.report_dir / "model_metrics.json"

    @property
    def rules_path(self) -> Path:
        return self.report_dir / "business_rules.json"

    @property
    def eda_summary_path(self) -> Path:
        return self.report_dir / "eda_summary.json"

    @property
    def eval_results_path(self) -> Path:
        return self.report_dir / "nl_sql_eval.json"

    @property
    def fairness_path(self) -> Path:
        return self.report_dir / "fairness_report.json"

    @property
    def holdout_path(self) -> Path:
        return self.processed_dir / "holdout.parquet"

    def ensure_dirs(self) -> None:
        for path in (
            self.data_dir, self.raw_dir, self.processed_dir,
            self.model_dir, self.report_dir, self.figures_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached singleton so every module shares one configuration instance."""
    return Settings()


settings = get_settings()
