"""Global configuration and settings management."""
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Automatically load .env if present
load_dotenv()


@dataclass
class Settings:
    """Project-wide settings and directory paths."""
    # Base directories
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent)
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("DATA_DIR", "data")))

    # API credentials
    fred_api_key: str | None = field(default_factory=lambda: os.getenv("FRED_API_KEY") or None)
    finnhub_api_key: str | None = field(default_factory=lambda: os.getenv("FINNHUB_API_KEY") or None)
    telegram_bot_token: str | None = field(default_factory=lambda: os.getenv("TELEGRAM_BOT_TOKEN") or None)
    telegram_chat_id: str | None = field(default_factory=lambda: os.getenv("TELEGRAM_CHAT_ID") or None)

    # Broker API credentials & execution mode
    alpaca_api_key: str | None = field(default_factory=lambda: os.getenv("ALPACA_API_KEY") or None)
    alpaca_secret_key: str | None = field(default_factory=lambda: os.getenv("ALPACA_SECRET_KEY") or None)
    alpaca_paper: bool = field(default_factory=lambda: os.getenv("ALPACA_PAPER", "true").lower() in ("true", "1", "yes"))
    alpaca_base_url: str = field(default_factory=lambda: os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets/v2"))

    trading212_api_key: str | None = field(default_factory=lambda: os.getenv("TRADING212_API_KEY") or None)
    trading212_api_secret: str | None = field(default_factory=lambda: os.getenv("TRADING212_API_SECRET") or None)
    trading212_env: str = field(default_factory=lambda: os.getenv("TRADING212_ENV", "demo").lower())
    trading212_base_url: str = field(default_factory=lambda: os.getenv("TRADING212_BASE_URL", "https://demo.trading212.com/api/v0"))

    default_broker: str = field(default_factory=lambda: os.getenv("DEFAULT_BROKER", "dry_run").lower())

    # Logging
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))

    # Timezone & Market Rules
    market_timezone: str = "America/New_York"
    market_close_hour: int = 16  # 4:00 PM ET cutoff
    market_close_minute: int = 0

    # FRED Macro Series Definitions
    fred_series: dict[str, str] = field(default_factory=lambda: {
        "T10Y2Y": "10-Year Treasury Minus 2-Year Treasury Yield Spread",
        "DFF": "Federal Funds Effective Rate",
        "VIXCLS": "CBOE Volatility Index",
        "BAMLH0A0HYM2": "ICE BofA US High Yield Index OAS",
    })

    # Rate Limiting & Delays (seconds)
    yfinance_delay_seconds: float = 0.2
    finnhub_delay_seconds: float = 1.1  # <= 60 calls/minute
    gdelt_delay_seconds: float = 1.5

    # Derived paths
    @property
    def prices_dir(self) -> Path:
        return self._resolve_path(self.data_dir / "prices")

    @property
    def macro_dir(self) -> Path:
        return self._resolve_path(self.data_dir / "macro")

    @property
    def news_archive_dir(self) -> Path:
        return self._resolve_path(self.data_dir / "news" / "archive")

    @property
    def fundamentals_dir(self) -> Path:
        return self._resolve_path(self.data_dir / "fundamentals")

    @property
    def logs_dir(self) -> Path:
        return self._resolve_path(self.data_dir / "logs")

    def _resolve_path(self, path: Path) -> Path:
        if path.is_absolute():
            return path
        return self.project_root / path

    def ensure_directories(self) -> None:
        """Create all required data directories if they do not exist."""
        for d in [self.prices_dir, self.macro_dir, self.news_archive_dir, self.fundamentals_dir, self.logs_dir]:
            d.mkdir(parents=True, exist_ok=True)



_SETTINGS_INSTANCE: Settings | None = None


def get_settings() -> Settings:
    """Singleton getter for settings."""
    global _SETTINGS_INSTANCE
    if _SETTINGS_INSTANCE is None:
        _SETTINGS_INSTANCE = Settings()
    return _SETTINGS_INSTANCE
