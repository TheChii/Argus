"""FinBERT sentiment scorer with persistent caching."""
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from src.config.settings import get_settings
from src.utils.logger import setup_logger

logger = setup_logger("finbert_scorer")


class FinBERTScorer:
    """
    Scores news headlines with ProsusAI/finbert.

    Guarantees:
    - Incremental execution: only unscored headlines are sent to the model.
    - Cached persistently in `data/news/scores.parquet`.
    - GPU acceleration when CUDA is available.
    - Output net sentiment score = P(positive) - P(negative) in [-1.0, 1.0].
    """

    SCORES_COLUMNS = [
        "article_hash",
        "positive_prob",
        "negative_prob",
        "neutral_prob",
        "sentiment_score",
        "scored_at",
    ]

    def __init__(self, scores_path: Path | None = None, model_name: str = "ProsusAI/finbert"):
        self.settings = get_settings()
        self.scores_path = scores_path or (self.settings.news_archive_dir.parent / "scores.parquet")
        self.scores_path.parent.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tokenizer = None
        self._model = None
        self._label_map = None

    def _init_model(self) -> None:
        """Lazy initialization of tokenizer and model."""
        if self._model is None:
            logger.info(f"Loading FinBERT model '{self.model_name}' on device: {self._device}...")
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name).to(self._device)
            self._model.eval()

            # Inspect label mappings from model config (typically 0: positive, 1: negative, 2: neutral)
            id2label = self._model.config.id2label
            self._label_map = {v.lower(): k for k, v in id2label.items()}
            logger.info(f"FinBERT loaded successfully with label mapping: {id2label}")

    def load_scores(self) -> pd.DataFrame:
        """Loads cached sentiment scores."""
        if not self.scores_path.exists():
            return pd.DataFrame(columns=self.SCORES_COLUMNS)
        try:
            return pd.read_parquet(self.scores_path)
        except Exception as e:
            logger.error(f"Failed to read sentiment scores cache {self.scores_path}: {e}")
            return pd.DataFrame(columns=self.SCORES_COLUMNS)

    def _save_scores(self, df: pd.DataFrame) -> None:
        """Atomically saves sentiment scores DataFrame."""
        tmp_path = self.scores_path.with_suffix(".tmp")
        df.to_parquet(tmp_path, index=False, engine="pyarrow", compression="snappy")
        tmp_path.replace(self.scores_path)

    def score_articles(
        self,
        articles: list[dict[str, Any]],
        batch_size: int = 32
    ) -> pd.DataFrame:
        """
        Scores a list of article dictionaries containing 'article_hash' and 'headline'.
        Only unscored article hashes are evaluated.

        Returns:
            DataFrame containing all scores for the requested articles.
        """
        if not articles:
            return self.load_scores()

        existing_df = self.load_scores()
        cached_hashes: set[str] = set(existing_df["article_hash"]) if not existing_df.empty else set()

        # Identify unscored articles
        to_score = [a for a in articles if a.get("article_hash") and a["article_hash"] not in cached_hashes]

        # Deduplicate within the to_score batch
        seen_batch: set[str] = set()
        unique_to_score = []
        for a in to_score:
            h = a["article_hash"]
            if h not in seen_batch and a.get("headline"):
                seen_batch.add(h)
                unique_to_score.append(a)

        if not unique_to_score:
            logger.info(f"All {len(articles)} headlines already scored in cache. 0 inference calls needed.")
            return existing_df

        self._init_model()
        logger.info(f"Scoring {len(unique_to_score)} new headlines with FinBERT (batch size: {batch_size})...")

        pos_idx = self._label_map.get("positive", 0)
        neg_idx = self._label_map.get("negative", 1)
        neu_idx = self._label_map.get("neutral", 2)

        scored_records = []
        now_iso = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

        for i in range(0, len(unique_to_score), batch_size):
            batch = unique_to_score[i:i + batch_size]
            headlines = [b["headline"] for b in batch]

            inputs = self._tokenizer(
                headlines,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            ).to(self._device)

            with torch.no_grad():
                outputs = self._model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()

            for item, prob in zip(batch, probs):
                p_pos = float(prob[pos_idx])
                p_neg = float(prob[neg_idx])
                p_neu = float(prob[neu_idx])
                net_score = p_pos - p_neg

                scored_records.append({
                    "article_hash": item["article_hash"],
                    "positive_prob": p_pos,
                    "negative_prob": p_neg,
                    "neutral_prob": p_neu,
                    "sentiment_score": net_score,
                    "scored_at": now_iso,
                })

            done_count = min(i + len(batch), len(unique_to_score))
            if done_count % 2560 < len(batch) or done_count == len(unique_to_score):
                logger.info(f"FinBERT progress: {done_count}/{len(unique_to_score)} headlines scored...")

        new_scores_df = pd.DataFrame(scored_records)
        if not existing_df.empty:
            combined_df = pd.concat([existing_df, new_scores_df], ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset=["article_hash"], keep="last").reset_index(drop=True)
        else:
            combined_df = new_scores_df

        self._save_scores(combined_df)
        logger.info(f"Successfully scored and cached {len(new_scores_df)} headlines. (Total cache: {len(combined_df)})")
        return combined_df
