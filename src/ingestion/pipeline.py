from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from src.ingestion.base import OptionsDataProvider
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class IngestionSchedule:
    start: datetime
    end: datetime
    frequency: str = "1D"


class OptionsIngestionPipeline:
    def __init__(self, provider: OptionsDataProvider, raw_dir: Path):
        self.provider = provider
        self.raw_dir = raw_dir
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def pull_snapshot(self, ticker: str, as_of: datetime | None = None) -> pd.DataFrame:
        snapshot = self.provider.fetch_chain(ticker, as_of=as_of)
        day = (as_of or datetime.now(timezone.utc)).strftime("%Y%m%d")
        out_file = self.raw_dir / f"{ticker}_{day}_snapshot.parquet"
        snapshot.to_parquet(out_file, index=False)
        LOGGER.info("Snapshot saved at %s", out_file)
        return snapshot

    def backfill(self, ticker: str, schedule: IngestionSchedule) -> pd.DataFrame:
        """
        Yahoo is not a historical option-chain archive. For free mode, we schedule
        repeated snapshots and build history incrementally.
        """
        timeline = pd.date_range(schedule.start, schedule.end, freq=schedule.frequency)
        records: list[pd.DataFrame] = []
        for ts in timeline:
            try:
                frame = self.pull_snapshot(ticker=ticker, as_of=ts.to_pydatetime())
                records.append(frame)
            except Exception as exc:
                LOGGER.warning("Backfill skipped at %s for %s: %s", ts, ticker, exc)
        if not records:
            raise ValueError(f"No snapshots collected for {ticker}")
        all_data = pd.concat(records, ignore_index=True)
        outfile = self.raw_dir / f"{ticker}_history.parquet"
        all_data.to_parquet(outfile, index=False)
        LOGGER.info("Backfill saved at %s rows=%d", outfile, len(all_data))
        return all_data

    @staticmethod
    def intraday_schedule(hours: int = 6, interval_minutes: int = 60) -> IngestionSchedule:
        now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        start = now - timedelta(hours=hours)
        return IngestionSchedule(start=start, end=now, frequency=f"{interval_minutes}min")
