from __future__ import annotations

import os

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.database.models import Base

DEFAULT_DB_URL = "sqlite:///data/processed/neural_vol.db"


def get_database_url(db_url: str | None = None) -> str:
    """
    Resolve database URL with precedence:
    1) explicit function argument
    2) DATABASE_URL environment variable
    3) local SQLite fallback
    """
    resolved = db_url or os.getenv("DATABASE_URL", DEFAULT_DB_URL)
    if resolved.startswith("postgres://"):
        return resolved.replace("postgres://", "postgresql://", 1)
    return resolved


def get_engine(db_url: str | None = None):
    resolved = get_database_url(db_url)
    kwargs = {"future": True, "echo": False, "pool_pre_ping": True}
    if resolved.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
    return create_engine(resolved, **kwargs)


def get_session_factory(db_url: str | None = None) -> sessionmaker[Session]:
    engine = get_engine(db_url)
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)


def init_db(db_url: str | None = None) -> None:
    engine = get_engine(db_url)
    Base.metadata.create_all(bind=engine)
