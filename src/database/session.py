from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.database.models import Base


def get_engine(db_url: str = "sqlite:///data/processed/neural_vol.db"):
    return create_engine(db_url, future=True, echo=False)


def get_session_factory(db_url: str = "sqlite:///data/processed/neural_vol.db") -> sessionmaker[Session]:
    engine = get_engine(db_url)
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)


def init_db(db_url: str = "sqlite:///data/processed/neural_vol.db") -> None:
    engine = get_engine(db_url)
    Base.metadata.create_all(bind=engine)
