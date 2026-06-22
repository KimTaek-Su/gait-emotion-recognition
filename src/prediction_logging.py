import os
import time
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import JSON, DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


def normalize_database_url(database_url: str) -> str:
    value = database_url.strip()

    if value.startswith("postgres://"):
        return "postgresql+psycopg://" + value[len("postgres://"):]

    if value.startswith("postgresql://") and not value.startswith("postgresql+"):
        return "postgresql+psycopg://" + value[len("postgresql://"):]

    return value


class Base(DeclarativeBase):
    pass


class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    request_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    route: Mapped[str] = mapped_column(String(64), nullable=False)
    client_host: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    input_type: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    joint_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    frame_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    feature_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    status_code: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    predicted_emotion: Mapped[Optional[str]] = mapped_column(String(32), nullable=True, index=True)
    confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    confidence_level: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    latency_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    request_preview: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)
    probabilities: Mapped[Optional[dict[str, float]]] = mapped_column(JSON, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


class PredictionLogStore:
    def __init__(self, database_url: Optional[str], retry_interval_seconds: float = 60.0):
        self.database_url = normalize_database_url(database_url or "")
        self.retry_interval_seconds = retry_interval_seconds
        self._engine: Optional[Engine] = None
        self._session_factory = None
        self._last_init_attempt = 0.0
        self._last_error: Optional[str] = None

    @property
    def enabled(self) -> bool:
        return bool(self.database_url)

    @property
    def ready(self) -> bool:
        return self._session_factory is not None

    @property
    def backend(self) -> str:
        if not self.enabled:
            return "disabled"
        if self.database_url.startswith("postgresql"):
            return "postgresql"
        return self.database_url.split(":", 1)[0]

    def health_payload(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "ready": self.ready,
            "backend": self.backend,
        }

    def _build_engine(self) -> Engine:
        connect_args = {}
        if self.database_url.startswith("sqlite"):
            connect_args["check_same_thread"] = False
        return create_engine(self.database_url, pool_pre_ping=True, connect_args=connect_args)

    def _ensure_ready(self) -> bool:
        if not self.enabled:
            return False

        if self._session_factory is not None:
            return True

        now = time.monotonic()
        if self._last_init_attempt and now - self._last_init_attempt < self.retry_interval_seconds:
            return False

        self._last_init_attempt = now

        try:
            self._engine = self._build_engine()
            Base.metadata.create_all(self._engine)
            self._session_factory = sessionmaker(bind=self._engine, expire_on_commit=False)
            self._last_error = None
            print("[LOGGING] prediction log store ready")
            return True
        except Exception as exc:
            self._last_error = repr(exc)
            print(f"[WARN] prediction log store init failed: {self._last_error}")
            return False

    def save_prediction(
        self,
        *,
        request_id: str,
        route: str,
        client_host: Optional[str],
        input_type: str,
        joint_count: Optional[int],
        frame_count: Optional[int],
        feature_count: Optional[int],
        status_code: int,
        predicted_emotion: Optional[str] = None,
        confidence: Optional[float] = None,
        confidence_level: Optional[str] = None,
        latency_ms: Optional[float] = None,
        request_preview: Optional[dict[str, Any]] = None,
        probabilities: Optional[dict[str, float]] = None,
        error_message: Optional[str] = None,
    ) -> bool:
        if not self._ensure_ready():
            return False

        session = self._session_factory()

        try:
            session.add(
                PredictionLog(
                    created_at=datetime.now(timezone.utc),
                    request_id=request_id,
                    route=route,
                    client_host=client_host,
                    input_type=input_type,
                    joint_count=joint_count,
                    frame_count=frame_count,
                    feature_count=feature_count,
                    status_code=status_code,
                    predicted_emotion=predicted_emotion,
                    confidence=confidence,
                    confidence_level=confidence_level,
                    latency_ms=latency_ms,
                    request_preview=request_preview,
                    probabilities=probabilities,
                    error_message=error_message,
                )
            )
            session.commit()
            return True
        except Exception as exc:
            session.rollback()
            self._last_error = repr(exc)
            print(f"[WARN] prediction log write failed: {self._last_error}")
            return False
        finally:
            session.close()


prediction_log_store = PredictionLogStore(os.getenv("DATABASE_URL"))