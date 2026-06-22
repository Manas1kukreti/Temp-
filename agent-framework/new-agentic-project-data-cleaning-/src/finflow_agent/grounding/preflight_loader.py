"""Preflight Data Loader — read-only file profiling before grounding.

Loads source files, computes content hashes, enforces size limits, and produces
a DataFrameProfile + DataSnapshotRef before Schema_Service and grounding stages.

The loader operates in strict read-only mode: it NEVER modifies, cleans, or
transforms the source data. It reads bytes for hashing, loads a DataFrame
for profiling, and returns immutable references.

Requirements: 16.1, 16.2, 16.3, 16.5, 16.6
"""

from __future__ import annotations

import hashlib
import os
import uuid
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from finflow_agent.models.fingerprints import (
    ProfileFingerprint,
    StructuralSchemaFingerprint,
)
from finflow_agent.models.snapshot import DataSnapshotRef
from finflow_agent.tools.dataframe_profile import DataFrameProfile, profile_dataframe


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SupportedFormat = Literal["csv", "xlsx", "xls"]

_DEFAULT_SUPPORTED_FORMATS: frozenset[str] = frozenset({"csv", "xlsx", "xls"})
_DEFAULT_MAX_FILE_BYTES: int = 100 * 1024 * 1024  # 100 MB
_PROFILER_VERSION: str = "1.0"
_TOP_N_VALUES_FOR_HASH: int = 10


class PreflightConfig(BaseModel):
    """Configurable limits for the Preflight Data Loader.

    Attributes:
        max_file_bytes: Maximum allowed file size in bytes. Files exceeding
            this limit are rejected before reading. Default: 100 MB.
        supported_formats: Set of file extensions (without dot) that may be
            loaded. Default: {"csv", "xlsx", "xls"}.
    """

    model_config = ConfigDict(strict=True)

    max_file_bytes: int = Field(default=_DEFAULT_MAX_FILE_BYTES, ge=1)
    supported_formats: frozenset[str] = Field(
        default=_DEFAULT_SUPPORTED_FORMATS,
    )


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class FileTooLargeError(Exception):
    """Raised when a file exceeds the configured size limit."""

    def __init__(self, file_path: str, byte_size: int, max_bytes: int) -> None:
        self.file_path = file_path
        self.byte_size = byte_size
        self.max_bytes = max_bytes
        super().__init__(
            f"File '{file_path}' is {byte_size} bytes, "
            f"exceeding the maximum allowed size of {max_bytes} bytes."
        )


class UnsupportedFormatError(Exception):
    """Raised when a file's format is not in the supported set."""

    def __init__(self, file_path: str, extension: str, supported: frozenset[str]) -> None:
        self.file_path = file_path
        self.extension = extension
        self.supported = supported
        super().__init__(
            f"File '{file_path}' has unsupported format '.{extension}'. "
            f"Supported formats: {sorted(supported)}"
        )


# ---------------------------------------------------------------------------
# Preflight Data Loader
# ---------------------------------------------------------------------------


class PreflightDataLoader:
    """Read-only loader that profiles source files for the grounding pipeline.

    The loader:
    1. Validates file format and size BEFORE reading data.
    2. Computes SHA-256 content_hash of the raw file bytes.
    3. Reads the file into a pandas DataFrame without any mutation.
    4. Produces a DataFrameProfile from the DataFrame.
    5. Produces a DataSnapshotRef linking file identity, hash, and fingerprints.

    The source file and resulting DataFrame are NEVER modified.

    Requirements: 16.1, 16.2, 16.3, 16.5, 16.6
    """

    def __init__(self, config: PreflightConfig | None = None) -> None:
        self._config = config or PreflightConfig()

    @property
    def config(self) -> PreflightConfig:
        return self._config

    def load(
        self,
        file_path: str,
        file_id: str,
        storage_version: str = "1.0",
    ) -> tuple[DataFrameProfile, DataSnapshotRef]:
        """Load and profile source file in read-only mode.

        Args:
            file_path: Absolute or relative path to the source file.
            file_id: Unique identifier for this file in the pipeline.
            storage_version: Version tag for the storage layer.

        Returns:
            A tuple of (DataFrameProfile, DataSnapshotRef).

        Raises:
            FileNotFoundError: If the file does not exist.
            UnsupportedFormatError: If the file extension is not supported.
            FileTooLargeError: If the file exceeds the configured size limit.
        """
        path = Path(file_path)

        # --- Validate file exists ---
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {file_path}")

        # --- Validate format BEFORE reading ---
        extension = self._get_extension(path)
        if extension not in self._config.supported_formats:
            raise UnsupportedFormatError(
                file_path=str(path),
                extension=extension,
                supported=self._config.supported_formats,
            )

        # --- Validate size BEFORE reading (Req 16.6) ---
        byte_size = path.stat().st_size
        if byte_size > self._config.max_file_bytes:
            raise FileTooLargeError(
                file_path=str(path),
                byte_size=byte_size,
                max_bytes=self._config.max_file_bytes,
            )

        # --- Compute content hash (Req 16.3) ---
        content_hash = self._compute_content_hash(path)

        # --- Read file into DataFrame (read-only, Req 16.2) ---
        df = self._read_file(path, extension)

        # --- Produce DataFrameProfile (Req 16.1) ---
        data_profile = profile_dataframe(df, sample_rows=5, include_samples=True)

        # --- Compute fingerprints ---
        structural_fp = self._compute_structural_fingerprint(df)
        profile_fp = self._compute_profile_fingerprint(df, data_profile)

        # --- Produce DataSnapshotRef (Req 16.1, 16.3) ---
        profile_id = str(uuid.uuid4())
        snapshot_ref = DataSnapshotRef(
            file_id=file_id,
            content_hash=content_hash,
            byte_size=byte_size,
            storage_version=storage_version,
            profile_id=profile_id,
            structural_schema_fingerprint=structural_fp.fingerprint,
            profile_fingerprint=profile_fp.fingerprint,
        )

        return data_profile, snapshot_ref

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_extension(path: Path) -> str:
        """Extract the lowercase file extension without the dot."""
        ext = path.suffix.lower().lstrip(".")
        return ext

    @staticmethod
    def _compute_content_hash(path: Path) -> str:
        """Compute SHA-256 hex digest of file content (Req 16.3)."""
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    @staticmethod
    def _read_file(path: Path, extension: str) -> pd.DataFrame:
        """Read file into a DataFrame without any transformation (Req 16.2).

        The DataFrame is loaded as-is. No cleaning, type coercion beyond
        pandas defaults, or mutation is performed.
        """
        if extension == "csv":
            return pd.read_csv(path)
        elif extension in ("xlsx", "xls"):
            return pd.read_excel(path, engine="openpyxl")
        else:
            # Should not reach here due to format validation above
            raise UnsupportedFormatError(
                file_path=str(path),
                extension=extension,
                supported=frozenset(),
            )

    @staticmethod
    def _compute_structural_fingerprint(df: pd.DataFrame) -> StructuralSchemaFingerprint:
        """Compute structural schema fingerprint from DataFrame columns."""
        columns = [str(c) for c in df.columns]
        dtypes = [str(df[c].dtype) for c in df.columns]
        nullable = [bool(df[c].isnull().any()) for c in df.columns]
        return StructuralSchemaFingerprint.compute(
            columns=columns,
            dtypes=dtypes,
            nullable=nullable,
            profiler_version=_PROFILER_VERSION,
        )

    @staticmethod
    def _compute_profile_fingerprint(
        df: pd.DataFrame, profile: DataFrameProfile
    ) -> ProfileFingerprint:
        """Compute profile fingerprint from semantic statistics.

        Uses cardinality (nunique) per column as buckets and hashes of
        top-N frequent values per column as representative value hashes.
        Raw values are never included — only their SHA-256 hashes (Req 5.6).
        """
        cardinality_buckets: list[int] = []
        representative_value_hashes: list[str] = []

        for col_profile in profile.columns:
            cardinality_buckets.append(col_profile.distinct_count)

            # Hash the top-N representative values (pre-hashed, no raw data)
            values_to_hash = col_profile.frequent_values[:_TOP_N_VALUES_FOR_HASH]
            for val in values_to_hash:
                val_hash = hashlib.sha256(repr(val).encode()).hexdigest()
                representative_value_hashes.append(val_hash)

        return ProfileFingerprint.compute(
            cardinality_buckets=cardinality_buckets,
            representative_value_hashes=representative_value_hashes,
            profiling_config_version=_PROFILER_VERSION,
        )


__all__ = [
    "DataFrameProfile",
    "DataSnapshotRef",
    "FileTooLargeError",
    "PreflightConfig",
    "PreflightDataLoader",
    "UnsupportedFormatError",
]
