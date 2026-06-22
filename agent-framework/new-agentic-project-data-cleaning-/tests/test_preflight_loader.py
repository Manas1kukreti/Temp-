"""Unit tests for grounding/preflight_loader.py.

Validates the Preflight Data Loader: read-only profiling, content hashing,
size limit enforcement, and format validation.

Requirements: 16.1, 16.2, 16.3, 16.5, 16.6
"""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "src")

import pandas as pd
import pytest

from finflow_agent.grounding.preflight_loader import (
    DataFrameProfile,
    DataSnapshotRef,
    FileTooLargeError,
    PreflightConfig,
    PreflightDataLoader,
    UnsupportedFormatError,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_csv(tmp_path: Path) -> Path:
    """Create a simple CSV file for testing."""
    csv_path = tmp_path / "test_data.csv"
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
            "amount": [100.5, 200.0, None, 350.75, 400.0],
            "status": ["active", "active", "inactive", "active", "inactive"],
        }
    )
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def tmp_xlsx(tmp_path: Path) -> Path:
    """Create a simple Excel file for testing."""
    xlsx_path = tmp_path / "test_data.xlsx"
    df = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "value": [10, 20, 30],
        }
    )
    df.to_excel(xlsx_path, index=False, engine="openpyxl")
    return xlsx_path


@pytest.fixture
def loader() -> PreflightDataLoader:
    """Default loader with standard config."""
    return PreflightDataLoader()


@pytest.fixture
def small_limit_loader() -> PreflightDataLoader:
    """Loader with a very small size limit for testing rejection."""
    config = PreflightConfig(max_file_bytes=50)
    return PreflightDataLoader(config=config)


# ---------------------------------------------------------------------------
# PreflightConfig Tests
# ---------------------------------------------------------------------------


class TestPreflightConfig:
    """Tests for PreflightConfig model."""

    def test_default_max_file_bytes(self):
        config = PreflightConfig()
        assert config.max_file_bytes == 100 * 1024 * 1024  # 100 MB

    def test_default_supported_formats(self):
        config = PreflightConfig()
        assert config.supported_formats == frozenset({"csv", "xlsx", "xls"})

    def test_custom_max_file_bytes(self):
        config = PreflightConfig(max_file_bytes=1024)
        assert config.max_file_bytes == 1024

    def test_custom_supported_formats(self):
        config = PreflightConfig(supported_formats=frozenset({"csv"}))
        assert config.supported_formats == frozenset({"csv"})

    def test_rejects_zero_max_file_bytes(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            PreflightConfig(max_file_bytes=0)

    def test_rejects_negative_max_file_bytes(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            PreflightConfig(max_file_bytes=-1)


# ---------------------------------------------------------------------------
# Exception Tests
# ---------------------------------------------------------------------------


class TestFileTooLargeError:
    """Tests for FileTooLargeError exception."""

    def test_message_contains_details(self):
        err = FileTooLargeError("data.csv", 200, 100)
        assert "data.csv" in str(err)
        assert "200" in str(err)
        assert "100" in str(err)

    def test_attributes(self):
        err = FileTooLargeError("data.csv", 200, 100)
        assert err.file_path == "data.csv"
        assert err.byte_size == 200
        assert err.max_bytes == 100


class TestUnsupportedFormatError:
    """Tests for UnsupportedFormatError exception."""

    def test_message_contains_details(self):
        err = UnsupportedFormatError("data.json", "json", frozenset({"csv", "xlsx"}))
        assert "data.json" in str(err)
        assert "json" in str(err)

    def test_attributes(self):
        err = UnsupportedFormatError("data.json", "json", frozenset({"csv"}))
        assert err.file_path == "data.json"
        assert err.extension == "json"
        assert err.supported == frozenset({"csv"})


# ---------------------------------------------------------------------------
# PreflightDataLoader Tests
# ---------------------------------------------------------------------------


class TestPreflightDataLoaderCSV:
    """Tests for loading CSV files."""

    def test_load_produces_profile_and_snapshot(self, loader: PreflightDataLoader, tmp_csv: Path):
        profile, snapshot = loader.load(str(tmp_csv), file_id="file-001")
        assert isinstance(profile, DataFrameProfile)
        assert isinstance(snapshot, DataSnapshotRef)

    def test_profile_has_correct_row_count(self, loader: PreflightDataLoader, tmp_csv: Path):
        profile, _ = loader.load(str(tmp_csv), file_id="file-001")
        assert profile.row_count == 5

    def test_profile_has_correct_column_count(self, loader: PreflightDataLoader, tmp_csv: Path):
        profile, _ = loader.load(str(tmp_csv), file_id="file-001")
        assert profile.column_count == 4

    def test_profile_column_names(self, loader: PreflightDataLoader, tmp_csv: Path):
        profile, _ = loader.load(str(tmp_csv), file_id="file-001")
        col_names = [c.column for c in profile.columns]
        assert "id" in col_names
        assert "name" in col_names
        assert "amount" in col_names
        assert "status" in col_names

    def test_snapshot_file_id(self, loader: PreflightDataLoader, tmp_csv: Path):
        _, snapshot = loader.load(str(tmp_csv), file_id="file-001")
        assert snapshot.file_id == "file-001"

    def test_snapshot_content_hash_is_sha256(self, loader: PreflightDataLoader, tmp_csv: Path):
        _, snapshot = loader.load(str(tmp_csv), file_id="file-001")
        # SHA-256 produces a 64-character hex string
        assert len(snapshot.content_hash) == 64
        assert all(c in "0123456789abcdef" for c in snapshot.content_hash)

    def test_snapshot_content_hash_matches_file(self, loader: PreflightDataLoader, tmp_csv: Path):
        _, snapshot = loader.load(str(tmp_csv), file_id="file-001")
        expected_hash = hashlib.sha256(tmp_csv.read_bytes()).hexdigest()
        assert snapshot.content_hash == expected_hash

    def test_snapshot_byte_size(self, loader: PreflightDataLoader, tmp_csv: Path):
        _, snapshot = loader.load(str(tmp_csv), file_id="file-001")
        assert snapshot.byte_size == tmp_csv.stat().st_size

    def test_snapshot_storage_version(self, loader: PreflightDataLoader, tmp_csv: Path):
        _, snapshot = loader.load(str(tmp_csv), file_id="file-001")
        assert snapshot.storage_version == "1.0"

    def test_snapshot_custom_storage_version(self, loader: PreflightDataLoader, tmp_csv: Path):
        _, snapshot = loader.load(str(tmp_csv), file_id="file-001", storage_version="2.0")
        assert snapshot.storage_version == "2.0"

    def test_snapshot_has_profile_id(self, loader: PreflightDataLoader, tmp_csv: Path):
        _, snapshot = loader.load(str(tmp_csv), file_id="file-001")
        assert snapshot.profile_id  # non-empty UUID

    def test_snapshot_has_structural_fingerprint(self, loader: PreflightDataLoader, tmp_csv: Path):
        _, snapshot = loader.load(str(tmp_csv), file_id="file-001")
        assert len(snapshot.structural_schema_fingerprint) == 64

    def test_snapshot_has_profile_fingerprint(self, loader: PreflightDataLoader, tmp_csv: Path):
        _, snapshot = loader.load(str(tmp_csv), file_id="file-001")
        assert len(snapshot.profile_fingerprint) == 64


class TestPreflightDataLoaderExcel:
    """Tests for loading Excel files."""

    def test_load_xlsx(self, loader: PreflightDataLoader, tmp_xlsx: Path):
        profile, snapshot = loader.load(str(tmp_xlsx), file_id="xlsx-001")
        assert profile.row_count == 3
        assert profile.column_count == 2
        assert snapshot.file_id == "xlsx-001"


class TestPreflightDataLoaderReadOnly:
    """Req 16.2: Verify read-only mode — no mutation of source data."""

    def test_source_file_unchanged_after_load(self, loader: PreflightDataLoader, tmp_csv: Path):
        original_bytes = tmp_csv.read_bytes()
        loader.load(str(tmp_csv), file_id="file-001")
        assert tmp_csv.read_bytes() == original_bytes

    def test_content_hash_stable_across_loads(self, loader: PreflightDataLoader, tmp_csv: Path):
        _, snap1 = loader.load(str(tmp_csv), file_id="file-001")
        _, snap2 = loader.load(str(tmp_csv), file_id="file-002")
        assert snap1.content_hash == snap2.content_hash


class TestPreflightDataLoaderSizeLimits:
    """Req 16.6: Enforce configurable size limits."""

    def test_rejects_oversized_file(self, small_limit_loader: PreflightDataLoader, tmp_csv: Path):
        with pytest.raises(FileTooLargeError) as exc_info:
            small_limit_loader.load(str(tmp_csv), file_id="file-001")
        assert exc_info.value.byte_size == tmp_csv.stat().st_size
        assert exc_info.value.max_bytes == 50

    def test_accepts_file_within_limit(self, loader: PreflightDataLoader, tmp_csv: Path):
        # Default 100MB limit should accept a small CSV
        profile, snapshot = loader.load(str(tmp_csv), file_id="file-001")
        assert profile is not None
        assert snapshot is not None


class TestPreflightDataLoaderFormatValidation:
    """Validate supported/unsupported format detection."""

    def test_rejects_unsupported_format(self, loader: PreflightDataLoader, tmp_path: Path):
        json_file = tmp_path / "data.json"
        json_file.write_text('{"key": "value"}')
        with pytest.raises(UnsupportedFormatError) as exc_info:
            loader.load(str(json_file), file_id="file-001")
        assert exc_info.value.extension == "json"

    def test_rejects_txt_format(self, loader: PreflightDataLoader, tmp_path: Path):
        txt_file = tmp_path / "data.txt"
        txt_file.write_text("hello world")
        with pytest.raises(UnsupportedFormatError):
            loader.load(str(txt_file), file_id="file-001")

    def test_custom_formats(self, tmp_path: Path):
        config = PreflightConfig(supported_formats=frozenset({"csv"}))
        loader = PreflightDataLoader(config=config)
        xlsx_file = tmp_path / "data.xlsx"
        df = pd.DataFrame({"a": [1]})
        df.to_excel(xlsx_file, index=False, engine="openpyxl")
        with pytest.raises(UnsupportedFormatError):
            loader.load(str(xlsx_file), file_id="file-001")


class TestPreflightDataLoaderFileNotFound:
    """Validate file-not-found handling."""

    def test_raises_file_not_found(self, loader: PreflightDataLoader):
        with pytest.raises(FileNotFoundError):
            loader.load("/nonexistent/path/data.csv", file_id="file-001")


class TestPreflightDataLoaderDeterminism:
    """Verify deterministic output for same input."""

    def test_structural_fingerprint_deterministic(self, loader: PreflightDataLoader, tmp_csv: Path):
        _, snap1 = loader.load(str(tmp_csv), file_id="file-001")
        _, snap2 = loader.load(str(tmp_csv), file_id="file-002")
        assert snap1.structural_schema_fingerprint == snap2.structural_schema_fingerprint

    def test_profile_fingerprint_deterministic(self, loader: PreflightDataLoader, tmp_csv: Path):
        _, snap1 = loader.load(str(tmp_csv), file_id="file-001")
        _, snap2 = loader.load(str(tmp_csv), file_id="file-002")
        assert snap1.profile_fingerprint == snap2.profile_fingerprint
