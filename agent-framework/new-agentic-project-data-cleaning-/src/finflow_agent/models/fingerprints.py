"""Fingerprint models for deterministic schema and profile hashing.

StructuralSchemaFingerprint: deterministic SHA-256 hash of dataset structural schema
(normalized column names, dtypes, nullable info, profiler version).

ProfileFingerprint: deterministic SHA-256 hash of semantic statistics using
pre-hashed representative values (no raw values that may contain sensitive information).

Requirements: 5.1, 5.2, 5.6
"""

import hashlib
import json

from pydantic import BaseModel, ConfigDict


class StructuralSchemaFingerprint(BaseModel):
    """Deterministic hash of dataset structural schema.

    Used as cache key for the Schema_Service structural role cache.
    The fingerprint is computed from normalized, sorted column names,
    their dtypes, nullable flags, and the profiler version.
    """

    model_config = ConfigDict(strict=True)

    fingerprint: str  # SHA-256 hex digest
    column_names: list[str]  # Normalized (lowercased), sorted
    column_dtypes: list[str]
    nullable: list[bool]
    profiler_version: str

    @classmethod
    def compute(
        cls,
        columns: list[str],
        dtypes: list[str],
        nullable: list[bool],
        profiler_version: str,
    ) -> "StructuralSchemaFingerprint":
        """Compute deterministic fingerprint from structural schema.

        Normalization: column names are lowercased, then all columns are
        sorted alphabetically by name. The fingerprint is a SHA-256 hex
        digest of the canonical JSON representation.
        """
        # Normalize: lowercase column names, sort by name
        normalized = sorted(
            zip(columns, dtypes, nullable), key=lambda x: x[0].lower()
        )
        payload = json.dumps(
            {
                "columns": [c.lower() for c, _, _ in normalized],
                "dtypes": [d for _, d, _ in normalized],
                "nullable": [n for _, _, n in normalized],
                "profiler_version": profiler_version,
            },
            sort_keys=True,
        )
        fp = hashlib.sha256(payload.encode()).hexdigest()
        return cls(
            fingerprint=fp,
            column_names=[c.lower() for c, _, _ in normalized],
            column_dtypes=[d for _, d, _ in normalized],
            nullable=[n for _, _, n in normalized],
            profiler_version=profiler_version,
        )


class ProfileFingerprint(BaseModel):
    """Deterministic hash of semantic statistics (no raw values).

    Used as part of the cache key for the Schema_Service value-evidence cache.
    Uses pre-hashed representative values to avoid exposing sensitive data
    (Requirement 5.6: Schema_Service SHALL NOT hash raw values that may
    contain sensitive information).
    """

    model_config = ConfigDict(strict=True)

    fingerprint: str  # SHA-256 hex digest
    cardinality_buckets: list[int]
    representative_value_hashes: list[str]  # Pre-hashed, not raw values
    profiling_config_version: str

    @classmethod
    def compute(
        cls,
        cardinality_buckets: list[int],
        representative_value_hashes: list[str],
        profiling_config_version: str,
    ) -> "ProfileFingerprint":
        """Compute deterministic fingerprint from profiling statistics.

        The representative_value_hashes must already be hashed by the caller;
        this ensures no raw sensitive values are included in the fingerprint
        computation.
        """
        payload = json.dumps(
            {
                "cardinality_buckets": cardinality_buckets,
                "profiling_config_version": profiling_config_version,
                "representative_value_hashes": representative_value_hashes,
            },
            sort_keys=True,
        )
        fp = hashlib.sha256(payload.encode()).hexdigest()
        return cls(
            fingerprint=fp,
            cardinality_buckets=cardinality_buckets,
            representative_value_hashes=representative_value_hashes,
            profiling_config_version=profiling_config_version,
        )
