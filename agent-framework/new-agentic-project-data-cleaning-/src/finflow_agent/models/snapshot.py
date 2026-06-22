"""DataSnapshotRef model — immutable reference to a profiled file version.

DataSnapshotRef captures the identity of a specific file version that has been
profiled. It links the file to its structural and profile fingerprints, enabling
cache lookups in the Schema_Service and consistency verification in the Executor.

Requirements: 16.3 (content hash for consistency checks)
"""

from pydantic import BaseModel, ConfigDict, Field


class DataSnapshotRef(BaseModel):
    """Immutable reference to a profiled file version.

    Produced by the Preflight_Data_Loader after loading and profiling the
    source file. Used by:
    - Schema_Service for cache key construction
    - Executor for content_hash consistency verification
    - Observability for structured tracing (data_snapshot_ref field)
    """

    model_config = ConfigDict(strict=True)

    file_id: str
    content_hash: str  # SHA-256 hex digest of file content
    byte_size: int = Field(ge=0)
    storage_version: str
    profile_id: str
    structural_schema_fingerprint: str  # Links to StructuralSchemaFingerprint.fingerprint
    profile_fingerprint: str  # Links to ProfileFingerprint.fingerprint
