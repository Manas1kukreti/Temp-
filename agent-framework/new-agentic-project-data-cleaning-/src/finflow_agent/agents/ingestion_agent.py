from typing import Literal
import os
import pandas as pd
from pydantic import BaseModel, ValidationError
from finflow_agent.registry import registry, AgentSpec
from finflow_agent.state import AgentResult
from finflow_agent.tools.path_safety import get_safe_input_path
from finflow_agent.operations.errors import UnsafeInputPathError

class IngestionAgentParams(BaseModel):
    resolved_file_path: str
    file_type: Literal["xlsx", "xls", "csv", "pdf"]

@registry.register
class IngestionAgent:
    spec = AgentSpec(
        name="ingestion_agent",
        description="Parses XLSX, XLS, and CSV files into a structured dataframe.",
        stage="ingest",
        accepts=["file"],
        produces=["dataframe"],
        params_schema={
            "resolved_file_path": {"type": "string"},
            "file_type": {"type": "string"}
        }
    )
    # Pydantic params model picked up by the registry so the validator and
    # engine can re-validate `step.params` before this agent is invoked.
    params_model = IngestionAgentParams

    def execute(self, params: dict, input_data: dict) -> AgentResult:
        # Strict parameter validation
        try:
            validated = IngestionAgentParams.model_validate(params)
        except ValidationError as e:
            return AgentResult(
                status="failed",
                error_message=(
                    "Invalid parameter schema for IngestionAgent "
                    f"(resolved_file_path={params.get('resolved_file_path')!r}, "
                    f"file_type={params.get('file_type')!r}): {e}"
                )
            )

        resolved_file_path = validated.resolved_file_path
        file_type = validated.file_type.lower()

        if file_type in ["png", "jpg", "jpeg", "gif"]:
            return AgentResult(
                status="failed",
                error_message=(
                    f"Image files are not supported by IngestionAgent "
                    f"(file_type={file_type!r}, path={resolved_file_path!r})."
                ),
            )

        if file_type not in ["xlsx", "xls", "csv", "pdf"]:
            return AgentResult(
                status="failed",
                error_message=(
                    "Unsupported file type for IngestionAgent: "
                    f"file_type={file_type!r}, path={resolved_file_path!r}. "
                    "Allowed values: 'xlsx', 'xls', 'csv', 'pdf'."
                ),
            )

        # Path safety: when UPLOAD_DIR is configured, enforce a sandbox boundary
        # so a malformed or malicious upstream cannot make us read files outside
        # the configured upload directory (e.g. via "..", an absolute path to
        # /etc/passwd, or a Windows system file). When UPLOAD_DIR is unset we
        # fall back to the legacy existence check for back-compat with callers
        # that have not been migrated yet.
        upload_dir = os.environ.get("UPLOAD_DIR")
        if upload_dir:
            try:
                safe_path = get_safe_input_path(upload_dir, resolved_file_path)
            except UnsafeInputPathError as exc:
                return AgentResult(
                    status="failed",
                    error_message=(
                        f"Unsafe input path for IngestionAgent: {exc} "
                        f"(requested_path={resolved_file_path!r}, upload_dir={upload_dir!r})"
                    ),
                )
            resolved_file_path = str(safe_path)
        else:
            if not os.path.exists(resolved_file_path):
                return AgentResult(
                    status="failed",
                    error_message=(
                        f"Input file not found for IngestionAgent: "
                        f"path={resolved_file_path!r}, file_type={file_type!r}"
                    ),
                )

        try:
            if file_type == "csv":
                df = pd.read_csv(resolved_file_path)
            elif file_type == "pdf":
                df = self._parse_pdf(resolved_file_path)
            else:
                df = pd.read_excel(resolved_file_path)

            row_count = len(df)
            column_count = len(df.columns)

            from finflow_agent.tools.dataframe_profile import profile_dataframe
            profile = profile_dataframe(df, include_samples=False)

            return AgentResult(
                status="success",
                data=df,
                summary=f"Successfully ingested {file_type.upper()} file with {row_count} rows and {column_count} columns.",
                metrics={
                    "row_count": row_count,
                    "column_count": column_count,
                    "profile": profile.model_dump(mode="json")
                }
            )
        except Exception as e:
            return AgentResult(
                status="failed",
                error_message=(
                    f"Failed to parse input file for IngestionAgent "
                    f"(file_type={file_type!r}, path={resolved_file_path!r}): {e}"
                ),
            )

    @staticmethod
    def _parse_pdf(file_path: str) -> pd.DataFrame:
        """Extract tabular data from a PDF file using pdfplumber."""
        import pdfplumber

        frames: list[pd.DataFrame] = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for table in tables:
                    if not table or len(table) < 2:
                        continue
                    headers = [str(cell or "").strip() for cell in table[0]]
                    if not any(headers):
                        continue
                    rows = []
                    for row in table[1:]:
                        rows.append([str(cell or "").strip() if cell is not None else "" for cell in row])
                    if rows:
                        frames.append(pd.DataFrame(rows, columns=headers))

        if not frames:
            raise ValueError(
                "No tabular data found in PDF. The file must contain at least "
                "one table with a header row."
            )

        combined = pd.concat(frames, ignore_index=True)
        combined = combined.dropna(how="all")
        combined = combined.loc[:, combined.columns.map(lambda c: bool(str(c).strip()))]
        return combined