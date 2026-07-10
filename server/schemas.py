"""Pydantic request/response models and the shared query-parameter constraints.

These describe the JSON bodies accepted and returned by the query, document, and workspace
endpoints. Keeping them here (rather than inline in the app module) makes the API contract
easy to find, lets the routers import a stable schema surface, and — for response models —
puts the real field names, nullability, and the job-status enum into /openapi.json where
tool-calling agents can read them.
"""

from typing import Literal

from pydantic import BaseModel, Field, model_validator

_MODE_DESC = (
    "Retrieval mode: 'mix' (default, graph + vector — recommended), 'local' (entity-centric), "
    "'global' (relationship/theme-centric), 'hybrid' (local + global), or 'naive' (plain vector search)."
)
# Constrain mode to the modes LightRAG actually supports: an unknown value is rejected with a
# 422 up front instead of being passed through to fail (or misbehave) deep in retrieval.
QueryMode = Literal["mix", "local", "global", "hybrid", "naive"]
# Upper bound on top_k: the default is 40; a very large value multiplies retrieval + LLM cost,
# so a single request can't (accidentally or maliciously) run the corpus/LLM budget away.
_TOP_K_MAX = 1000


class QueryRequest(BaseModel):
    query: str = Field(
        description="The natural-language question to ask the workspace's corpus.",
        examples=["What did the Q3 report say about churn?"],
    )
    mode: QueryMode = Field("mix", description=_MODE_DESC)
    include_references: bool = Field(
        True, description="Include source-document citations in the response. Default true."
    )
    # LightRAG's tuned default (entities/relations retrieved per keyword set).
    top_k: int = Field(
        40,
        ge=1,
        le=_TOP_K_MAX,
        description=f"Entities/relationships retrieved per keyword set. Default 40, max {_TOP_K_MAX}.",
    )


class QueryDataRequest(BaseModel):
    query: str = Field(
        description="The natural-language question used to retrieve graph/vector data.",
        examples=["List the entities related to onboarding."],
    )
    mode: QueryMode = Field("mix", description=_MODE_DESC)
    include_references: bool = Field(
        True,
        description="Resolve and include source-document references for retrieved data. Default true.",
    )
    top_k: int = Field(
        40,
        ge=1,
        le=_TOP_K_MAX,
        description=f"Entities/relationships retrieved per keyword set. Default 40, max {_TOP_K_MAX}.",
    )
    file_path_contains: list[str] = Field(
        default_factory=list,
        description=(
            "Optional folder/file scope filter. **Omit it, or leave it empty, to get ALL data "
            "(no filtering) — this is the default.** When provided, it is a case-insensitive OR "
            "substring filter on each result's file_path: an entity/relationship/chunk/reference "
            "is kept if its file_path contains ANY of the strings (blank strings are ignored). "
            "Matching runs AFTER retrieval (the retrieval budget is auto-boosted when set), so a "
            "very narrow folder may return fewer items than exist. "
            'Example (to narrow): ["/corpus/career/", "/corpus/projects/"].'
        ),
    )


class WorkspaceCreate(BaseModel):
    id: str = Field(
        description="Workspace slug — must match ^[a-z][a-z0-9_]{0,47}$. Also used as the storage namespace.",
        examples=["acme_corp"],
    )
    name: str = Field(
        description="Human-readable display name for the workspace.", examples=["Acme Corp"]
    )
    description: str | None = Field(
        None, description="Optional free-text description of the workspace."
    )
    # `lightrag_workspace` is deliberately NOT a field: for API-created workspaces the
    # service forces lightrag_workspace == id. Any client-supplied value is ignored.


class FileDeleteRequest(BaseModel):
    rel_path: str | None = Field(
        None,
        description="Workspace-relative path of the file (matched against the stored source_path).",
    )
    external_path: str | None = Field(
        None,
        description=(
            "Caller-supplied absolute path — the real path recorded at upload "
            "(`path_root/source_path`)."
        ),
    )
    doc_id: str | None = Field(
        None, description="LightRAG doc id (`doc-<md5>`). If given, used directly — most precise."
    )

    @model_validator(mode="after")
    def _require_one_identifier(self) -> "FileDeleteRequest":
        """Absent *file* → noop is right; absent *identifier* is a client error. Without this,
        a misspelled field name would read as silent success."""
        if not (self.doc_id or self.external_path or self.rel_path):
            raise ValueError(
                "Provide at least one identifier: doc_id, external_path, or rel_path."
            )
        return self


# --------------------------------------------------------------------------- #
# Response models — the machine-readable API contract in /openapi.json
# --------------------------------------------------------------------------- #

# Canonical ingest-JOB status vocabulary. Distinct from LightRAG's internal per-document
# status (which includes 'processed'): a finished job is 'done'. 'save_failed' means the
# uploaded bytes could not be written before processing started.
JobStatus = Literal["pending", "processing", "retrying", "done", "failed", "save_failed"]


class HealthResponse(BaseModel):
    status: str = Field(examples=["ok"])


class Reference(BaseModel):
    """One resolved source-document citation. `file_path` is the REAL, openable path recorded
    at upload (`path_root/source_path`), or null when the reference can't be resolved to a
    stored file — LightRAG's internal name is never emitted."""

    reference_id: str | int | None = None
    file_path: str | None = None
    job_id: str | None = None
    file_description: str | None = None
    last_modified_time: str | None = None
    uploaded_at: str | None = None
    llm_model_extracted: str | None = Field(
        None, description="Text LLM that extracted this document's entities at ingest time."
    )
    llm_model_answered: str | None = Field(
        None, description="Text LLM that synthesised the answer (present on /query only)."
    )


class QueryResponse(BaseModel):
    result: str = Field(description="The synthesised natural-language answer.")
    references: list[Reference] = Field(
        description="Source citations; empty when include_references is false."
    )


class FileIndexEntry(BaseModel):
    """One row of the durable per-file index (`/files`)."""

    job_id: str | None = None
    file: str | None = Field(None, description="Original uploaded filename.")
    file_path: str | None = Field(
        None, description="Real display path recorded at upload (path_root/source_path)."
    )
    source_path: str | None = None
    doc_id: str | None = Field(None, description="LightRAG doc id (`doc-<md5>`).")
    content_hash: str | None = Field(None, description="SHA-256 of the ingested bytes.")
    status: JobStatus
    last_modified_time: str | None = None
    uploaded_at: str | None = None


class JobRecord(FileIndexEntry):
    """An ingestion job: the file-index fields plus job bookkeeping."""

    batch_id: str | None = None
    attempts: int | None = None
    error: str | None = None
    description: str | None = None


class BatchResponse(BaseModel):
    batch_id: str
    summary: dict[str, int] = Field(
        description="Per-status job counts for the batch, plus 'total'."
    )
    jobs: list[JobRecord]


class JobsResponse(BaseModel):
    jobs: list[JobRecord]


class FilesResponse(BaseModel):
    files: list[FileIndexEntry]


class FileDeleteResponse(BaseModel):
    status: Literal["deleted", "noop"]
    doc_id: str | None = None
    reason: str | None = Field(None, description="Set to 'not_found' on a noop.")


class WorkspacePublic(BaseModel):
    id: str
    name: str
    description: str | None = None
    created_at: str | None = None


class WorkspaceListResponse(BaseModel):
    workspaces: list[WorkspacePublic]


class WorkspaceActionResponse(BaseModel):
    status: Literal["soft-deleted", "purged", "restored"]
    id: str


class DocumentCounts(BaseModel):
    by_status: dict[str, int] = Field(
        description="LightRAG per-DOCUMENT statuses (a different vocabulary than job status)."
    )
    total: int


class IngestSummary(BaseModel):
    by_status: dict[str, int] = Field(description="Ingest-JOB counts by status.")
    last_uploaded_at: str | None = None


class WorkspaceOverview(BaseModel):
    id: str
    name: str
    description: str | None = None
    active: bool = Field(description="False when the workspace is soft-deleted.")
    documents: DocumentCounts
    chunks: int
    entities: int | None = None
    relationships: int | None = None
    ingest: IngestSummary
