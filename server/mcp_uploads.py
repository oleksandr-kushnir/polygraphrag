"""Validate and decode a bounded MCP upload before passing it to document ingestion."""

import base64
import binascii
import io
import re
from pathlib import Path

from fastapi import UploadFile

from server import config
from server.graph_store import graph_error
from server.ingest import SUPPORTED_DOCUMENT_SUFFIXES
from server.schemas import UploadDocumentsInput
from server.uploads import enqueue_uploads

_UNSAFE_FILENAME = re.compile(r'[<>:"/\\|?*\x00-\x1f\x7f\ud800-\udfff]')


def base64_length(size: int) -> int:
    return 4 * ((size + 2) // 3)


async def upload_documents(ws: dict, args: UploadDocumentsInput) -> dict:
    limit = config.MCP_UPLOAD_MAX_BYTES
    if len(args.files) > 8:
        raise graph_error(422, "too_many_documents", "Send at most eight files per upload call.")

    files: list[UploadFile] = []
    metadata = []
    remaining = limit
    size_error = graph_error(
        413,
        "document_upload_too_large",
        f"Decoded files must total at most {limit} bytes per call.",
        limit_bytes=limit,
    )
    for document in args.files:
        name = document.filename
        if _UNSAFE_FILENAME.search(name) or name != name.strip() or len(name.encode()) > 240:
            raise graph_error(422, "invalid_document_filename", "Supply a plain file basename.")
        if Path(name).suffix.lower() not in SUPPORTED_DOCUMENT_SUFFIXES:
            raise graph_error(
                415, "unsupported_document_type", "Supply a supported file extension."
            )
        if len(document.content_base64) > base64_length(remaining):
            raise size_error
        try:
            raw = base64.b64decode(document.content_base64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise graph_error(
                422, "invalid_document_base64", "Send the original file bytes as standard base64."
            ) from exc
        if not raw:
            raise graph_error(422, "empty_document", "Supply a non-empty file.")
        if len(raw) > remaining:
            raise size_error
        remaining -= len(raw)
        files.append(UploadFile(io.BytesIO(raw), filename=name))
        metadata.append(document.model_dump(exclude={"filename", "content_base64"}))

    return await enqueue_uploads(ws, files, metadata)
