"""Negotiate gzip responses and decode bounded gzip request bodies."""

import re
import zlib

from fastapi import HTTPException
from starlette.datastructures import Headers, MutableHeaders
from starlette.middleware.gzip import GZipMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send

_MINIMUM_SIZE = 1024
_CODING = re.compile(
    r"(gzip|identity|\*)\s*(?:;\s*q\s*=\s*(0(?:\.\d{0,3})?|1(?:\.0{0,3})?))?", re.I
)


async def decode_gzip_request(request: Request, limit: int) -> tuple[Scope, Receive]:
    """Inflate a single gzip member and replay it to the ASGI JSON handler."""
    body = bytearray()
    wire_bytes = 0
    decoder = zlib.decompressobj(wbits=31)
    too_large = HTTPException(
        413,
        {
            "code": "mcp_request_too_large",
            "message": f"The gzip request exceeds the MCP body limit of {limit} bytes.",
            "limit_bytes": limit,
        },
    )
    invalid = HTTPException(
        400, {"code": "invalid_gzip_request", "message": "Send one complete gzip member."}
    )
    try:
        async for chunk in request.stream():
            wire_bytes += len(chunk)
            if wire_bytes > limit:
                raise too_large
            body.extend(decoder.decompress(chunk, limit - len(body) + 1))
            if len(body) > limit:
                raise too_large
            if decoder.unused_data:
                raise invalid
    except zlib.error as exc:
        raise invalid from exc
    if not decoder.eof:
        raise invalid

    scope = {**request.scope, "headers": list(request.scope["headers"])}
    headers = MutableHeaders(scope=scope)
    del headers["content-encoding"]
    headers["content-length"] = str(len(body))
    decoded: bytes | None = bytes(body)
    body.clear()

    async def receive() -> Message:
        nonlocal decoded
        if decoded is None:
            return await request.receive()
        chunk, decoded = decoded, None
        return {"type": "http.request", "body": chunk, "more_body": False}

    return scope, receive


def _qualities(headers: Headers) -> tuple[float, float]:
    weights: dict[str, float] = {}
    for line in headers.getlist("accept-encoding"):
        for item in line.split(","):
            match = _CODING.fullmatch(item.strip())
            if match:
                coding, weight = match.groups()
                coding = coding.lower()
                # Use the lower quality if a client repeats a coding with conflicting preferences.
                weights[coding] = min(weights.get(coding, 1), float(weight or "1"))
    wildcard = weights.get("*", 0)
    gzip = weights.get("gzip", wildcard)
    # An implicit identity is an available fallback for any acceptable gzip quality.
    identity = weights.get("identity", 0 if weights.get("*") == 0 else gzip or 1)
    return gzip, identity


def _merge_vary(message: Message) -> None:
    headers = MutableHeaders(raw=message["headers"])
    tokens = {"accept-encoding": "Accept-Encoding"}
    for field in headers.getlist("vary"):
        for value in field.split(","):
            token = value.strip()
            if token:
                tokens.setdefault(token.lower(), token)
    headers["Vary"] = "*" if "*" in tokens else ", ".join(tokens.values())


def _compressible(message: Message) -> bool:
    headers = Headers(raw=message["headers"])
    media_type = headers.get("content-type", "").partition(";")[0].strip().lower()
    return (
        message["status"] not in (204, 205, 304)
        and "content-encoding" not in headers
        and (media_type in ("text/html", "application/json") or media_type.endswith("+json"))
    )


class NegotiatedGZip:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        gzip, identity = _qualities(Headers(scope=scope))
        if not (gzip or identity):
            await Response(status_code=406, headers={"Vary": "Accept-Encoding"})(
                scope, receive, send
            )
            return

        start: Message | None = None

        async def send_response(message: Message) -> None:
            nonlocal start
            if message["type"] == "http.response.start" and _compressible(message):
                start = message
                return
            if start is None:
                await send(message)
                return

            initial, start = start, None
            buffered = (
                message["type"] == "http.response.body"
                and message.get("body")
                and not message.get("more_body")
            )
            if buffered:
                _merge_vary(initial)
            if not buffered or gzip < identity:
                await send(initial)
                await send(message)
                return

            # Give Starlette only the buffered response, with a normalized Accept-Encoding.
            # The route saw the original request and streaming responses bypass compression.
            async def emit(_scope: Scope, _receive: Receive, emit_send: Send) -> None:
                await emit_send(initial)
                await emit_send(message)

            async def send_compressed(event: Message) -> None:
                if event["type"] == "http.response.start":
                    _merge_vary(event)
                await send(event)

            compressor = GZipMiddleware(
                emit, minimum_size=_MINIMUM_SIZE if identity else 0, compresslevel=4
            )
            await compressor(
                {**scope, "headers": [(b"accept-encoding", b"gzip")]}, receive, send_compressed
            )

        await self.app(scope, receive, send_response)
