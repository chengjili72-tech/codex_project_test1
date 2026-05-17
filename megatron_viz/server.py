"""Small stdlib web service for uploading and comparing Megatron configs."""

from __future__ import annotations

import html
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import NamedTuple
from urllib.parse import parse_qs, urlparse

from megatron_viz.inputs.loader import config_from_text
from megatron_viz.renderers.compare_html import render_compare_html
from megatron_viz.renderers.html import render_html

_MAX_UPLOAD_BYTES = 16 * 1024 * 1024


class UploadedFile(NamedTuple):
    """One uploaded form file."""

    name: str
    filename: str
    content: bytes


def _index_html() -> str:
    return """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Megatron Viz Service</title>
  <style>
    body { margin:0; font-family:Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background:#f8fafc; color:#172033; }
    header { padding:34px 42px; background:linear-gradient(135deg,#172033,#2563eb); color:white; }
    main { max-width:1080px; margin:0 auto; padding:30px 42px 60px; display:grid; grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); gap:22px; }
    form { background:white; border:1px solid #dbe4f0; border-radius:18px; padding:24px; box-shadow:0 12px 30px rgba(15,23,42,.08); }
    label { display:block; margin:14px 0 6px; font-weight:700; }
    input, select, button { width:100%; padding:10px 12px; border-radius:10px; border:1px solid #cbd5e1; }
    button { margin-top:18px; color:white; background:#2563eb; border:0; font-weight:800; cursor:pointer; }
    p { color:#475569; }
  </style>
</head>
<body>
  <header><h1>Megatron-LM 在线可视化服务</h1><p>上传 sh 或 log 文件，在线生成模型结构；上传两份配置可进行关键参数差异标注。</p></header>
  <main>
    <form method="post" action="/visualize" enctype="multipart/form-data">
      <h2>生成模型结构</h2>
      <p>支持启动脚本 .sh、训练日志 .log/.txt，输出浏览器 HTML 可视化。</p>
      <label for="file">上传文件</label>
      <input id="file" name="file" type="file" required />
      <label for="kind">解析类型</label>
      <select id="kind" name="kind"><option value="auto">自动识别</option><option value="script">启动脚本/命令</option><option value="log">训练日志</option></select>
      <button type="submit">生成可视化</button>
    </form>
    <form method="post" action="/compare" enctype="multipart/form-data">
      <h2>对比两份配置</h2>
      <p>对模型结构、TP/PP/EP/CP/DP、batch、精度等关键参数做高亮差异标注。</p>
      <label for="left_file">基准文件</label>
      <input id="left_file" name="left_file" type="file" required />
      <label for="right_file">对比文件</label>
      <input id="right_file" name="right_file" type="file" required />
      <label for="compare_kind">解析类型</label>
      <select id="compare_kind" name="kind"><option value="auto">自动识别</option><option value="script">启动脚本/命令</option><option value="log">训练日志</option></select>
      <button type="submit">生成对比报告</button>
    </form>
  </main>
</body>
</html>
"""


def _decode_text(upload: UploadedFile) -> str:
    return upload.content.decode("utf-8", errors="replace")


def _split_multipart(body: bytes, boundary: bytes) -> list[bytes]:
    delimiter = b"--" + boundary
    parts = []
    for raw_part in body.split(delimiter):
        part = raw_part.strip(b"\r\n")
        if not part or part == b"--":
            continue
        if part.endswith(b"--"):
            part = part[:-2].rstrip(b"\r\n")
        parts.append(part)
    return parts


def _parse_content_disposition(value: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in value.split(";"):
        item = item.strip()
        if "=" not in item:
            continue
        key, raw_value = item.split("=", 1)
        result[key.strip().lower()] = raw_value.strip().strip('"')
    return result


def parse_multipart_form(body: bytes, content_type: str) -> tuple[dict[str, str], dict[str, UploadedFile]]:
    """Parse a small multipart/form-data request using only the standard library."""

    _, _, params_text = content_type.partition(";")
    params = _parse_content_disposition(params_text)
    boundary = params.get("boundary")
    if not boundary:
        raise ValueError("missing multipart boundary")

    fields: dict[str, str] = {}
    files: dict[str, UploadedFile] = {}
    for part in _split_multipart(body, boundary.encode()):
        raw_headers, separator, content = part.partition(b"\r\n\r\n")
        if not separator:
            continue
        headers: dict[str, str] = {}
        for line in raw_headers.decode("utf-8", errors="replace").split("\r\n"):
            if ":" in line:
                key, value = line.split(":", 1)
                headers[key.strip().lower()] = value.strip()
        disposition = _parse_content_disposition(headers.get("content-disposition", ""))
        name = disposition.get("name")
        if not name:
            continue
        content = content.rstrip(b"\r\n")
        filename = disposition.get("filename")
        if filename:
            files[name] = UploadedFile(name=name, filename=Path(filename).name, content=content)
        else:
            fields[name] = content.decode("utf-8", errors="replace")
    return fields, files


class MegatronVizHandler(BaseHTTPRequestHandler):
    """HTTP handler for visualization and comparison uploads."""

    server_version = "MegatronViz/0.1"

    def _send_html(self, body: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        payload = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_json(self, data: dict[str, object], status: HTTPStatus = HTTPStatus.OK) -> None:
        payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _error(self, message: str, status: HTTPStatus = HTTPStatus.BAD_REQUEST) -> None:
        escaped = html.escape(message)
        self._send_html(f"<h1>请求失败</h1><p>{escaped}</p><p><a href='/'>返回首页</a></p>", status)

    def _read_form(self) -> tuple[dict[str, str], dict[str, UploadedFile]]:
        length = int(self.headers.get("Content-Length", "0"))
        if length > _MAX_UPLOAD_BYTES:
            raise ValueError("uploaded payload is too large")
        content_type = self.headers.get("Content-Type", "")
        body = self.rfile.read(length)
        if content_type.startswith("multipart/form-data"):
            return parse_multipart_form(body, content_type)
        fields = {key: values[-1] for key, values in parse_qs(body.decode("utf-8", errors="replace")).items()}
        return fields, {}

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/" or path == "/index.html":
            self._send_html(_index_html())
        elif path == "/healthz":
            self._send_json({"ok": True})
        else:
            self._error("not found", HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        try:
            fields, files = self._read_form()
            kind = fields.get("kind", "auto")
            path = urlparse(self.path).path
            if path == "/visualize":
                upload = files.get("file")
                if upload is None:
                    self._error("missing uploaded file")
                    return
                config = config_from_text(_decode_text(upload), filename=upload.filename, kind=kind)
                self._send_html(render_html(config))
            elif path == "/compare":
                left = files.get("left_file")
                right = files.get("right_file")
                if left is None or right is None:
                    self._error("missing comparison files")
                    return
                left_config = config_from_text(_decode_text(left), filename=left.filename, kind=kind)
                right_config = config_from_text(_decode_text(right), filename=right.filename, kind=kind)
                self._send_html(render_compare_html(left_config, right_config, left_name=left.filename, right_name=right.filename))
            else:
                self._error("not found", HTTPStatus.NOT_FOUND)
        except ValueError as exc:
            self._error(str(exc))

    def log_message(self, format: str, *args: object) -> None:
        print(f"{self.address_string()} - {format % args}")


def run_server(host: str = "127.0.0.1", port: int = 8765) -> None:
    """Run the upload web service until interrupted."""

    server = ThreadingHTTPServer((host, port), MegatronVizHandler)
    print(f"Megatron Viz service listening on http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Megatron Viz service")
    finally:
        server.server_close()
