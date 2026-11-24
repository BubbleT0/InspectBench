from __future__ import annotations

import argparse
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit


def create_server(
    report_root: Path,
    host: str,
    port: int,
) -> ThreadingHTTPServer:
    report_root = report_root.resolve(strict=True)
    manifest = json.loads((report_root / "manifest.json").read_text("utf-8"))
    input_root = Path(manifest["input_root"]).resolve(strict=True)
    image_map = {}
    for row in manifest["images"]:
        source = Path(row["source_path"]).resolve(strict=True)
        if not source.is_relative_to(input_root):
            raise ValueError("Manifest image is outside the configured input root.")
        image_map[row["image_id"]] = (source, row.get("mime_type", "application/octet-stream"))

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            raw_path = urlsplit(self.path).path
            if "%" in raw_path or "\\" in raw_path or "\x00" in raw_path:
                self._error()
                return
            if raw_path in {"/", "/index.html"}:
                self._send_file(report_root / "index.html", "text/html; charset=utf-8")
                return
            prefix = "/images/"
            if raw_path.startswith(prefix):
                image_id = raw_path[len(prefix) :]
                if (
                    not image_id
                    or "/" in image_id
                    or "." in image_id
                    or image_id not in image_map
                ):
                    self._error()
                    return
                path, mime_type = image_map[image_id]
                self._send_file(path, mime_type)
                return
            self._error()

        def _send_file(self, path: Path, content_type: str) -> None:
            data = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "private, max-age=3600")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(data)

        def _error(self) -> None:
            data = b"Not found"
            self.send_response(404)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, format, *args):
            return

    return ThreadingHTTPServer((host, port), Handler)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=11223, type=int)
    args = parser.parse_args(argv)
    server = create_server(args.output_root, args.host, args.port)
    print(f"Serving on http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
