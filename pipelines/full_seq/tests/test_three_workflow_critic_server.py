from __future__ import annotations

import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from u1_critic_v41.serve import create_server


class RestrictedServerTests(unittest.TestCase):
    def test_only_report_and_manifest_images_are_served(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "allowed.jpg"
            image.write_bytes(b"allowed image")
            (root / "index.html").write_text("report", encoding="utf-8")
            (root / "manifest.json").write_text(
                json.dumps(
                    {
                        "input_root": str(root),
                        "images": [
                            {
                                "image_id": "img_allowed",
                                "source_path": str(image),
                                "mime_type": "image/jpeg",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            server = create_server(root, "127.0.0.1", 0)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                self.assertEqual(urllib.request.urlopen(base + "/").read(), b"report")
                self.assertEqual(
                    urllib.request.urlopen(base + "/images/img_allowed").read(),
                    b"allowed image",
                )
                for path in (
                    "/images/unknown",
                    "/manifest.json",
                    "/images/..%2Fmanifest.json",
                ):
                    with self.subTest(path=path):
                        with self.assertRaises(urllib.error.HTTPError):
                            urllib.request.urlopen(base + path)
            finally:
                server.shutdown()
                server.server_close()
                thread.join()


if __name__ == "__main__":
    unittest.main()
