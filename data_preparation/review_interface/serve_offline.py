#!/usr/bin/env python3
import argparse
import os
import pathlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer


def main():
    parser = argparse.ArgumentParser(description="Serve the offline critic review bundle")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    os.chdir(pathlib.Path(__file__).resolve().parent)
    server = ThreadingHTTPServer((args.host, args.port), SimpleHTTPRequestHandler)
    print(f"Offline review: http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
