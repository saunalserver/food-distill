#!/usr/bin/env python3
"""food-distill demo app (v5). Stdlib only. Listens on FD_HOST:FD_PORT (default 127.0.0.1:8138).

  meal text -> student model in llama-server (FD_MODEL_URL, streamed tokens)
            -> parsed items -> FTS search over ~800k foods -> model picks -> portion -> nutrients
            -> table + log (data/app.sqlite)

The model never runs on this server; only the database search and the arithmetic do (app/pipeline.py).
The v3 app (151-food vocabulary, local llama :8137) is kept as app/server_v3.py for reference.

Run: python3 app/server.py     (systemd user unit food-distill-app)
"""
import json, os, sqlite3, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pipeline

ROOT = Path(__file__).resolve().parent.parent
HOST, PORT = os.environ.get("FD_HOST", "127.0.0.1"), int(os.environ.get("FD_PORT", 8138))

def food_count():
    c = pipeline.fooddb.connect()
    n = c.execute("select count(*) from food").fetchone()[0]
    c.close()
    return n

N_FOODS = food_count()

def log_db():
    c = sqlite3.connect(ROOT / "data" / "app.sqlite")
    c.execute("CREATE TABLE IF NOT EXISTS log (ts TEXT, text TEXT, items TEXT, kcal REAL)")
    return c

class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a): pass

    def _send(self, code, body, ctype="text/html; charset=utf-8"):
        b = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        path = self.path.split("?")[0]            # /?q=<meal> runs a meal on load (index.html)
        if path in ("/", "/index.html"):
            self._send(200, (ROOT / "app" / "index.html").read_text().replace("{{N_FOODS}}", f"{N_FOODS:,}"))
        elif path == "/bench.json":     # vs-Gemma panel; built by scripts/build_bench_json.py
            self._send(200, (ROOT / "app" / "bench.json").read_text(), "application/json")
        elif path == "/manifest.webmanifest":
            self._send(200, json.dumps({"name": "food-distill", "display": "standalone",
                "background_color": "#111", "theme_color": "#111",
                "icons": [{"src": "/icon.svg", "sizes": "any", "type": "image/svg+xml"}]}),
                "application/manifest+json")
        elif path == "/icon.svg":
            self._send(200, '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
                '<rect width="100" height="100" rx="20" fill="#111"/>'
                '<text x="50" y="68" font-size="52" text-anchor="middle">🍔</text></svg>', "image/svg+xml")
        else:
            self._send(404, "nope", "text/plain")

    def do_POST(self):
        if self.path != "/api/parse":
            self._send(404, "nope", "text/plain"); return
        text = json.loads(self.rfile.read(int(self.headers["Content-Length"])))["text"]
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

        def sse(event, data):
            self.wfile.write(f"event: {event}\ndata: {json.dumps(data)}\n\n".encode())
            self.wfile.flush()

        t0 = time.time()
        try:
            items = pipeline.parse(text, on_token=lambda t: sse("tok", {"t": t}))
            t_parse = time.time() - t0
            sse("items", {"items": items})
            rows, totals = pipeline.analyze(items, on_row=lambda r: sse("row", r))
        except pipeline.ModelDown as e:
            sse("err", {"msg": str(e)}); return
        except Exception as e:          # never leave the page hanging
            sse("err", {"msg": f"{type(e).__name__}: {e}"}); return
        c = log_db()
        c.execute("insert into log values (?,?,?,?)",
                  (time.strftime("%Y-%m-%dT%H:%M:%S"), text, json.dumps(items), totals["kcal"]))
        c.commit(); c.close()
        sse("done", {"rows": rows, "totals": totals, "dt": round(time.time() - t0, 2), "dt_parse": round(t_parse, 2)})

if __name__ == "__main__":
    print(f"http://{HOST}:{PORT}")
    ThreadingHTTPServer((HOST, PORT), H).serve_forever()
