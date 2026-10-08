"""The v5 meal pipeline, shared by the demo app (app/server.py) and the Chompass shim (app/chompass_shim.py).

    text --(student model on the 2060 PC, grammar-constrained)--> items [{food, brand, qty, unit, size, state, prep}]
         --(app/fooddb.resolve: FTS search over ~800k foods -> model picks -> portion)--> rows with grams + nutrients

Every number comes from the database; the model only parses and picks. Stdlib only.
The model URL is fooddb.LLAMA (env FD_MODEL_URL, default http://127.0.0.1:8139).
"""
import json, urllib.error, urllib.request
from pathlib import Path

import fooddb

ROOT = Path(__file__).resolve().parent.parent
PARSE_SYS = (ROOT / "prompts" / "student_parse_v5.txt").read_text().strip()
GRAMMAR = (ROOT / "prompts" / "parse_v5.gbnf").read_text()
FIELDS = ("food", "brand", "qty", "unit", "size", "state", "prep")
NUTRIENTS = ("kcal", "protein", "carbs", "fat", "fiber", "sugar", "satfat", "sodium_mg")

class ModelDown(RuntimeError):
    """The model server is not reachable (GPU machine off, or llama-server not running)."""

def _body(text, stream):
    return json.dumps({"temperature": 0, "max_tokens": 900, "stream": stream, "grammar": GRAMMAR,
                       "chat_template_kwargs": {"enable_thinking": False},
                       "messages": [{"role": "system", "content": PARSE_SYS},
                                    {"role": "user", "content": text}]}).encode()

def _items(raw):
    try:
        items = json.loads(raw[raw.find("{"): raw.rfind("}") + 1]).get("items", [])
    except (json.JSONDecodeError, AttributeError):
        return []
    return [{k: it.get(k, 1 if k == "qty" else "") for k in FIELDS} for it in items if isinstance(it, dict)]

def parse(text, on_token=None, url=None, timeout=60):
    """Meal text -> parsed items. on_token(str) is called for each streamed token (demo pipeline view)."""
    req = urllib.request.Request(url or fooddb.LLAMA, _body(text, on_token is not None),
                                 {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            if on_token is None:
                return _items(json.load(r)["choices"][0]["message"]["content"])
            raw = ""
            for line in r:
                line = line.decode().strip()
                if not line.startswith("data:") or line[5:].strip() == "[DONE]": continue
                tok = json.loads(line[5:]).get("choices", [{}])[0].get("delta", {}).get("content")
                if tok:
                    raw += tok; on_token(tok)
            return _items(raw)
    except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
        raise ModelDown(f"model server not reachable at {url or fooddb.LLAMA} ({e}); "
                        "is llama-server running at FD_MODEL_URL?") from e

def analyze(items, c=None, url=None, on_row=None):
    """Parsed items -> (rows, totals). rows = fooddb.resolve() dicts; totals summed over rows with grams.
    on_row(row) is called as each item resolves (demo search view)."""
    c = c or fooddb.connect()
    rows = []
    for it in items:
        try:
            rows.append(fooddb.resolve(c, it, url=url))
            if on_row: on_row(rows[-1])
        except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
            raise ModelDown(f"model server not reachable during pick ({e})") from e
    totals = {n: round(sum(r.get(n) or 0 for r in rows if r.get("grams")), 1) for n in NUTRIENTS}
    totals["grams"] = round(sum(r["grams"] for r in rows if r.get("grams")), 1)
    return rows, totals

def run(text, on_token=None, url=None):
    """Full chain: text -> (items, rows, totals)."""
    items = parse(text, on_token, url)
    rows, totals = analyze(items, url=url)
    return items, rows, totals
