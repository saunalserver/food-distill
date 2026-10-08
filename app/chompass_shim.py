#!/usr/bin/env python3
"""Chompass ↔ food-distill shim: an OpenAI-compatible endpoint that serves ONE model, `food-distill`.

Stdlib only. Design: docs/chompass-integration.md. Kept deliberately simple: no forwarding,
no fallbacks — only the small food model.

  Chompass (Custom provider, base https://<your-host>:8444/v1, model food-distill)
    -> tailscale serve --https=8444 -> this shim on 127.0.0.1:8140
       * POST /v1/chat/completions whose text starts "Estimate the nutritional content for a food logging app."
         (typed meal, no images) -> unwrap <user_data> -> app/pipeline (student model on the 2060 PC + 825k-food DB)
         -> Chompass FoodAnalysis JSON (meal totals = exact sums of the per-item constituents)
       * Coach chat (request carries the propose_log_food tool): a food-only "coach" —
           food text -> tool call propose_log_food(description=<the user's words>) -> the app then runs its normal
           meal analysis (= the meal path above) and sends the result back -> one-line summary to confirm in the app;
           non-food text -> "I can only log food"; model offline -> says so. No chat model behind it.
       * anything else (photos, labels) -> 400 "food-distill only handles typed meals"
       * model server unreachable (GPU machine off / llama-server not running) -> 503 with that message
       * GET /v1/models -> [food-distill]

Run: python3 app/chompass_shim.py      (systemd user unit food-distill-chompass)
"""
import json, os, re, sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipeline  # noqa: E402

HOST, PORT = "127.0.0.1", int(os.environ.get("FD_SHIM_PORT", 8140))
OUR_MODEL = "food-distill"
ENTRY = "Estimate the nutritional content for a food logging app."
USER_DATA = re.compile(r"User description[^\n]*\n\s*<user_data>\s*(.*?)\s*</user_data>", re.S)
LOG = Path(__file__).resolve().parent.parent / "out" / "chompass_shim.log"

def log(msg):
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f: f.write(line + "\n")

def text_parts(messages):
    """(texts of all user parts, has_image, has_system_or_history)."""
    texts, img = [], False
    for m in messages:
        c = m.get("content")
        if isinstance(c, str): texts.append(c)
        elif isinstance(c, list):
            for p in c:
                if p.get("type") == "text": texts.append(p.get("text", ""))
                elif p.get("type") in ("image_url", "input_image", "image"): img = True
    return texts, img, len(messages) > 1 or any(m.get("role") == "system" for m in messages)

def meal_text(req):
    """The user's meal text if this is a Chompass text-entry request we should answer, else None."""
    if req.get("tools"): return None
    texts, img, multi = text_parts(req.get("messages", []))
    if img or multi or not texts: return None
    t = "\n".join(texts)
    if ENTRY not in t: return None      # anywhere: Chompass prepends "User context …" / a language line when set
    m = USER_DATA.search(t)
    return m.group(1).strip() if m else None

def r1(x): return round(float(x or 0), 1)

def chompass_json(rows):
    """Pipeline rows -> Chompass FoodAnalysis JSON. Totals are sums of the constituents (ConstituentReconcile-safe)."""
    cons = []
    for r in rows:
        if not r.get("grams"): continue
        c = {"name": r["food"].title(), "calories": int(round(r.get("kcal") or 0)),
             "protein": r1(r.get("protein")), "carbs": r1(r.get("carbs")), "fat": r1(r.get("fat")),
             "serving_size_grams": r1(r["grams"]), "fiber": r1(r.get("fiber")), "sugar": r1(r.get("sugar")),
             "saturated_fat": r1(r.get("satfat")), "sodium": r1(r.get("sodium_mg"))}
        if r["unit"] not in ("g", "kg", "oz", "lb") and r.get("qty"):
            c["unit_options"] = [{"unit": r["unit"], "quantity": r["qty"], "grams_per_unit": r1(r["grams"] / float(r["qty"]))}]
        cons.append(c)
    if not cons: return None
    tot = lambda k: r1(sum(c[k] for c in cons))
    name = cons[0]["name"] if len(cons) == 1 else ", ".join(c["name"] for c in cons[:3]) + (" +" if len(cons) > 3 else "")
    out = {"name": name, "calories": sum(c["calories"] for c in cons), "protein": tot("protein"),
           "carbs": tot("carbs"), "fat": tot("fat"), "serving_size_grams": tot("serving_size_grams"),
           "fiber": tot("fiber"), "sugar": tot("sugar"), "saturated_fat": tot("saturated_fat"),
           "sodium": tot("sodium"), "unit_options": cons[0].get("unit_options", []) if len(cons) == 1 else []}
    if len(cons) > 1: out["constituents"] = cons[:12]
    return out

class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a): pass

    def _reply(self, code, body, ctype="application/json"):
        b = body if isinstance(body, bytes) else body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path.rstrip("/") in ("/v1/models", "/models"):
            self._reply(200, json.dumps({"object": "list", "data": [{"id": OUR_MODEL, "object": "model",
                                                                     "owned_by": "food-distill"}]}))
        elif self.path in ("/", "/health"):
            self._reply(200, json.dumps({"ok": True, "model": OUR_MODEL}))
        else:
            self._error(404, "not found")

    def _error(self, code, msg):
        self._reply(code, json.dumps({"error": {"message": msg, "type": "food_distill"}}))

    def _chat(self, message, finish):
        self._reply(200, json.dumps({"id": f"chatcmpl-fd{int(time.time() * 1000)}", "object": "chat.completion",
                                     "created": int(time.time()), "model": OUR_MODEL,
                                     "choices": [{"index": 0, "finish_reason": finish, "message": message}],
                                     "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}))

    def coach(self, req):
        """Food-only coach: turn food text into a propose_log_food tool call; summarize the tool's result."""
        msgs = req.get("messages", [])
        say = lambda text: self._chat({"role": "assistant", "content": text}, "stop")
        last = msgs[-1] if msgs else {}
        if last.get("role") == "tool":                        # the app ran propose_log_food and sends the result
            try: res = json.loads(last.get("content") or "{}")
            except json.JSONDecodeError: res = {}
            if res.get("proposed"):
                log(f"coach: proposed {res.get('name')!r} {res.get('calories')} kcal")
                return say(f"{res.get('name')}: {res.get('calories')} kcal (protein {res.get('protein_g')} g, carbs "
                           f"{res.get('carbs_g')} g, fat {res.get('fat_g')} g). Confirm it in the app to log it.")
            return say(f"I couldn't estimate that: {res.get('error') or 'unknown error'}")
        texts, _, _ = text_parts([m for m in msgs if m.get("role") == "user"][-1:])
        text = " ".join(texts).strip()
        try:
            items = pipeline.parse(text) if text else []
        except pipeline.ModelDown:
            log("coach: model down")
            return say("The food model is offline. Start llama-server on the GPU machine "
                       "(the 2060 PC must be on).")
        if not items:
            log(f"coach: no food in {text!r}")
            return say("I'm the food-logging model: I can only log what you ate. Tell me a meal, e.g. \"2 eggs and toast\".")
        log(f"coach: propose_log_food {text!r}")
        return self._chat({"role": "assistant", "content": None, "tool_calls": [{
            "id": f"call_fd{int(time.time() * 1000)}", "type": "function",
            "function": {"name": "propose_log_food", "arguments": json.dumps({"description": text})}}]}, "tool_calls")

    def do_POST(self):
        raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        try: req = json.loads(raw)
        except json.JSONDecodeError: req = None
        if req is not None and any((t.get("function") or {}).get("name") == "propose_log_food" for t in req.get("tools") or []):
            return self.coach(req)
        meal = meal_text(req) if req is not None and self.path.rstrip("/").endswith("/chat/completions") else None
        if meal is None:
            log(f"rejected non-meal request {self.path}")
            try:   # keep a copy (images stripped) to see what the app actually sent
                slim = json.loads(json.dumps(req)) if req else {"raw": raw[:2000].decode("utf-8", "replace")}
                for m in (slim.get("messages") or []):
                    if isinstance(m.get("content"), list):
                        m["content"] = [p if p.get("type") == "text" else {"type": p.get("type")} for p in m["content"]]
                with open(LOG.with_name("chompass_rejected.jsonl"), "a") as f:
                    f.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "req": slim})[:20000] + "\n")
            except Exception as e:
                log(f"could not save rejected request: {e}")
            return self._error(400, "food-distill only handles typed meals (no photos, labels or chat)")
        t0 = time.time()
        try:
            items, rows, totals = pipeline.run(meal)
        except pipeline.ModelDown as e:
            log(f"model down for {meal!r}: {e}")
            return self._error(503, "food-distill model is offline: start llama-server on the GPU machine")
        answer = chompass_json(rows)
        if answer is None:
            log(f"no food found in {meal!r}")
            return self._error(422, "no food found in that description")
        content = json.dumps(answer, ensure_ascii=False)
        log(f"food-distill {meal!r} -> {answer['calories']} kcal, {len(rows)} items, {time.time() - t0:.2f}s")
        created, cid = int(time.time()), f"chatcmpl-fd{int(time.time() * 1000)}"
        if req.get("stream"):
            chunk = lambda delta, fin: json.dumps({"id": cid, "object": "chat.completion.chunk", "created": created,
                                                   "model": OUR_MODEL, "choices": [{"index": 0, "delta": delta,
                                                                                    "finish_reason": fin}]})
            body = (f"data: {chunk({'role': 'assistant', 'content': content}, None)}\n\n"
                    f"data: {chunk({}, 'stop')}\n\ndata: [DONE]\n\n")
            return self._reply(200, body, "text/event-stream")     # one write: headers + whole stream
        self._reply(200, json.dumps({"id": cid, "object": "chat.completion", "created": created, "model": OUR_MODEL,
                                     "choices": [{"index": 0, "finish_reason": "stop",
                                                  "message": {"role": "assistant", "content": content}}],
                                     "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}))

if __name__ == "__main__":
    log(f"chompass shim on http://{HOST}:{PORT} (food-distill only)")
    ThreadingHTTPServer((HOST, PORT), H).serve_forever()
