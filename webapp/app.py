"""
Flavor Swap: find an alternative ingredient by flavor chemistry or by nutrition.

Local:       python3 webapp/app.py            -> opens http://127.0.0.1:8000 in Chrome
             (uses the next free port if 8000 is busy; add --no-browser to skip opening Chrome)
Production:  gunicorn --chdir webapp app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120

API
  GET /api/search?q=but                      ingredient autocomplete
  GET /api/ingredient/<name>                 profile, nutrients, what real cooks substituted
  GET /api/recommend?ingredient=butter&mode=flavor|nutrient&k=8
        flavor:   &weighting=jaccard|idf
        nutrient: &catalog=all|recipe &features=full|macro &lower=fat,sugar &higher=protein
        both:     &same_category=1 &min_confidence=Medium|High
  GET /api/shared?a=lemon&b=lime             flavor compounds two ingredients share
"""
from __future__ import annotations

import os
import time
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from service import SubstitutionService

HERE = Path(__file__).resolve().parent
app = Flask(__name__, static_folder=str(HERE / "static"), static_url_path="/static")
app.json.sort_keys = False

t0 = time.time()
service = SubstitutionService()
print(f"models loaded in {time.time() - t0:.1f}s")

def _flag(name: str) -> bool:
    return request.args.get(name, "").lower() in {"1", "true", "yes", "on"}


def _list(name: str) -> list[str]:
    return [x.strip() for x in request.args.get(name, "").split(",") if x.strip()]


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/search")
def api_search():
    return jsonify(service.search(request.args.get("q", "")))


@app.get("/api/ingredient/<name>")
def api_ingredient(name):
    try:
        return jsonify(service.ingredient(name))
    except KeyError:
        return jsonify(error=f"No ingredient called '{name}'. Try the search box."), 404


@app.get("/api/recommend")
def api_recommend():
    name = request.args.get("ingredient", "")
    try:
        out = service.recommend(
            name,
            mode=request.args.get("mode", "flavor"),
            k=request.args.get("k", 8),
            weighting=request.args.get("weighting", "jaccard"),
            catalog=request.args.get("catalog", "all"),
            features=request.args.get("features", "full"),
            lower=_list("lower"), higher=_list("higher"),
            same_category=_flag("same_category"),
            min_confidence=request.args.get("min_confidence") or None,
        )
        return jsonify(out)
    except KeyError:
        return jsonify(error=f"No ingredient called '{name}'. Try the search box."), 404
    except (LookupError, ValueError) as e:
        return jsonify(error=str(e)), 400


@app.get("/api/shared")
def api_shared():
    try:
        return jsonify(service.shared_compounds(request.args.get("a", ""), request.args.get("b", "")))
    except KeyError:
        return jsonify(error="Both ingredients need flavor-compound data."), 404


@app.get("/healthz")
def health():
    return {"ok": True}


if __name__ == "__main__":
    import socket
    import subprocess
    import sys
    import threading
    import webbrowser

    def free_port(start: int) -> int:
        """First port from `start` upward that nothing is listening on."""
        for port in range(start, start + 50):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex(("127.0.0.1", port)) != 0:
                    return port
        raise SystemExit(f"No free port between {start} and {start + 49}.")

    def open_in_chrome(url: str) -> None:
        """Open the site in Google Chrome; fall back to the default browser."""
        try:
            if sys.platform == "darwin":
                subprocess.run(["open", "-a", "Google Chrome", url], check=True, capture_output=True)
                return
            for name in ("google-chrome", "chrome", "chromium", "chromium-browser"):
                try:
                    webbrowser.get(name).open(url)
                    return
                except webbrowser.Error:
                    continue
        except (OSError, subprocess.CalledProcessError):
            pass
        webbrowser.open(url)

    wanted = int(os.environ.get("PORT", 8000))
    port = free_port(wanted)
    if port != wanted:
        print(f"Port {wanted} is busy, using {port} instead.")
    url = f"http://127.0.0.1:{port}"
    print(f"\n  Flavor Swap is running at {url}\n  Press Ctrl+C to stop.\n")
    if "--no-browser" not in sys.argv:
        threading.Timer(1.0, open_in_chrome, args=(url,)).start()   # after the server is listening
    app.run(host="127.0.0.1", port=port, debug=False)
