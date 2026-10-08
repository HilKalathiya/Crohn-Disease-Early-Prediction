"""One-command launcher:  python run.py   ->  open http://127.0.0.1:8000"""
import os, sys, webbrowser, threading
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
try:
    import uvicorn
except ImportError:
    sys.exit("Missing packages. Run:  pip install -r requirements.txt")
PORT = int(os.environ.get("PORT", 8000))
if __name__ == "__main__":
    if "--no-browser" not in sys.argv:
        threading.Timer(2.0, lambda: webbrowser.open(f"http://127.0.0.1:{PORT}")).start()
    print(f"\n  CrohnSight running at http://127.0.0.1:{PORT}   (Ctrl+C to stop)\n")
    uvicorn.run("backend.app:app", host="127.0.0.1", port=PORT, log_level="warning")
