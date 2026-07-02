import json
import mimetypes
import subprocess
import sys
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent
TEMPLATES_DIR = ROOT / "templates"
INDEX_FILE = TEMPLATES_DIR / "index.html"


class TaskHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.serve_file(INDEX_FILE, "text/html; charset=utf-8")
            return

        if parsed.path == "/api/health":
            self.send_json({"status": "ok"})
            return

        if parsed.path.startswith("/api/results/"):
            task_id = parsed.path.rsplit("/", 1)[-1].lower()
            self.send_result(task_id)
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Not Found")

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/tasks/"):
            task_id = parsed.path.rsplit("/", 1)[-1].lower()
            self.run_task(task_id)
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Not Found")

    def serve_file(self, path: Path, content_type: str):
        if not path.exists():
            self.send_error(HTTPStatus.NOT_FOUND, "Not Found")
            return
        data = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_result(self, task_id: str):
        mapping = {
            "task2": ROOT / "Q2" / "test_results.txt",
            "task3": ROOT / "Q3" / "result.txt",
            "task4": ROOT / "Q4" / "result.txt",
        }
        path = mapping.get(task_id)
        if not path or not path.exists():
            self.send_json({"success": False, "message": "No result file yet."})
            return
        self.send_json({"success": True, "content": path.read_text(encoding="utf-8", errors="ignore")})

    def run_task(self, task_id: str):
        if task_id == "task2":
            self.start_background_task("Task 2", [sys.executable, "install_and_run.py"], ROOT / "Q2")
            self.send_json({"success": True, "message": "Task 2 started in background."})
            return

        if task_id == "task3":
            self.start_background_task("Task 3", [sys.executable, "generate_q3_results.py"], ROOT / "Q3")
            self.send_json({"success": True, "message": "Task 3 started in background."})
            return

        if task_id == "task4":
            self.start_background_task("Task 4", [sys.executable, "test_cv_pipeline.py"], ROOT / "Q4")
            self.send_json({"success": True, "message": "Task 4 started in background."})
            return

        self.send_json({"success": False, "message": "Unknown task"})

    def start_background_task(self, title: str, command, cwd: Path):
        def worker():
            try:
                subprocess.Popen(command, cwd=str(cwd), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                pass

        threading.Thread(target=worker, daemon=True).start()


def main():
    server = ThreadingHTTPServer(("127.0.0.1", 5000), TaskHandler)
    print("Local dashboard running at http://127.0.0.1:5000")
    server.serve_forever()


if __name__ == "__main__":
    main()
