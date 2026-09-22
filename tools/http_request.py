# Archive #51/#52/#68/#129: isolated HTTP client so the supervisor enforces a wall deadline.
"""Internal localhost probe client; its parent owns the deadline and process."""
import json
from pathlib import Path
import sys
import time

from .phase import atomic_json
from .service_probe import _http


def main():
    request_path, response_path = map(Path, sys.argv[1:3])
    started = time.monotonic()
    try:
        request = json.loads(request_path.read_text())
        atomic_json(response_path.with_name("client-state.json"), {"state": "waiting_for_http_response",
                    "wall_time": time.time(), "prompt_tokens": len(request["payload"]["prompt"])})
        body = _http(request["url"], payload=request["payload"], timeout=request["timeout"])
        result = {"status": "passed", "body": json.loads(body)}
        code = 0
    except Exception as error:
        result = {"status": "failed", "error": f"{type(error).__name__}: {error}"}
        print(result["error"], file=sys.stderr, flush=True)
        code = 1
    atomic_json(response_path.with_name("client-state.json"), {"state": "response_received" if code == 0 else "http_error",
                "wall_time": time.time(), "elapsed_seconds": time.monotonic()-started})
    atomic_json(response_path, result)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
