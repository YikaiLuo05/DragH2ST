"""Dependency-light HTTP server; the frontend is served locally, without a CDN."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse
import cgi
import io
import json
import traceback
from .workspace import Workspace
from .messages import UserError, error_payload

STATIC = Path(__file__).parent / "static"


def serve(cfg):
    workspace = Workspace(cfg)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            if self.path.startswith("/api/jobs") or "/tile?" in self.path:
                return
            super().log_message(format, *args)

        def respond(self, data, content_type="application/json; charset=utf-8", status=200):
            if content_type.startswith("application/json"):
                data = json.dumps(data, ensure_ascii=False, allow_nan=False).encode()
            elif isinstance(data, str):
                data = data.encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "private, max-age=3600" if content_type.startswith("image/") else "no-cache")
            self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'")
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            try:
                url = urlparse(self.path)
                args = {key: values[0] for key, values in parse_qs(url.query).items()}
                path = url.path
                if path == "/":
                    return self.respond((STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
                if path in {"/app.js", "/i18n.js", "/style.css", "/translations.json"}:
                    content_type = "application/javascript" if path.endswith(".js") else "application/json" if path.endswith(".json") else "text/css"
                    data = (STATIC / path[1:]).read_bytes()
                    return self.respond(json.loads(data) if path.endswith(".json") else data, content_type)
                if path == "/favicon.ico":
                    return self.respond(b"", "image/x-icon", 204)
                if path == "/api/health":
                    return self.respond({"status": "ready"})
                if path == "/api/catalog":
                    return self.respond(workspace.catalog())
                if path == "/api/slide":
                    return self.respond(workspace.info(args["id"]))
                if path == "/api/preview":
                    return self.respond(workspace.preview(args["id"]), "image/jpeg")
                if path == "/api/tile":
                    return self.respond(workspace.tile(args["id"], int(args["level"]), int(args["x"]), int(args["y"])), "image/jpeg")
                if path == "/api/prediction":
                    result = workspace.result(args["id"])
                    return self.respond({"genes": result["genes"].tolist(), "spots": len(result["prediction"]), "source": "predicted"})
                if path == "/api/heatmap":
                    return self.respond(workspace.heatmap(args["id"], args["gene"]))
                if path == "/api/jobs":
                    if args["id"] not in workspace.jobs:
                        raise UserError("error.jobMissing", status=404)
                    return self.respond(dict(workspace.jobs[args["id"]]))
                if path == "/api/library":
                    return self.respond({"tissues": workspace.library.catalog(),
                        "prototypes": workspace.describe_prototypes(workspace.library.browse(args.get("tissue"), int(args.get("offset", 0))))})
                if path == "/api/thumbnail":
                    return self.respond(workspace.thumbnail(int(args["id"])), "image/jpeg")
                if path == "/api/download":
                    result = workspace.result(args["id"])
                    import csv
                    text = io.StringIO(); writer = csv.writer(text)
                    writer.writerow(["spot", "pixel_x", "pixel_y", *result["genes"]])
                    for i, row in enumerate(result["prediction"]):
                        writer.writerow([i, *result["pixels"][i], *row])
                    data = text.getvalue().encode("utf-8-sig")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/csv; charset=utf-8")
                    self.send_header("Content-Disposition", 'attachment; filename="dragh2st_expression.csv"')
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers(); self.wfile.write(data)
                    return
                self.respond(error_payload(UserError("error.resourceMissing")), status=404)
            except UserError as error:
                self.respond(error_payload(error), status=error.status)
            except KeyError as error:
                self.respond(error_payload(error), status=404)
            except (ValueError, FileNotFoundError) as error:
                self.respond(error_payload(error), status=400)
            except Exception:
                traceback.print_exc()
                self.respond(error_payload(UserError("error.serverFailure")), status=500)

        def do_POST(self):
            try:
                origin = self.headers.get("Origin")
                if origin and urlparse(origin).netloc != self.headers.get("Host"):
                    return self.respond(error_payload(UserError("error.origin")), status=403)
                path = urlparse(self.path).path
                length = int(self.headers.get("Content-Length", 0))
                if length <= 0 or length > 513 * 1024 * 1024:
                    raise UserError("error.requestSize")
                if path == "/api/upload":
                    form = cgi.FieldStorage(fp=self.rfile, headers=self.headers,
                                             environ={"REQUEST_METHOD": "POST", "CONTENT_TYPE": self.headers.get("Content-Type"), "CONTENT_LENGTH": str(length)})
                    if "image" not in form:
                        raise UserError("error.selectImage")
                    image = form["image"]
                    coordinates = form["coordinates"].file.read().decode("utf-8-sig") if "coordinates" in form and form["coordinates"].filename else None
                    return self.respond(workspace.upload(image.filename, image.file.read(), form.getfirst("dataset", "her2st"), coordinates))
                if length > 1024 * 1024:
                    raise UserError("error.requestTooLarge")
                body = json.loads(self.rfile.read(length))
                if path == "/api/predict":
                    return self.respond(workspace.submit("predict", body["id"]))
                if path == "/api/index":
                    return self.respond(workspace.submit("index", body["id"], body.get("tissue")))
                if path == "/api/region":
                    return self.respond(workspace.region(body["id"], body.get("selection", {})))
                self.respond(error_payload(UserError("error.resourceMissing")), status=404)
            except UserError as error:
                self.respond(error_payload(error), status=error.status)
            except (KeyError, ValueError, FileNotFoundError) as error:
                self.respond(error_payload(error), status=400)
            except Exception:
                traceback.print_exc()
                self.respond(error_payload(UserError("error.serverFailure")), status=500)

    server = ThreadingHTTPServer((cfg["web"]["host"], cfg["web"]["port"]), Handler)
    server.daemon_threads = True
    print(f"DragH2ST Prototype Bank: http://localhost:{cfg['web']['port']}", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        workspace.executor.shutdown(wait=True)
