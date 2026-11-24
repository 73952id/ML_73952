"""A small local browser interface; audio is processed on this computer."""
import argparse
import json
import tempfile
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .audio import ROOT
from .predict import Predictor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    predictor = Predictor()
    temporary = ROOT / '.tmp'
    temporary.mkdir(exist_ok=True)

    class Handler(BaseHTTPRequestHandler):
        def respond(self, status, payload, content_type='application/json; charset=utf-8'):
            if not isinstance(payload, bytes):
                payload = json.dumps(payload).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            if self.path == '/':
                self.respond(200, (ROOT / 'src' / 'index.html').read_bytes(), 'text/html; charset=utf-8')
            elif self.path == '/config':
                self.respond(200, predictor.config)
            else:
                self.respond(404, {'error': 'Not found'})

        def do_POST(self):
            parsed = urlparse(self.path)
            if parsed.path != '/predict':
                self.respond(404, {'error': 'Not found'})
                return
            # Browsers on other sites cannot submit raw audio to the local app.
            if self.headers.get('Sec-Fetch-Site') == 'cross-site':
                self.respond(403, {'error': 'Use the local application page.'})
                return
            try:
                length = int(self.headers.get('Content-Length', 0))
                if not 0 < length <= 30_000_000:
                    self.respond(413, {'error': 'Choose an audio file smaller than 30 MB.'})
                    return
                filename = parse_qs(parsed.query).get('filename', ['audio.mp3'])[0]
                suffix = Path(filename).suffix.lower()
                if suffix not in {'.mp3', '.wav', '.flac', '.ogg'}:
                    self.respond(400, {'error': 'Supported formats: MP3, WAV, FLAC, OGG.'})
                    return
                with tempfile.TemporaryDirectory(dir=temporary) as directory:
                    path = Path(directory) / ('audio' + suffix)
                    path.write_bytes(self.rfile.read(length))
                    result = predictor.predict(path)
                self.respond(200, {'filename': Path(filename).name, **result})
            except Exception as error:
                self.respond(400, {'error': f'Could not process audio: {error}'})

    server = HTTPServer(('127.0.0.1', args.port), Handler)
    print(f'Open http://127.0.0.1:{args.port} in your browser. Ctrl+C stops the app.', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
