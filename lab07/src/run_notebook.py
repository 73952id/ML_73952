"""Execute the Lab 07 notebook and retain its real outputs."""
import contextlib
import base64
import io
import json
import os
import sys
from pathlib import Path


class Tee(io.StringIO):
    def __init__(self, terminal):
        super().__init__()
        self.terminal = terminal

    def write(self, text):
        self.terminal.write(text)
        self.terminal.flush()
        return super().write(text)

    def flush(self):
        self.terminal.flush()


def main():
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    path = root/'compression.ipynb'
    notebook = json.loads(path.read_text(encoding='utf-8'))
    namespace = {'__name__':'__main__'}
    index = 0
    for cell in notebook['cells']:
        if cell['cell_type'] != 'code':
            continue
        index += 1
        print(f'Executing section {index}/5', flush=True)
        output = Tee(sys.stdout)
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            exec(compile(''.join(cell['source']),f'<section {index}>','exec'),namespace)
        cell['execution_count'] = index
        cell['outputs'] = [{'output_type':'stream','name':'stdout','text':output.getvalue().splitlines(True)}]
        image_name = {2:'sparsity_curves.png', 3:'storage_latency.png', 5:'storage_latency.png'}.get(index)
        if image_name:
            encoded = base64.b64encode((root/'results'/image_name).read_bytes()).decode('ascii')
            cell['outputs'].append({'output_type':'display_data','data':{'image/png':encoded},'metadata':{}})
        path.write_text(json.dumps(notebook,indent=1,ensure_ascii=False),encoding='utf-8')
    from lab07.src.report import build_report
    build_report(root)
    print('All five sections completed; report updated.', flush=True)


if __name__ == '__main__':
    main()
