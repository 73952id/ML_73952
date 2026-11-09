"""Execute and save notebook outputs without an additional notebook runtime."""
import contextlib
import io
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    path = root / 'autodiff.ipynb'
    notebook = json.loads(path.read_text(encoding='utf-8'))
    namespace = {'__name__': '__main__'}
    import os
    os.chdir(root)
    count = 0
    for cell in notebook['cells']:
        if cell['cell_type'] != 'code':
            continue
        count += 1
        print(f'Executing code cell {count}/6...', flush=True)
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            exec(compile(''.join(cell['source']), f'<cell {count}>', 'exec'), namespace)
        cell['execution_count'] = count
        cell['outputs'] = [{'output_type':'stream', 'name':'stdout',
                            'text':output.getvalue().splitlines(True)}]
        path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding='utf-8')
        print(output.getvalue(), flush=True)
    print('All notebook cells completed.', flush=True)


if __name__ == '__main__':
    main()
