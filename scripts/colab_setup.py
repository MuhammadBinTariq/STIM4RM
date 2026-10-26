from pathlib import Path
import subprocess, sys

ROOT = Path('/content/rewardmaxxing-v2')

def checked(command):
    """Show subprocess output in the notebook, including the actual failure."""
    command = list(map(str, command))
    print('Running:', ' '.join(command), flush=True)
    with subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
        for line in process.stdout:
            print(line, end='', flush=True)
        status = process.wait()
    if status:
        raise subprocess.CalledProcessError(status, command)

# uv creates/seeds the environment without relying on Colab's system ensurepip.
checked([sys.executable, '-m', 'pip', 'install', 'uv==0.9.5'])
UV = [sys.executable, '-m', 'uv']
VENV = Path('/content/rewardmaxxing-venv-py312')
PY = str(VENV / 'bin/python')
if not Path(PY).exists():
    if VENV.exists():
        raise RuntimeError('Incomplete py312 environment: choose a fresh VENV path and rerun this cell.')
    checked(UV + ['venv', '--python', '3.12', '--seed', str(VENV)])

checked([PY, '-c', 'import sys, pip; assert sys.version_info[:2] == (3, 12), sys.version; print(sys.version)'])
checked([PY, '-m', 'pip', 'install', 'torch==2.6.0',
         '--index-url', 'https://download.pytorch.org/whl/cu124'])
checked([PY, '-m', 'pip', 'install', '-r', ROOT / 'requirements-v2.txt'])
checked([PY, '-m', 'pip', 'check'])
checked([PY, '-u', 'tests/smoke_test.py'])
checked([PY, '-c', "import torch; print('Torch:', torch.__version__, 'CUDA:', torch.version.cuda); assert torch.cuda.is_available(), 'Select a GPU runtime'; print('GPU:', torch.cuda.get_device_name(0))"])
