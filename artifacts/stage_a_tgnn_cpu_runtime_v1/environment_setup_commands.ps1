$ErrorActionPreference = 'Stop'

# Dedicated Stage A TGNN CPU runtime; this path is intentionally outside the repository.
$RuntimeRoot = 'C:\Users\johns\satnet-stage-a-tgnn-runtime-v1'
$Venv = Join-Path $RuntimeRoot '.venv'
$Python = Join-Path $Venv 'Scripts\python.exe'
$Wheelhouse = Join-Path $RuntimeRoot 'wheelhouse'

py -3.11 -m venv $Venv

# Official Windows AMD64 CPython 3.11 CPU wheel; no CUDA index is used.
& $Python -m pip install --no-cache-dir --only-binary=:all: `
  torch==2.12.1 --index-url 'https://download.pytorch.org/whl/cpu'

# Frozen pure/PyPI packages.
& $Python -m pip install --no-cache-dir --only-binary=:all: `
  numpy==1.26.4 pandas==2.3.3 scipy==1.17.1 `
  scikit-learn==1.9.0 joblib==1.5.3 torch-geometric==2.8.0 `
  --index-url 'https://pypi.org/simple'

# PyG CPU native wheels resolved from the official PyG compatibility index.
# The index resolves the pt212cpu wheels for Windows AMD64 / CPython 3.11.
& $Python -m pip install --no-cache-dir --only-binary=:all: `
  torch-sparse==0.6.18 torch-scatter==2.1.2 `
  --no-index -f 'https://data.pyg.org/whl/torch-2.12.1+cpu.html'

& $Python -m pip install --no-cache-dir --only-binary=:all: `
  torch-geometric-temporal==0.56.2 --index-url 'https://pypi.org/simple'

# Retain the resolved wheels for evidence and reproducibility; this does not alter the repository.
New-Item -ItemType Directory -Force -Path $Wheelhouse | Out-Null
& $Python -m pip download --no-deps --only-binary=:all: `
  torch==2.12.1 --index-url 'https://download.pytorch.org/whl/cpu' --dest $Wheelhouse
& $Python -m pip download --no-deps --only-binary=:all: `
  numpy==1.26.4 pandas==2.3.3 scipy==1.17.1 scikit-learn==1.9.0 `
  joblib==1.5.3 torch-geometric==2.8.0 torch-geometric-temporal==0.56.2 `
  decorator==4.4.2 cython==3.2.9 --index-url 'https://pypi.org/simple' --dest $Wheelhouse
& $Python -m pip download --no-deps --only-binary=:all: `
  torch-sparse==0.6.18 torch-scatter==2.1.2 `
  --no-index -f 'https://data.pyg.org/whl/torch-2.12.1+cpu.html' --dest $Wheelhouse

# Validation was performed in fresh Python processes with import-only, constructor-only,
# and synthetic-only checks. No project dataset path is an input to this setup sequence.
