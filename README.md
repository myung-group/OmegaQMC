# ΩQMC

Variational Monte Carlo with point group correlated sampling (for force estimators)

Auxiliary field quantum Monte Carlo with cavity coupling

## Installation
Install directly from the remote repository:
```bash
pip install git+ssh://git@github.com:myung-group/OmegaQMC.git
```
Or install from the local repository (cloned with `git`):
```bash
pip install .
```
Or in (local) development mode:
```bash
pip install -e .
```

## Usage
```bash
python test/test_H2.py
```

## Documentation

The API reference and usage guide are built with [Sphinx](https://www.sphinx-doc.org/).
Install the build dependencies, then run `make` from the `doc-sphinx/` directory:

```bash
cd doc-sphinx

# HTML (browse at doc-sphinx/_build/html/index.html)
make html

# PDF via LaTeX (output generated in doc-sphinx/_build/latex/)
make latexpdf
```

Required packages: `sphinx`, `sphinx-rtd-theme` (or another theme configured in `conf.py`).

## License
MIT License
