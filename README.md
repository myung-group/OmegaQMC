# VMC with MLSW

Variational Monte Carlo with machine-learned space warping (for force estimators)

## Update list
- Added restart capability
- Added N H2O reflection (N = 1, 2, ...)
- Added torque computation
- Added error computation of energy, gradients, torques.
- Added aug-cc-pCVQZ basis set informations

## Installation
Install directly from the repository:
```bash
pip install .
```
Or in development mode:
```bash
pip install -e .
```

## Usage
```bash
python test/test_H2.py
```

## License
MIT License
