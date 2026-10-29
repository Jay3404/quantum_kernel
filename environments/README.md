# Environments

## Portable Aggregate Replay

Verified environment: Python 3.14.4, NumPy 2.4.3, pandas 3.0.1 and SciPy 1.17.1.
The root `requirements.txt` pins these three analysis libraries. No Qiskit,
GPU package or original dataset is needed for this replay.

## Original Scientific Fitting

The fixed server environment used Python 3.10.12, scikit-learn 1.7.2,
XGBoost 3.0.5, Qiskit 2.5.2, Qiskit Aer 0.17.2, NumPy 2.2.6, SciPy 1.15.3
and CuPy 13.6.0. `server-requirements.lock.txt` is the unchanged archived pip
lock; the GPU CuPy environment was maintained separately.

CPU Aer generated exact double-precision 7/8-qubit statevectors for PCA7 and
9/10-qubit statevectors for PCA9. GPU devices performed Gram algebra. The
server had 16 physical cores, 32 logical CPUs, approximately 188.1 GiB host RAM
and an A100 80GB exposed through two MIG devices of 39.25 GiB each.

Calibration and the inherited-setting Risk-only fitting used local Python 3.14.4
and scikit-learn 1.8.0. Independently tuned Risk-only used the fixed server.
The environments and selection algorithms are therefore documented separately.
