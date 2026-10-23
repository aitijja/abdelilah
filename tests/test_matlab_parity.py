"""
Vérifie que matlab/QuaternionEKF.m donne les mêmes estimations que la version Python

Nécessite GNU Octave dans le PATH, sinon le test est ignoré.
"""

import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from src.fusion import QuaternionEKF

OCTAVE = shutil.which('octave-cli') or shutil.which('octave')


@pytest.mark.skipif(OCTAVE is None, reason="GNU Octave non installé")
def test_matlab_ekf_matches_python(tmp_path):
    rng = np.random.default_rng(3)
    fs, n = 100.0, 500
    pitch = np.linspace(0, np.deg2rad(45), n)
    rate = np.append(np.diff(pitch) * fs, 0.0)
    accel = np.column_stack([-9.81 * np.sin(pitch), np.zeros(n), 9.81 * np.cos(pitch)])
    accel += 0.04 * rng.standard_normal((n, 3))
    gyro = np.column_stack([np.zeros(n), rate, np.zeros(n)]) + np.deg2rad([0.5, -0.3, 0.2])
    gyro += np.deg2rad(0.05) * rng.standard_normal((n, 3))

    data = tmp_path / 'imu.csv'
    out = tmp_path / 'state.csv'
    np.savetxt(data, np.hstack([gyro, accel]), delimiter=',')

    script = (
        f"addpath('{os.path.join(ROOT, 'matlab')}');"
        f"d = csvread('{data}'); ekf = QuaternionEKF('SampleRate', {fs});"
        "X = zeros(rows(d), 7);"
        "for k = 1:rows(d), ekf.predict(d(k,1:3)'); ekf.update(d(k,4:6)'); X(k,:) = ekf.x'; end;"
        f"dlmwrite('{out}', X, 'precision', 17);"
    )
    subprocess.run([OCTAVE, '--no-gui', '--no-window-system', '--quiet', '--eval', script],
                   check=True, capture_output=True, timeout=300)
    x_matlab = np.loadtxt(out, delimiter=',')

    ekf = QuaternionEKF(sample_rate=fs)
    x_python = np.zeros((n, 7))
    for k in range(n):
        ekf.predict(gyro[k])
        ekf.update(accel[k])
        x_python[k] = ekf.x

    np.testing.assert_allclose(x_matlab, x_python, atol=1e-9)
