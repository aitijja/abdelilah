"""
Tests de convergence des algorithmes de fusion sur données simulées
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.mpu6050_model import MPU6050Model, MPU6050Config
from src.fusion import QuaternionEKF, ComplementaryFilter, MahonyFilter, MadgwickFilter

FS = 100.0
G = 9.81


def make_filters():
    return {
        'ekf': QuaternionEKF(sample_rate=FS),
        'complementary': ComplementaryFilter(alpha=0.98, sample_rate=FS),
        'mahony': MahonyFilter(kp=2.0, ki=0.1, sample_rate=FS),
        'madgwick': MadgwickFilter(beta=0.1, sample_rate=FS),
    }


def step(name, filt, gyro, accel):
    if name == 'ekf':
        filt.predict(gyro)
        filt.update(accel)
    else:
        filt.update(gyro, accel)


def gravity_in_sensor(roll, pitch):
    """Gravité mesurée (repère capteur) pour une attitude roll/pitch, convention ZYX"""
    return G * np.array([
        -np.sin(pitch),
        np.sin(roll) * np.cos(pitch),
        np.cos(roll) * np.cos(pitch)
    ])


@pytest.mark.parametrize('name', ['ekf', 'complementary', 'mahony', 'madgwick'])
@pytest.mark.parametrize('roll_deg, pitch_deg', [(30, 0), (0, -40), (20, 25)])
def test_static_convergence(name, roll_deg, pitch_deg):
    """Partant de l'identité, chaque filtre converge vers l'inclinaison réelle"""
    np.random.seed(0)
    sensor = MPU6050Model(MPU6050Config(sample_rate=FS))
    filt = make_filters()[name]

    roll, pitch = np.deg2rad(roll_deg), np.deg2rad(pitch_deg)
    accel_true = gravity_in_sensor(roll, pitch)

    for _ in range(int(20 * FS)):
        accel = sensor.measure_accel(accel_true, add_noise=True)
        gyro = sensor.measure_gyro(np.zeros(3), accel=accel_true, add_noise=True)
        step(name, filt, gyro, accel)

    roll_est, pitch_est, _ = filt.euler_angles
    assert np.rad2deg(abs(roll_est - roll)) < 1.0
    assert np.rad2deg(abs(pitch_est - pitch)) < 1.0


@pytest.mark.parametrize('name', ['ekf', 'complementary', 'mahony', 'madgwick'])
def test_pitch_ramp_tracking(name):
    """Suivi d'une rotation de 45° en pitch (scénario de l'exemple)"""
    np.random.seed(1)
    sensor = MPU6050Model(MPU6050Config(sample_rate=FS))
    filt = make_filters()[name]

    n = int(5 * FS)
    pitch_true = np.linspace(0, np.deg2rad(45), n)
    errors = []
    for i in range(n):
        rate = (pitch_true[i+1] - pitch_true[i]) * FS if i < n - 1 else 0.0
        accel_true = gravity_in_sensor(0.0, pitch_true[i])
        accel = sensor.measure_accel(accel_true, add_noise=True)
        gyro = sensor.measure_gyro(np.array([0.0, rate, 0.0]), accel=accel_true, add_noise=True)
        step(name, filt, gyro, accel)
        errors.append(filt.euler_angles[1] - pitch_true[i])

    rmse = np.rad2deg(np.sqrt(np.mean(np.square(errors))))
    assert rmse < 1.0


def test_complementary_preserves_yaw():
    """
    La correction accéléromètre ne ramène pas le lacet vers 0

    Une correction d'inclinaison (axe horizontal) déplace légèrement le lacet
    d'Euler ZYX par effet géométrique, mais ne doit pas l'écraser comme le
    ferait un SLERP vers un quaternion absolu calculé depuis l'accéléromètre.
    """
    yaw = np.deg2rad(60)
    q0 = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
    filt = ComplementaryFilter(alpha=0.9, sample_rate=FS, initial_orientation=q0)

    accel = gravity_in_sensor(np.deg2rad(10), np.deg2rad(-15))
    for _ in range(int(10 * FS)):
        filt.update(np.zeros(3), accel)

    roll_est, pitch_est, yaw_est = filt.euler_angles
    assert np.rad2deg(abs(roll_est - np.deg2rad(10))) < 0.1
    assert np.rad2deg(abs(pitch_est - np.deg2rad(-15))) < 0.1
    assert np.rad2deg(abs(yaw_est - yaw)) < 3.0


def test_madgwick_adaptive_gain():
    """Le facteur adaptatif vaut 1 au repos et 0 aux seuils de mouvement"""
    filt = MadgwickFilter(adaptive_gain=True, gravity=G)
    assert filt._adaptive_factor(G) == pytest.approx(1.0)
    assert filt._adaptive_factor(1.2 * G) == pytest.approx(0.0)
    assert filt._adaptive_factor(0.8 * G) == pytest.approx(0.0, abs=1e-12)
    assert 0.0 < filt._adaptive_factor(1.1 * G) < 1.0


def test_ekf_estimates_gyro_bias():
    """L'EKF estime le biais du gyroscope sur les axes observables (x, y)"""
    np.random.seed(2)
    sensor = MPU6050Model(MPU6050Config(sample_rate=FS))
    ekf = QuaternionEKF(sample_rate=FS)
    bias = np.deg2rad([0.5, -0.3, 0.0])
    accel_true = gravity_in_sensor(0.0, 0.0)

    for _ in range(int(60 * FS)):
        accel = sensor.measure_accel(accel_true, add_noise=True)
        gyro = sensor.measure_gyro(np.zeros(3), accel=accel_true, add_noise=True) + bias
        ekf.predict(gyro)
        ekf.update(accel)

    est = ekf.gyro_bias
    assert np.rad2deg(np.abs(est[:2] - bias[:2])).max() < 0.1
