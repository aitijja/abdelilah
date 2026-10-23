"""
Tests de la partie dynamique : cinématique des trajectoires et chaîne de mesure
(DLPF, décimation, saturation, quantification)
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import MPU6050Model, MPU6050Config, Signal, Trajectory
from src.fusion import MahonyFilter
from src.utils.constants import GRAVITY, GYRO_G_SENSITIVITY, GYRO_SCALE_FACTOR, ACCEL_SCALE_FACTOR


def omega_matrix(w):
    wx, wy, wz = w
    return np.array([
        [0,   -wx,  -wy,  -wz],
        [wx,   0,    wz,  -wy],
        [wy,  -wz,   0,    wx],
        [wz,   wy,  -wx,   0]
    ])


def quat_to_matrix(q):
    q0, q1, q2, q3 = q
    return np.array([
        [1 - 2*(q2**2 + q3**2), 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2)],
        [2*(q1*q2 + q0*q3), 1 - 2*(q1**2 + q3**2), 2*(q2*q3 - q0*q1)],
        [2*(q1*q3 - q0*q2), 2*(q2*q3 + q0*q1), 1 - 2*(q1**2 + q2**2)]
    ])


GENERIC = Trajectory(
    roll=Signal(0.1, 0.0, [(0.5, 0.7, 0.3)]),
    pitch=Signal(-0.2, 0.05, [(0.4, 0.3, 1.0)]),
    yaw=Signal(0.3, 0.8, [(1.0, 0.2, 0.0)]),
    x=Signal(1.0, 0.5, [(1.0, 0.5, 0.0)]),
    y=Signal(sines=[(0.3, 1.2, 0.2)]),
    z=Signal(2.0, 0.0, [(0.3, 1.0, 0.5)])
)


def test_signal_derivatives():
    """Les dérivées analytiques de Signal correspondent aux différences finies"""
    s = Signal(0.5, -0.2, [(1.3, 0.7, 0.4), (0.2, 3.0, -1.0)])
    t = np.linspace(0, 5, 101)
    h = 1e-5
    np.testing.assert_allclose(s.derivative(t), (s.value(t + h) - s.value(t - h)) / (2 * h), atol=1e-7)
    np.testing.assert_allclose(s.second_derivative(t),
                               (s.derivative(t + h) - s.derivative(t - h)) / (2 * h), atol=1e-5)


def test_static_trajectory_is_gravity_only():
    """Capteur immobile et à plat : f = [0, 0, g], ω = 0"""
    t = np.linspace(0, 1, 11)
    traj = Trajectory()
    np.testing.assert_allclose(traj.specific_force(t), np.tile([0, 0, GRAVITY], (11, 1)))
    np.testing.assert_allclose(traj.angular_velocity(t), 0.0)
    np.testing.assert_allclose(traj.quaternion(t), np.tile([1, 0, 0, 0], (11, 1)))


def test_angular_velocity_integrates_to_attitude():
    """Intégrer dq/dt = ½·Ω(ω_S)·q redonne le quaternion de la trajectoire"""
    dt, duration = 1e-3, 3.0
    q = GENERIC.quaternion(0.0)
    for k in range(int(duration / dt)):
        t = k * dt
        def f(tt, qq):
            return 0.5 * omega_matrix(GENERIC.angular_velocity(tt)) @ qq
        k1 = f(t, q)
        k2 = f(t + dt / 2, q + dt / 2 * k1)
        k3 = f(t + dt / 2, q + dt / 2 * k2)
        k4 = f(t + dt, q + dt * k3)
        q = q + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        q /= np.linalg.norm(q)

    q_true = GENERIC.quaternion(duration)
    assert min(np.abs(q - q_true).max(), np.abs(q + q_true).max()) < 1e-9


def test_specific_force_matches_quaternion_convention():
    """f_S = R(q)ᵀ·(p̈ + g·e_z) avec la matrice des filtres de fusion"""
    for t in [0.0, 0.37, 1.9]:
        R = quat_to_matrix(GENERIC.quaternion(t))
        np.testing.assert_allclose(GENERIC.rotation_matrix(t), R, atol=1e-12)
        expected = R.T @ (GENERIC.acceleration(t) + [0, 0, GRAVITY])
        np.testing.assert_allclose(GENERIC.specific_force(t), expected, atol=1e-12)


def test_ideal_chain_reproduces_truth():
    """Sans bruit ni quantification, une entrée constante traverse la chaîne intacte"""
    sensor = MPU6050Model(MPU6050Config(sample_rate=250))
    tilt = Trajectory(roll=Signal(np.deg2rad(20)), pitch=Signal(np.deg2rad(-10)))
    data = sensor.simulate_dynamic_measurement(tilt, 1.0, add_noise=False, quantize=False)

    assert data.time.shape == (250,)
    np.testing.assert_allclose(np.diff(data.time), 1 / 250)
    np.testing.assert_allclose(data.accel, data.accel_true, atol=1e-9)
    # Le gyroscope ne voit que sa g-sensibilité
    np.testing.assert_allclose(data.gyro, GYRO_G_SENSITIVITY * data.accel_true / GRAVITY, atol=1e-12)


@pytest.mark.parametrize('cfg', [1, 3, 5])
def test_dlpf_cutoff_is_minus_3db(cfg):
    """Le modèle du DLPF atténue de -3 dB à la fréquence de coupure du gyroscope"""
    from src.utils.constants import DLPF_CUTOFF_FREQ_GYRO
    fc = DLPF_CUTOFF_FREQ_GYRO[cfg]
    sensor = MPU6050Model(MPU6050Config(sample_rate=1000, gyro_range=2000, dlpf_cfg=cfg))
    traj = Trajectory(roll=Signal(sines=[(np.deg2rad(2), fc, 0.0)]))
    data = sensor.simulate_dynamic_measurement(traj, 3.0, add_noise=False, quantize=False)

    steady = data.time > 1.0
    gain = np.std(data.gyro[steady, 0]) / np.std(data.gyro_true[steady, 0])
    assert gain == pytest.approx(1 / np.sqrt(2), rel=0.02)


def test_dlpf_rejects_vibrations():
    """Une vibration à 150 Hz traverse DLPF_CFG = 0 mais est atténuée par DLPF_CFG = 3"""
    vib = 0.5 * GRAVITY / (2 * np.pi * 150) ** 2
    traj = Trajectory(z=Signal(sines=[(vib, 150.0, 0.0)]))
    residual = {}
    for cfg in (0, 3):
        sensor = MPU6050Model(MPU6050Config(accel_range=4, sample_rate=250, dlpf_cfg=cfg))
        data = sensor.simulate_dynamic_measurement(traj, 2.0, add_noise=False, quantize=False)
        residual[cfg] = np.std(data.accel[data.time > 0.5, 2]) / GRAVITY
    assert residual[0] > 0.2
    assert residual[3] < 0.05


def test_saturation_and_quantization():
    """Les mesures saturent à la pleine échelle et tombent sur le LSB"""
    sensor = MPU6050Model(MPU6050Config(accel_range=2, gyro_range=250, sample_rate=250))
    spin = Trajectory(yaw=Signal(rate=np.deg2rad(400)),
                      x=Signal(sines=[(1.0, 2.0, 0.0)]))  # 400 °/s et jusqu'à 16 m/s²
    data = sensor.simulate_dynamic_measurement(spin, 1.0, add_noise=False)

    gyro_lsb = np.rad2deg(data.gyro) * GYRO_SCALE_FACTOR[250]
    accel_lsb = data.accel / GRAVITY * ACCEL_SCALE_FACTOR[2]
    assert gyro_lsb.max() == pytest.approx(32767)
    assert np.abs(accel_lsb).max() == pytest.approx(32768)
    np.testing.assert_allclose(gyro_lsb, np.round(gyro_lsb), atol=1e-6)
    np.testing.assert_allclose(accel_lsb, np.round(accel_lsb), atol=1e-6)


def test_invalid_sample_rate():
    """La cadence de sortie doit valoir 1 kHz / (1 + SMPLRT_DIV)"""
    sensor = MPU6050Model(MPU6050Config(sample_rate=300))
    with pytest.raises(ValueError):
        sensor.simulate_dynamic_measurement(Trajectory(), 1.0)


def test_noise_level_after_dlpf():
    """Au repos, le bruit après DLPF reste de l'ordre de la densité × √(2·bande)"""
    np.random.seed(0)
    sensor = MPU6050Model(MPU6050Config(sample_rate=250, dlpf_cfg=3))
    data = sensor.simulate_dynamic_measurement(Trajectory(), 20.0, quantize=False)
    sigma = np.std(data.accel - data.accel_true, axis=0)
    # Bande équivalente de bruit d'un Butterworth d'ordre 2 : ≈ 1.11·f_c
    expected = 3.924e-3 * np.sqrt(2 * 1.11 * 44)
    np.testing.assert_allclose(sigma, expected, rtol=0.15)


def test_mahony_tracks_dynamic_rotation():
    """En rotation pure (sans accélération linéaire), Mahony suit l'attitude à mieux que 0.5°"""
    np.random.seed(1)
    sensor = MPU6050Model(MPU6050Config(accel_range=4, gyro_range=500, sample_rate=250, dlpf_cfg=3))
    traj = Trajectory(roll=Signal(sines=[(np.deg2rad(15), 0.5, 0.0)]),
                      pitch=Signal(sines=[(np.deg2rad(10), 0.3, 0.5)]),
                      yaw=Signal(rate=np.deg2rad(20)))
    data = sensor.simulate_dynamic_measurement(traj, 20.0)

    mahony = MahonyFilter(kp=2.0, ki=0.1, sample_rate=250)
    euler = np.zeros((len(data.time), 3))
    for k in range(len(data.time)):
        mahony.update(data.gyro[k], data.accel[k])
        euler[k] = mahony.euler_angles

    steady = data.time > 2.0
    rmse = np.rad2deg(np.sqrt(np.mean((euler[steady, :2] - data.euler_true[steady, :2]) ** 2, axis=0)))
    assert rmse.max() < 0.5
