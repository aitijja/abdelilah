"""
Simulation dynamique du MPU6050 : capteur en mouvement

1. Manœuvre de multirotor (oscillations d'attitude, lacet, déplacements) :
   effet des accélérations linéaires sur les 4 filtres de fusion
2. Vibrations des moteurs : effet du filtre passe-bas DLPF du capteur
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Ajouter le répertoire parent au path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.mpu6050_model import MPU6050Model, MPU6050Config
from src.models.trajectory import Signal, Trajectory
from src.fusion import QuaternionEKF, ComplementaryFilter, MahonyFilter, MadgwickFilter

FS = 250.0          # Hz, comme le sketch Arduino (1 kHz / (1 + SMPLRT_DIV), SMPLRT_DIV = 3)
DURATION = 30.0     # s
SETTLING = 2.0      # s exclues des statistiques (convergence initiale)
deg = np.deg2rad


def maneuver(linear_motion: bool = True, vibration: bool = False) -> Trajectory:
    """
    Trajectoire type multirotor

    Args:
        linear_motion: Ajouter des déplacements (accélérations jusqu'à ~0.2 g)
        vibration: Ajouter des vibrations moteur à 150 Hz (0.5 g, 0.02°)
    """
    roll = [(deg(15), 0.5, 0.0), (deg(3), 2.0, 1.0)]
    z = []
    if vibration:
        roll.append((deg(0.02), 150.0, 0.0))
        z.append((0.5 * 9.81 / (2 * np.pi * 150.0)**2, 150.0, 0.0))
    if linear_motion:
        x, y = [(0.5, 0.25, 0.0)], [(0.3, 0.3, 1.0)]
        z.append((0.05, 1.0, 0.0))
    else:
        x, y = [], []

    return Trajectory(
        roll=Signal(sines=roll),
        pitch=Signal(sines=[(deg(10), 0.3, 0.5)]),
        yaw=Signal(rate=deg(20)),
        x=Signal(sines=x),
        y=Signal(sines=y),
        z=Signal(sines=z)
    )


def run_filters(data):
    """
    Exécute les 4 filtres sur une simulation dynamique

    Returns:
        dict nom -> (euler (N, 3) [rad], taux d'utilisation de l'accéléromètre,
                     biais gyro max estimé [rad/s], EKF uniquement)
    """
    filters = {
        'EKF': QuaternionEKF(sample_rate=FS),
        'Complémentaire': ComplementaryFilter(alpha=0.98, sample_rate=FS),
        'Mahony': MahonyFilter(kp=2.0, ki=0.1, sample_rate=FS),
        'Madgwick': MadgwickFilter(beta=0.1, sample_rate=FS),
    }
    results = {}
    for name, filt in filters.items():
        euler = np.zeros((len(data.time), 3))
        accepted = 0
        bias_max = 0.0
        for k in range(len(data.time)):
            if name == 'EKF':
                filt.predict(data.gyro[k])
                accepted += filt.update(data.accel[k])
                bias_max = max(bias_max, np.abs(filt.gyro_bias).max())
            else:
                accepted += filt.update(data.gyro[k], data.accel[k])
            euler[k] = filt.euler_angles
        results[name] = (euler, accepted / len(data.time), bias_max)
    return results


def rmse_deg(data, euler):
    """RMSE roll et pitch [°] après la phase de convergence"""
    sel = data.time >= SETTLING
    err = euler[sel, :2] - data.euler_true[sel, :2]
    return np.rad2deg(np.sqrt(np.mean(err**2, axis=0)))


def simulate_maneuver():
    """
    Compare les filtres avec et sans accélérations linéaires
    """
    print("=== Manœuvre de multirotor (DLPF 44 Hz, 250 Hz) ===\n")

    scenarios = {}
    for label, linear in [("Rotation seule", False), ("Rotation + déplacements", True)]:
        np.random.seed(0)
        sensor = MPU6050Model(MPU6050Config(accel_range=4, gyro_range=500,
                                            sample_rate=FS, dlpf_cfg=3))
        data = sensor.simulate_dynamic_measurement(maneuver(linear_motion=linear), DURATION)
        results = run_filters(data)
        scenarios[label] = (data, results)

        lin = np.linalg.norm(data.accel_true, axis=1) / 9.80665
        print(f"--- {label} : ||f||/g dans [{lin.min():.2f}, {lin.max():.2f}] ---")
        print(f"{'Filtre':<16}{'Roll RMSE':>11}{'Pitch RMSE':>12}{'Accel utilisé':>15}")
        for name, (euler, used, _) in results.items():
            r, p = rmse_deg(data, euler)
            print(f"{name:<16}{r:>10.2f}°{p:>11.2f}°{100 * used:>14.0f}%")
        print(f"Biais gyro max estimé par l'EKF : {np.rad2deg(results['EKF'][2]):.1f} °/s (vrai ≈ 0)\n")

    # Visualisation du scénario avec déplacements
    data, results = scenarios["Rotation + déplacements"]
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    for i, angle in enumerate(['Roll', 'Pitch']):
        axes[i].plot(data.time, np.rad2deg(data.euler_true[:, i]), 'k--', linewidth=2, label='Vrai')
        for name, (euler, _, _) in results.items():
            axes[i].plot(data.time, np.rad2deg(euler[:, i]), label=name, alpha=0.8)
        axes[i].set_ylabel(f'{angle} [°]')
        axes[i].legend(loc='upper right')
        axes[i].grid(True, alpha=0.3)
    axes[0].set_title('Manœuvre avec accélérations linéaires')

    axes[2].plot(data.time, np.linalg.norm(data.accel, axis=1) / 9.80665, label='||a_m|| / g')
    axes[2].axhline(0.8, color='r', linestyle=':', label='Seuils de détection de mouvement')
    axes[2].axhline(1.2, color='r', linestyle=':')
    axes[2].set_xlabel('Temps [s]')
    axes[2].set_ylabel('||a_m|| / g')
    axes[2].legend(loc='upper right')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('dynamic_maneuver.png', dpi=150)
    print("Graphique sauvegardé: dynamic_maneuver.png")


def simulate_vibrations():
    """
    Vibrations moteur à 150 Hz : DLPF_CFG = 0 (260 Hz) contre DLPF_CFG = 3 (44 Hz)
    """
    print("\n\n=== Vibrations moteur à 150 Hz (rotation seule, 250 Hz) ===\n")
    print(f"{'DLPF':<14}{'σ vibration':>12}{'Roll RMSE (Mahony)':>21}{'Accel utilisé':>15}")

    # Force spécifique sans vibration : ce que le filtre de fusion doit voir
    reference = maneuver(linear_motion=False, vibration=False)

    traces = {}
    for cfg, label in [(0, "260 Hz (0)"), (3, "44 Hz (3)")]:
        np.random.seed(0)
        sensor = MPU6050Model(MPU6050Config(accel_range=4, gyro_range=500,
                                            sample_rate=FS, dlpf_cfg=cfg))
        data = sensor.simulate_dynamic_measurement(
            maneuver(linear_motion=False, vibration=True), DURATION)

        mahony = MahonyFilter(kp=2.0, ki=0.1, sample_rate=FS)
        euler = np.zeros((len(data.time), 3))
        accepted = 0
        for k in range(len(data.time)):
            accepted += mahony.update(data.gyro[k], data.accel[k])
            euler[k] = mahony.euler_angles

        residual = data.accel - reference.specific_force(data.time)
        sigma = np.std(residual, axis=0).max() / 9.80665
        r, _ = rmse_deg(data, euler)
        print(f"{label:<14}{sigma:>11.3f}g{r:>20.2f}°{100 * accepted / len(data.time):>14.0f}%")
        traces[label] = data

    fig, ax = plt.subplots(figsize=(12, 4))
    for label, data in traces.items():
        sel = data.time < 0.5
        ax.plot(data.time[sel], data.accel[sel, 2], label=f'a_z, DLPF {label}')
    ax.set_xlabel('Temps [s]')
    ax.set_ylabel('a_z [m/s²]')
    ax.set_title('Repliement des vibrations à 150 Hz échantillonnées à 250 Hz')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('dynamic_vibrations.png', dpi=150)
    print("\nGraphique sauvegardé: dynamic_vibrations.png")


if __name__ == "__main__":
    simulate_maneuver()
    simulate_vibrations()

    print("\n=== Simulation terminée ===")
    plt.show()
