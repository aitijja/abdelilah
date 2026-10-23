"""
Exemple de simulation basique du MPU6050 et fusion de capteurs
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Ajouter le répertoire parent au path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.mpu6050_model import MPU6050Model, MPU6050Config, SensorCalibration
from src.fusion.kalman_filter import QuaternionEKF
from src.fusion.complementary_filter import ComplementaryFilter
from src.fusion.mahony_filter import MahonyFilter
from src.fusion.madgwick_filter import MadgwickFilter


def simulate_static_scenario():
    """
    Simule un scénario statique avec le capteur incliné
    """
    print("=== Simulation statique du MPU6050 ===\n")

    # Configuration du capteur
    config = MPU6050Config(
        accel_range=2,
        gyro_range=250,
        sample_rate=100.0
    )

    # Créer le modèle du capteur
    sensor = MPU6050Model(config)

    # Paramètres de simulation
    duration = 10.0  # secondes
    n_samples = int(duration * config.sample_rate)
    dt = 1.0 / config.sample_rate

    # Direction de la gravité (vers le bas) dans le repère capteur,
    # capteur incliné de 30° en roll
    roll_angle = np.deg2rad(30)
    gravity_direction = -np.array([
        0,
        np.sin(roll_angle),
        np.cos(roll_angle)
    ])

    print(f"Durée: {duration} s")
    print(f"Fréquence: {config.sample_rate} Hz")
    print(f"Inclinaison: {np.rad2deg(roll_angle):.1f}°\n")

    # Simulation
    print("Simulation en cours...")
    accel_measurements, gyro_measurements = sensor.simulate_static_measurement(
        gravity_direction=gravity_direction,
        duration=duration
    )

    # Statistiques
    print("\n--- Statistiques des mesures ---")
    print(f"Accéléromètre:")
    print(f"  Moyenne: {np.mean(accel_measurements, axis=0)}")
    print(f"  Écart-type: {np.std(accel_measurements, axis=0)}")
    print(f"  Magnitude moyenne: {np.mean(np.linalg.norm(accel_measurements, axis=1)):.4f} m/s²")

    print(f"\nGyroscope:")
    print(f"  Moyenne: {np.rad2deg(np.mean(gyro_measurements, axis=0))} °/s")
    print(f"  Écart-type: {np.rad2deg(np.std(gyro_measurements, axis=0))} °/s")

    # Visualisation
    time = np.arange(n_samples) * dt

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Accéléromètre
    axes[0].plot(time, accel_measurements[:, 0], label='X', alpha=0.7)
    axes[0].plot(time, accel_measurements[:, 1], label='Y', alpha=0.7)
    axes[0].plot(time, accel_measurements[:, 2], label='Z', alpha=0.7)
    axes[0].set_xlabel('Temps [s]')
    axes[0].set_ylabel('Accélération [m/s²]')
    axes[0].set_title('Mesures de l\'accéléromètre (statique)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Gyroscope
    axes[1].plot(time, np.rad2deg(gyro_measurements[:, 0]), label='X', alpha=0.7)
    axes[1].plot(time, np.rad2deg(gyro_measurements[:, 1]), label='Y', alpha=0.7)
    axes[1].plot(time, np.rad2deg(gyro_measurements[:, 2]), label='Z', alpha=0.7)
    axes[1].set_xlabel('Temps [s]')
    axes[1].set_ylabel('Vitesse angulaire [°/s]')
    axes[1].set_title('Mesures du gyroscope (statique)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('static_simulation.png', dpi=150)
    print("\nGraphique sauvegardé: static_simulation.png")


def compare_fusion_algorithms():
    """
    Compare les algorithmes de fusion de capteurs
    """
    print("\n\n=== Comparaison des algorithmes de fusion ===\n")

    # Configuration
    config = MPU6050Config(sample_rate=100.0)
    sensor = MPU6050Model(config)

    # Initialiser les filtres
    ekf = QuaternionEKF(sample_rate=config.sample_rate)
    complementary = ComplementaryFilter(alpha=0.98, sample_rate=config.sample_rate)
    mahony = MahonyFilter(kp=2.0, ki=0.1, sample_rate=config.sample_rate)
    madgwick = MadgwickFilter(beta=0.1, sample_rate=config.sample_rate)

    # Scénario: rotation de 45° en pitch
    duration = 5.0
    n_samples = int(duration * config.sample_rate)
    dt = 1.0 / config.sample_rate

    # Orientation vraie (rotation progressive)
    pitch_final = np.deg2rad(45)
    pitch_true = np.linspace(0, pitch_final, n_samples)

    # Stockage des résultats
    results = {
        'ekf': {'roll': [], 'pitch': [], 'yaw': []},
        'complementary': {'roll': [], 'pitch': [], 'yaw': []},
        'mahony': {'roll': [], 'pitch': [], 'yaw': []},
        'madgwick': {'roll': [], 'pitch': [], 'yaw': []}
    }

    print("Simulation de rotation de 45° en pitch...")

    for i in range(n_samples):
        # Vitesse angulaire vraie
        if i < n_samples - 1:
            omega_true = np.array([0, (pitch_true[i+1] - pitch_true[i]) / dt, 0])
        else:
            omega_true = np.zeros(3)

        # Gravité dans le repère capteur (simplifié)
        pitch = pitch_true[i]
        accel_true = np.array([
            -9.81 * np.sin(pitch),
            0,
            9.81 * np.cos(pitch)
        ])

        # Mesures avec bruit
        accel_meas = sensor.measure_accel(accel_true, add_noise=True)
        gyro_meas = sensor.measure_gyro(omega_true, accel=accel_true, add_noise=True)

        # Mise à jour des filtres
        ekf.predict(gyro_meas)
        ekf.update(accel_meas)

        complementary.update(gyro_meas, accel_meas)

        mahony.update(gyro_meas, accel_meas)

        madgwick.update(gyro_meas, accel_meas)

        # Enregistrer les résultats
        roll_ekf, pitch_ekf, yaw_ekf = ekf.euler_angles
        results['ekf']['roll'].append(roll_ekf)
        results['ekf']['pitch'].append(pitch_ekf)
        results['ekf']['yaw'].append(yaw_ekf)

        roll_comp, pitch_comp, yaw_comp = complementary.euler_angles
        results['complementary']['roll'].append(roll_comp)
        results['complementary']['pitch'].append(pitch_comp)
        results['complementary']['yaw'].append(yaw_comp)

        roll_mah, pitch_mah, yaw_mah = mahony.euler_angles
        results['mahony']['roll'].append(roll_mah)
        results['mahony']['pitch'].append(pitch_mah)
        results['mahony']['yaw'].append(yaw_mah)

        roll_mad, pitch_mad, yaw_mad = madgwick.euler_angles
        results['madgwick']['roll'].append(roll_mad)
        results['madgwick']['pitch'].append(pitch_mad)
        results['madgwick']['yaw'].append(yaw_mad)

    # Visualisation
    time = np.arange(n_samples) * dt

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Pitch (angle principal)
    axes[0].plot(time, np.rad2deg(pitch_true), 'k--', linewidth=2, label='Vrai')
    axes[0].plot(time, np.rad2deg(results['ekf']['pitch']), label='EKF', alpha=0.8)
    axes[0].plot(time, np.rad2deg(results['complementary']['pitch']), label='Complémentaire', alpha=0.8)
    axes[0].plot(time, np.rad2deg(results['mahony']['pitch']), label='Mahony', alpha=0.8)
    axes[0].plot(time, np.rad2deg(results['madgwick']['pitch']), label='Madgwick', alpha=0.8)
    axes[0].set_xlabel('Temps [s]')
    axes[0].set_ylabel('Pitch [°]')
    axes[0].set_title('Comparaison des filtres - Angle de Pitch')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Erreur de pitch
    error_ekf = np.rad2deg(np.array(results['ekf']['pitch']) - pitch_true)
    error_comp = np.rad2deg(np.array(results['complementary']['pitch']) - pitch_true)
    error_mah = np.rad2deg(np.array(results['mahony']['pitch']) - pitch_true)
    error_mad = np.rad2deg(np.array(results['madgwick']['pitch']) - pitch_true)

    axes[1].plot(time, error_ekf, label='EKF', alpha=0.8)
    axes[1].plot(time, error_comp, label='Complémentaire', alpha=0.8)
    axes[1].plot(time, error_mah, label='Mahony', alpha=0.8)
    axes[1].plot(time, error_mad, label='Madgwick', alpha=0.8)
    axes[1].set_xlabel('Temps [s]')
    axes[1].set_ylabel('Erreur [°]')
    axes[1].set_title('Erreur de Pitch')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Roll (doit rester proche de 0)
    axes[2].plot(time, np.rad2deg(results['ekf']['roll']), label='EKF', alpha=0.8)
    axes[2].plot(time, np.rad2deg(results['complementary']['roll']), label='Complémentaire', alpha=0.8)
    axes[2].plot(time, np.rad2deg(results['mahony']['roll']), label='Mahony', alpha=0.8)
    axes[2].plot(time, np.rad2deg(results['madgwick']['roll']), label='Madgwick', alpha=0.8)
    axes[2].set_xlabel('Temps [s]')
    axes[2].set_ylabel('Roll [°]')
    axes[2].set_title('Angle de Roll (doit rester ~0°)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('fusion_comparison.png', dpi=150)
    print("Graphique sauvegardé: fusion_comparison.png")

    # Statistiques d'erreur
    print("\n--- Statistiques d'erreur (RMSE en pitch) ---")
    rmse_ekf = np.sqrt(np.mean(error_ekf**2))
    rmse_comp = np.sqrt(np.mean(error_comp**2))
    rmse_mah = np.sqrt(np.mean(error_mah**2))
    rmse_mad = np.sqrt(np.mean(error_mad**2))

    print(f"EKF:            {rmse_ekf:.4f}°")
    print(f"Complémentaire: {rmse_comp:.4f}°")
    print(f"Mahony:         {rmse_mah:.4f}°")
    print(f"Madgwick:       {rmse_mad:.4f}°")


if __name__ == "__main__":
    # Exécuter les simulations
    simulate_static_scenario()
    compare_fusion_algorithms()

    print("\n=== Simulation terminée ===")
    plt.show()
