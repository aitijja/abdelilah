"""
Exemple de calibration du MPU6050
"""

import numpy as np
import json
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.mpu6050_model import MPU6050Model, MPU6050Config, SensorCalibration, create_scale_matrix_with_errors
from src.utils.constants import GRAVITY


def calibrate_accelerometer_6_positions():
    """
    Calibration de l'accéléromètre avec la méthode des 6 positions
    """
    print("=== Calibration de l'accéléromètre (6 positions) ===\n")

    # Créer un capteur avec des erreurs connues pour tester
    config = MPU6050Config(accel_range=2)

    # Créer des erreurs de calibration
    true_calibration = SensorCalibration()
    true_calibration.S_a = create_scale_matrix_with_errors(
        scale_error_percent=2.0,
        cross_axis_percent=1.5
    )
    true_calibration.b_a = np.array([0.05, -0.03, 0.08]) * GRAVITY

    print("Paramètres de calibration vrais (à estimer):")
    print(f"Biais: {true_calibration.b_a}")
    print(f"Matrice de sensibilité:\n{true_calibration.S_a}\n")

    # Créer le modèle avec ces paramètres
    sensor = MPU6050Model(config, calibration=true_calibration)

    # 6 positions de référence (orientations de la gravité)
    positions = [
        np.array([1, 0, 0]),   # +X up
        np.array([-1, 0, 0]),  # -X up
        np.array([0, 1, 0]),   # +Y up
        np.array([0, -1, 0]),  # -Y up
        np.array([0, 0, 1]),   # +Z up
        np.array([0, 0, -1])   # -Z up
    ]

    position_names = ["+X up", "-X up", "+Y up", "-Y up", "+Z up", "-Z up"]

    # Collecter les mesures dans chaque position
    measurements = []
    references = []

    print("Collecte des mesures dans les 6 positions...\n")

    for i, gravity_dir in enumerate(positions):
        # Accélération vraie dans cette position
        accel_true = -GRAVITY * gravity_dir

        # Simuler N mesures et prendre la moyenne
        N = 1000
        samples = []
        for _ in range(N):
            meas = sensor.measure_accel(accel_true, add_noise=True)
            samples.append(meas)

        mean_measurement = np.mean(samples, axis=0)
        measurements.append(mean_measurement)
        references.append(accel_true)

        print(f"Position {i+1} ({position_names[i]}):")
        print(f"  Mesure moyenne: {mean_measurement}")
        print(f"  Référence:      {accel_true}\n")

    # Convertir en arrays numpy
    A = np.array(measurements)  # Mesures (6 x 3)
    G = np.array(references)    # Références (6 x 3)

    # Estimation des paramètres par moindres carrés
    print("Estimation des paramètres de calibration...")

    # Centrer les données
    A_mean = np.mean(A, axis=0)
    G_mean = np.mean(G, axis=0)
    A_centered = A - A_mean
    G_centered = G - G_mean

    # Estimation de S_a
    S_a_estimated = np.linalg.lstsq(G_centered, A_centered, rcond=None)[0].T

    # Estimation de b_a
    b_a_estimated = A_mean - S_a_estimated @ G_mean

    print("\nRésultats de la calibration:")
    print(f"Biais estimé: {b_a_estimated}")
    print(f"Biais vrai:   {true_calibration.b_a}")
    print(f"Erreur:       {b_a_estimated - true_calibration.b_a}\n")

    print(f"Matrice S_a estimée:")
    print(S_a_estimated)
    print(f"\nMatrice S_a vraie:")
    print(true_calibration.S_a)
    print(f"\nErreur:")
    print(S_a_estimated - true_calibration.S_a)

    # Test de la calibration
    print("\n--- Test de la calibration ---")

    # Corriger les mesures avec les paramètres estimés
    corrected_measurements = []
    for meas in measurements:
        corrected = np.linalg.inv(S_a_estimated) @ (meas - b_a_estimated)
        corrected_measurements.append(corrected)

    # Vérifier les magnitudes
    magnitudes = [np.linalg.norm(corr) for corr in corrected_measurements]
    print(f"\nMagnitudes après correction:")
    for i, mag in enumerate(magnitudes):
        error = mag - GRAVITY
        print(f"  Position {i+1}: {mag:.6f} m/s² (erreur: {error*1000:.2f} mg)")

    mean_magnitude = np.mean(magnitudes)
    std_magnitude = np.std(magnitudes)
    print(f"\nMoyenne: {mean_magnitude:.6f} m/s²")
    print(f"Écart-type: {std_magnitude*1000:.4f} mg")
    print(f"Erreur: {(mean_magnitude - GRAVITY)*1000:.4f} mg")

    return b_a_estimated, S_a_estimated


def calibrate_gyroscope_bias():
    """
    Calibration du biais du gyroscope (méthode statique)
    """
    print("\n\n=== Calibration du gyroscope (biais statique) ===\n")

    # Créer un capteur avec biais connu
    config = MPU6050Config(gyro_range=250, sample_rate=100.0)

    true_calibration = SensorCalibration()
    true_calibration.b_g = np.deg2rad(np.array([0.5, -0.3, 0.8]))  # °/s -> rad/s

    print(f"Biais vrai: {np.rad2deg(true_calibration.b_g)} °/s\n")

    sensor = MPU6050Model(config, calibration=true_calibration)

    # Mesure statique (vitesse angulaire nulle)
    omega_true = np.zeros(3)

    # Collecter N échantillons
    N = 5000
    duration = N / config.sample_rate
    print(f"Collecte de {N} échantillons ({duration} s à {config.sample_rate} Hz)...")

    samples = []
    for _ in range(N):
        meas = sensor.measure_gyro(omega_true, add_noise=True)
        samples.append(meas)

    samples = np.array(samples)

    # Estimation du biais
    b_g_estimated = np.mean(samples, axis=0)
    sigma_g = np.std(samples, axis=0)

    print(f"\nRésultats:")
    print(f"Biais estimé: {np.rad2deg(b_g_estimated)} °/s")
    print(f"Biais vrai:   {np.rad2deg(true_calibration.b_g)} °/s")
    print(f"Erreur:       {np.rad2deg(b_g_estimated - true_calibration.b_g)} °/s")

    print(f"\nÉcart-type:   {np.rad2deg(sigma_g)} °/s")

    # Test de la correction
    print("\n--- Test de la correction ---")
    corrected_samples = samples - b_g_estimated
    mean_corrected = np.mean(corrected_samples, axis=0)
    std_corrected = np.std(corrected_samples, axis=0)

    print(f"Moyenne après correction: {np.rad2deg(mean_corrected)} °/s")
    print(f"Écart-type après correction: {np.rad2deg(std_corrected)} °/s")

    # Dérive intégrée
    dt = 1.0 / config.sample_rate
    angle_drift_before = np.cumsum(samples, axis=0) * dt
    angle_drift_after = np.cumsum(corrected_samples, axis=0) * dt

    print(f"\nDérive angulaire après {duration} s:")
    print(f"  Avant correction: {np.rad2deg(angle_drift_before[-1])} °")
    print(f"  Après correction: {np.rad2deg(angle_drift_after[-1])} °")

    return b_g_estimated


def save_calibration_to_json(b_a, S_a, b_g, filename="calibration.json"):
    """
    Sauvegarde les paramètres de calibration en JSON
    """
    calibration_data = {
        "calibration_date": "2024-01-15T10:30:00Z",
        "serial_number": "MPU6050-EXAMPLE",
        "accelerometer": {
            "bias": b_a.tolist(),
            "scale_matrix": S_a.tolist(),
            "misalignment": [0.0, 0.0, 0.0]
        },
        "gyroscope": {
            "bias": b_g.tolist(),
            "scale_matrix": np.eye(3).tolist()
        },
        "noise_parameters": {
            "accel_noise_density": 0.003924,
            "gyro_noise_density": 8.73e-5,
            "gyro_bias_instability": 0.01,
            "gyro_arw": 0.1
        }
    }

    with open(filename, 'w') as f:
        json.dump(calibration_data, f, indent=2)

    print(f"\n\nParamètres de calibration sauvegardés dans: {filename}")


if __name__ == "__main__":
    # Exécuter les calibrations
    b_a, S_a = calibrate_accelerometer_6_positions()
    b_g = calibrate_gyroscope_bias()

    # Sauvegarder en JSON
    save_calibration_to_json(b_a, S_a, b_g)

    print("\n=== Calibration terminée ===")
