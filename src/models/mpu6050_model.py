"""
Modèle mathématique complet du capteur MPU6050
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass

from ..utils.constants import (
    GRAVITY, ACCEL_SCALE_FACTOR, GYRO_SCALE_FACTOR,
    ACCEL_NOISE_DENSITY_SI, GYRO_NOISE_DENSITY_RAD,
    ACCEL_BIAS_STABILITY, GYRO_RRW,
    TEMP_SENSITIVITY, TEMP_OFFSET,
    TEMP_COEFF_GYRO, SAMPLE_RATE_HZ
)


@dataclass
class MPU6050Config:
    """Configuration du MPU6050"""
    accel_range: int = 2        # ±2g, ±4g, ±8g, ±16g
    gyro_range: int = 250       # ±250, ±500, ±1000, ±2000 °/s
    sample_rate: float = SAMPLE_RATE_HZ  # Hz
    dlpf_cfg: int = 3           # Digital Low Pass Filter configuration


@dataclass
class SensorCalibration:
    """Paramètres de calibration du capteur"""
    # Matrice de sensibilité de l'accéléromètre (3x3)
    S_a: np.ndarray = None
    # Matrice de sensibilité du gyroscope (3x3)
    S_g: np.ndarray = None
    # Biais de l'accéléromètre (3,)
    b_a: np.ndarray = None
    # Biais du gyroscope (3,)
    b_g: np.ndarray = None
    # Désalignement de l'accéléromètre (3,)
    m_a: np.ndarray = None
    # Désalignement du gyroscope (3,)
    m_g: np.ndarray = None
    # Température de référence de calibration
    T_ref: float = 25.0  # °C

    def __post_init__(self):
        if self.S_a is None:
            self.S_a = np.eye(3)
        if self.S_g is None:
            self.S_g = np.eye(3)
        if self.b_a is None:
            self.b_a = np.zeros(3)
        if self.b_g is None:
            self.b_g = np.zeros(3)
        if self.m_a is None:
            self.m_a = np.zeros(3)
        if self.m_g is None:
            self.m_g = np.zeros(3)


class MPU6050Model:
    """
    Modèle mathématique complet du MPU6050

    Ce modèle implémente les équations de mesure incluant:
    - Conversion ADC vers unités physiques
    - Matrices de sensibilité et non-orthogonalité
    - Biais déterministes et stochastiques
    - Bruits de mesure blanc gaussien
    - Compensation thermique
    - G-sensibilité du gyroscope
    """

    def __init__(self, config: MPU6050Config, calibration: Optional[SensorCalibration] = None):
        """
        Initialise le modèle du MPU6050

        Args:
            config: Configuration du capteur
            calibration: Paramètres de calibration (optionnel)
        """
        self.config = config
        self.calibration = calibration if calibration else SensorCalibration()

        # Facteurs d'échelle ADC
        self.accel_scale = ACCEL_SCALE_FACTOR[config.accel_range]
        self.gyro_scale = GYRO_SCALE_FACTOR[config.gyro_range]

        # Calcul des variances de bruit discrètes
        dt = 1.0 / config.sample_rate
        self.sigma_a_d = ACCEL_NOISE_DENSITY_SI * np.sqrt(config.sample_rate)
        self.sigma_g_d = GYRO_NOISE_DENSITY_RAD * np.sqrt(config.sample_rate)

        # Variance du bruit de processus pour le biais (random walk)
        self.sigma_wa = ACCEL_BIAS_STABILITY * np.sqrt(dt)
        self.sigma_wg = GYRO_RRW * np.sqrt(dt)

        # État du biais évoluant (pour simulation)
        self.bias_a_current = self.calibration.b_a.copy()
        self.bias_g_current = self.calibration.b_g.copy()

    def adc_to_accel(self, adc_value: np.ndarray) -> np.ndarray:
        """
        Convertit les valeurs ADC de l'accéléromètre en m/s²

        Args:
            adc_value: Valeurs ADC brutes (3,)

        Returns:
            Accélération en m/s² (3,)
        """
        return (adc_value / self.accel_scale) * GRAVITY

    def adc_to_gyro(self, adc_value: np.ndarray) -> np.ndarray:
        """
        Convertit les valeurs ADC du gyroscope en rad/s

        Args:
            adc_value: Valeurs ADC brutes (3,)

        Returns:
            Vitesse angulaire en rad/s (3,)
        """
        return np.deg2rad(adc_value / self.gyro_scale)

    def accel_to_adc(self, accel: np.ndarray) -> np.ndarray:
        """
        Convertit l'accélération (m/s²) en valeurs ADC

        Args:
            accel: Accélération en m/s² (3,)

        Returns:
            Valeurs ADC (3,)
        """
        return (accel / GRAVITY) * self.accel_scale

    def gyro_to_adc(self, gyro: np.ndarray) -> np.ndarray:
        """
        Convertit la vitesse angulaire (rad/s) en valeurs ADC

        Args:
            gyro: Vitesse angulaire en rad/s (3,)

        Returns:
            Valeurs ADC (3,)
        """
        return np.rad2deg(gyro) * self.gyro_scale

    def temp_adc_to_celsius(self, adc_value: float) -> float:
        """
        Convertit la valeur ADC de température en °C

        Args:
            adc_value: Valeur ADC brute

        Returns:
            Température en °C
        """
        return (adc_value / TEMP_SENSITIVITY) + TEMP_OFFSET

    def measure_accel(self,
                     accel_true: np.ndarray,
                     temperature: float = 25.0,
                     add_noise: bool = True) -> np.ndarray:
        """
        Simule la mesure de l'accéléromètre avec modèle complet

        Équation: y_acc = S_a · (a_true + b_a + n_a) + m_a

        Args:
            accel_true: Accélération vraie dans le repère capteur (3,) [m/s²]
            temperature: Température du capteur [°C]
            add_noise: Ajouter le bruit de mesure

        Returns:
            Mesure de l'accéléromètre (3,) [m/s²]
        """
        # Bruit de mesure blanc gaussien
        if add_noise:
            noise = np.random.normal(0, self.sigma_a_d, 3)
        else:
            noise = np.zeros(3)

        # Application du modèle complet
        measurement = (self.calibration.S_a @
                      (accel_true + self.bias_a_current + noise) +
                      self.calibration.m_a)

        return measurement

    def measure_gyro(self,
                    gyro_true: np.ndarray,
                    temperature: float = 25.0,
                    accel: Optional[np.ndarray] = None,
                    add_noise: bool = True) -> np.ndarray:
        """
        Simule la mesure du gyroscope avec modèle complet

        Équation: y_gyro = S_g · (ω_true + b_g(T) + n_g) + m_g + K_ga · a

        Args:
            gyro_true: Vitesse angulaire vraie dans le repère capteur (3,) [rad/s]
            temperature: Température du capteur [°C]
            accel: Accélération pour compensation g-sensitivity (3,) [m/s²]
            add_noise: Ajouter le bruit de mesure

        Returns:
            Mesure du gyroscope (3,) [rad/s]
        """
        # Compensation thermique du biais
        delta_T = temperature - self.calibration.T_ref
        bias_temp = self.bias_g_current + TEMP_COEFF_GYRO * delta_T

        # Bruit de mesure blanc gaussien
        if add_noise:
            noise = np.random.normal(0, self.sigma_g_d, 3)
        else:
            noise = np.zeros(3)

        # Application du modèle de base
        measurement = (self.calibration.S_g @
                      (gyro_true + bias_temp + noise) +
                      self.calibration.m_g)

        # G-sensibilité (effet des accélérations linéaires)
        if accel is not None:
            # Modèle simplifié: effet proportionnel sur chaque axe
            g_effect = 0.002 * accel / GRAVITY  # ~0.2 (°/s)/g converted
            measurement += g_effect

        return measurement

    def update_bias_random_walk(self):
        """
        Met à jour le biais selon le modèle de marche aléatoire (random walk)

        Équation discrète: b[k+1] = b[k] + w[k]
        où w[k] ~ N(0, σ_w² · Δt)
        """
        # Bruit de processus pour l'accéléromètre
        w_a = np.random.normal(0, self.sigma_wa, 3)
        self.bias_a_current += w_a

        # Bruit de processus pour le gyroscope
        w_g = np.random.normal(0, self.sigma_wg, 3)
        self.bias_g_current += w_g

    def get_measurement_covariance(self) -> np.ndarray:
        """
        Retourne la matrice de covariance du bruit de mesure R

        Returns:
            Matrice R (6x6) pour [accel, gyro]
        """
        R = np.zeros((6, 6))
        R[0:3, 0:3] = (self.sigma_a_d ** 2) * np.eye(3)
        R[3:6, 3:6] = (self.sigma_g_d ** 2) * np.eye(3)
        return R

    def get_process_noise_covariance(self) -> np.ndarray:
        """
        Retourne la matrice de covariance du bruit de processus Q pour le biais

        Returns:
            Matrice Q (6x6) pour [b_a, b_g]
        """
        Q = np.zeros((6, 6))
        Q[0:3, 0:3] = (self.sigma_wa ** 2) * np.eye(3)
        Q[3:6, 3:6] = (self.sigma_wg ** 2) * np.eye(3)
        return Q

    def correct_measurement_accel(self, measurement: np.ndarray) -> np.ndarray:
        """
        Corrige la mesure de l'accéléromètre (calibration inverse)

        Args:
            measurement: Mesure brute (3,) [m/s²]

        Returns:
            Accélération corrigée (3,) [m/s²]
        """
        # Inverse du modèle: a_true = S_a^(-1) · (y_acc - m_a) - b_a
        corrected = np.linalg.inv(self.calibration.S_a) @ (measurement - self.calibration.m_a)
        corrected -= self.bias_a_current
        return corrected

    def correct_measurement_gyro(self,
                                measurement: np.ndarray,
                                temperature: float = 25.0) -> np.ndarray:
        """
        Corrige la mesure du gyroscope (calibration inverse avec température)

        Args:
            measurement: Mesure brute (3,) [rad/s]
            temperature: Température du capteur [°C]

        Returns:
            Vitesse angulaire corrigée (3,) [rad/s]
        """
        # Compensation thermique du biais
        delta_T = temperature - self.calibration.T_ref
        bias_temp = self.bias_g_current + TEMP_COEFF_GYRO * delta_T

        # Inverse du modèle: ω_true = S_g^(-1) · (y_gyro - m_g) - b_g(T)
        corrected = np.linalg.inv(self.calibration.S_g) @ (measurement - self.calibration.m_g)
        corrected -= bias_temp
        return corrected

    def simulate_static_measurement(self,
                                   gravity_direction: np.ndarray,
                                   duration: float,
                                   temperature: float = 25.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simule des mesures statiques (pas de mouvement, seulement gravité)

        Args:
            gravity_direction: Direction de la gravité dans le repère capteur (3,) [unitaire]
            duration: Durée de la simulation [s]
            temperature: Température [°C]

        Returns:
            Tuple (accel_measurements, gyro_measurements)
            - accel_measurements: (N, 3) [m/s²]
            - gyro_measurements: (N, 3) [rad/s]
        """
        n_samples = int(duration * self.config.sample_rate)

        # Accélération vraie = -g dans la direction de la gravité
        accel_true = -GRAVITY * (gravity_direction / np.linalg.norm(gravity_direction))

        # Vitesse angulaire nulle (statique)
        gyro_true = np.zeros(3)

        accel_measurements = np.zeros((n_samples, 3))
        gyro_measurements = np.zeros((n_samples, 3))

        for i in range(n_samples):
            # Mesures avec bruit et biais évoluant
            accel_measurements[i] = self.measure_accel(accel_true, temperature)
            gyro_measurements[i] = self.measure_gyro(gyro_true, temperature, accel_true)

            # Mise à jour du biais (random walk)
            self.update_bias_random_walk()

        return accel_measurements, gyro_measurements


def create_scale_matrix_with_errors(scale_error_percent: float = 0.0,
                                   cross_axis_percent: float = 0.0) -> np.ndarray:
    """
    Crée une matrice de sensibilité avec erreurs d'échelle et de non-orthogonalité

    Args:
        scale_error_percent: Erreur d'échelle en % (ex: 2.0 pour ±2%)
        cross_axis_percent: Sensibilité croisée en % (ex: 2.0 pour ±2%)

    Returns:
        Matrice de sensibilité S (3x3)
    """
    # Erreurs d'échelle sur la diagonale
    scale_errors = 1.0 + np.random.uniform(-scale_error_percent/100,
                                          scale_error_percent/100, 3)

    # Non-orthogonalité hors diagonale
    cross_axis = np.random.uniform(-cross_axis_percent/100,
                                  cross_axis_percent/100, (3, 3))

    S = np.diag(scale_errors)
    S[0, 1] = cross_axis[0, 1]  # α_yx
    S[0, 2] = cross_axis[0, 2]  # α_zx
    S[1, 0] = cross_axis[1, 0]  # α_xy
    S[1, 2] = cross_axis[1, 2]  # α_zy
    S[2, 0] = cross_axis[2, 0]  # α_xz
    S[2, 1] = cross_axis[2, 1]  # α_yz

    return S
