"""
Modèle mathématique complet du capteur MPU6050
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass
from scipy.signal import butter, lfilter, lfilter_zi

from .trajectory import Trajectory
from ..utils.constants import (
    GRAVITY, ACCEL_SCALE_FACTOR, GYRO_SCALE_FACTOR,
    ACCEL_NOISE_DENSITY_SI, GYRO_NOISE_DENSITY_RAD,
    ACCEL_BIAS_STABILITY, GYRO_RRW, GYRO_G_SENSITIVITY,
    DLPF_CUTOFF_FREQ, DLPF_CUTOFF_FREQ_GYRO,
    INTERNAL_SAMPLE_RATE_HZ, SAMPLE_RATE_HZ
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


@dataclass
class DynamicMeasurements:
    """Résultat d'une simulation dynamique, à la cadence de sortie du capteur"""
    time: np.ndarray             # (N,) instants d'échantillonnage [s]
    accel: np.ndarray            # (N, 3) mesures de l'accéléromètre [m/s²]
    gyro: np.ndarray             # (N, 3) mesures du gyroscope [rad/s]
    accel_true: np.ndarray       # (N, 3) force spécifique vraie [m/s²]
    gyro_true: np.ndarray        # (N, 3) vitesse angulaire vraie [rad/s]
    quaternion_true: np.ndarray  # (N, 4) attitude vraie
    euler_true: np.ndarray       # (N, 3) roll, pitch, yaw vrais [rad]


class MPU6050Model:
    """
    Modèle mathématique complet du MPU6050

    Ce modèle implémente les équations de mesure incluant:
    - Conversion ADC vers unités physiques
    - Matrices de sensibilité et non-orthogonalité
    - Biais déterministes et stochastiques
    - Bruits de mesure blanc gaussien
    - G-sensibilité du gyroscope
    - Réponse dynamique (simulate_dynamic_measurement) : filtre passe-bas
      DLPF, décimation, saturation et quantification ADC
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

    def measure_accel(self,
                     accel_true: np.ndarray,
                     add_noise: bool = True) -> np.ndarray:
        """
        Simule la mesure de l'accéléromètre avec modèle complet

        Équation: y_acc = S_a · (a_true + b_a + n_a) + m_a

        Args:
            accel_true: Accélération vraie dans le repère capteur (3,) [m/s²]
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
                    accel: Optional[np.ndarray] = None,
                    add_noise: bool = True) -> np.ndarray:
        """
        Simule la mesure du gyroscope avec modèle complet

        Équation: y_gyro = S_g · (ω_true + b_g + n_g) + m_g + K_ga · a

        Args:
            gyro_true: Vitesse angulaire vraie dans le repère capteur (3,) [rad/s]
            accel: Force spécifique pour la g-sensibilité (3,) [m/s²]
            add_noise: Ajouter le bruit de mesure

        Returns:
            Mesure du gyroscope (3,) [rad/s]
        """
        # Bruit de mesure blanc gaussien
        if add_noise:
            noise = np.random.normal(0, self.sigma_g_d, 3)
        else:
            noise = np.zeros(3)

        # Application du modèle de base
        measurement = (self.calibration.S_g @
                      (gyro_true + self.bias_g_current + noise) +
                      self.calibration.m_g)

        # G-sensibilité (effet des accélérations linéaires)
        if accel is not None:
            # Modèle simplifié: effet proportionnel sur chaque axe, 0.2 (°/s)/g
            measurement += GYRO_G_SENSITIVITY * accel / GRAVITY

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

    def correct_measurement_gyro(self, measurement: np.ndarray) -> np.ndarray:
        """
        Corrige la mesure du gyroscope (calibration inverse)

        Args:
            measurement: Mesure brute (3,) [rad/s]

        Returns:
            Vitesse angulaire corrigée (3,) [rad/s]
        """
        # Inverse du modèle: ω_true = S_g^(-1) · (y_gyro - m_g) - b_g
        corrected = np.linalg.inv(self.calibration.S_g) @ (measurement - self.calibration.m_g)
        corrected -= self.bias_g_current
        return corrected

    def simulate_static_measurement(self,
                                   gravity_direction: np.ndarray,
                                   duration: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simule des mesures statiques (pas de mouvement, seulement gravité)

        Args:
            gravity_direction: Direction de la gravité dans le repère capteur (3,) [unitaire]
            duration: Durée de la simulation [s]

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
            accel_measurements[i] = self.measure_accel(accel_true)
            gyro_measurements[i] = self.measure_gyro(gyro_true, accel_true)

            # Mise à jour du biais (random walk)
            self.update_bias_random_walk()

        return accel_measurements, gyro_measurements

    def simulate_dynamic_measurement(self,
                                    trajectory: Trajectory,
                                    duration: float,
                                    add_noise: bool = True,
                                    quantize: bool = True) -> DynamicMeasurements:
        """
        Simule les mesures du capteur en mouvement le long d'une trajectoire

        Chaîne de mesure (docs/MATHEMATICAL_MODELS.md, §5.7) :
          1. ω_S(t) et f_S(t) vrais calculés à la cadence interne (1 kHz)
          2. Modèle d'erreurs : y = S·(x + b + n) + m, plus la g-sensibilité
             du gyroscope (b suit une marche aléatoire)
          3. Filtre passe-bas DLPF (Butterworth d'ordre 2, fréquence de
             coupure selon config.dlpf_cfg) à la cadence interne
          4. Décimation à config.sample_rate = 1 kHz / (1 + SMPLRT_DIV)
          5. Saturation à la pleine échelle et quantification ADC 16 bits

        Args:
            trajectory: Trajectoire du capteur (attitude + position)
            duration: Durée de la simulation [s]
            add_noise: Ajouter le bruit blanc et la marche aléatoire des biais
            quantize: Arrondir les mesures au LSB de l'ADC

        Returns:
            DynamicMeasurements (mesures et grandeurs vraies aux instants de sortie)
        """
        fs_int = float(INTERNAL_SAMPLE_RATE_HZ)
        decimation = fs_int / self.config.sample_rate
        if decimation < 1 or abs(decimation - round(decimation)) > 1e-9:
            raise ValueError(
                f"sample_rate = {self.config.sample_rate} Hz invalide : la cadence de "
                f"sortie doit valoir {fs_int:.0f} Hz / (1 + SMPLRT_DIV)")
        decimation = int(round(decimation))
        if self.config.dlpf_cfg not in DLPF_CUTOFF_FREQ:
            raise ValueError(f"dlpf_cfg = {self.config.dlpf_cfg} invalide (0 à 6)")

        n_out = int(round(duration * self.config.sample_rate))
        if n_out < 1:
            raise ValueError("duration doit couvrir au moins une période d'échantillonnage")
        n_int = n_out * decimation
        t_int = np.arange(n_int) / fs_int

        # 1. Grandeurs vraies à la cadence interne
        f_true = trajectory.specific_force(t_int)
        w_true = trajectory.angular_velocity(t_int)

        # 2. Modèle d'erreurs
        if add_noise:
            n_a = np.random.normal(0, ACCEL_NOISE_DENSITY_SI * np.sqrt(fs_int), (n_int, 3))
            n_g = np.random.normal(0, GYRO_NOISE_DENSITY_RAD * np.sqrt(fs_int), (n_int, 3))
            w_a = np.random.normal(0, ACCEL_BIAS_STABILITY / np.sqrt(fs_int), (n_int, 3))
            w_g = np.random.normal(0, GYRO_RRW / np.sqrt(fs_int), (n_int, 3))
        else:
            n_a = n_g = w_a = w_g = np.zeros((n_int, 3))

        # Biais en marche aléatoire, partant de l'état courant
        b_a = self.bias_a_current + np.cumsum(w_a, axis=0) - w_a
        b_g = self.bias_g_current + np.cumsum(w_g, axis=0) - w_g
        self.bias_a_current = self.bias_a_current + w_a.sum(axis=0)
        self.bias_g_current = self.bias_g_current + w_g.sum(axis=0)

        cal = self.calibration
        raw_a = (f_true + b_a + n_a) @ cal.S_a.T + cal.m_a
        raw_g = ((w_true + b_g + n_g) @ cal.S_g.T + cal.m_g +
                 GYRO_G_SENSITIVITY * f_true / GRAVITY)

        # 3. Filtre passe-bas numérique (DLPF)
        filt_a = self._dlpf(raw_a, DLPF_CUTOFF_FREQ[self.config.dlpf_cfg], fs_int)
        filt_g = self._dlpf(raw_g, DLPF_CUTOFF_FREQ_GYRO[self.config.dlpf_cfg], fs_int)

        # 4. Décimation : la sortie k lit le dernier échantillon interne à t_k
        idx = np.arange(n_out) * decimation
        accel = filt_a[idx]
        gyro = filt_g[idx]

        # 5. Saturation et quantification ADC
        accel = self.adc_to_accel(self._adc(self.accel_to_adc(accel), quantize))
        gyro = self.adc_to_gyro(self._adc(self.gyro_to_adc(gyro), quantize))

        t_out = t_int[idx]
        return DynamicMeasurements(
            time=t_out,
            accel=accel,
            gyro=gyro,
            accel_true=f_true[idx],
            gyro_true=w_true[idx],
            quaternion_true=trajectory.quaternion(t_out),
            euler_true=trajectory.euler(t_out)
        )

    @staticmethod
    def _dlpf(x: np.ndarray, cutoff: float, fs: float) -> np.ndarray:
        """
        Filtre passe-bas Butterworth d'ordre 2 (modèle du DLPF), appliqué
        colonne par colonne, initialisé en régime permanent sur x[0]

        Args:
            x: Signal (N, 3)
            cutoff: Fréquence de coupure -3 dB [Hz]
            fs: Cadence du signal [Hz]

        Returns:
            Signal filtré (N, 3)
        """
        b, a = butter(2, cutoff, fs=fs)
        zi = lfilter_zi(b, a)[:, None] * x[0][None, :]
        y, _ = lfilter(b, a, x, axis=0, zi=zi)
        return y

    @staticmethod
    def _adc(counts: np.ndarray, quantize: bool) -> np.ndarray:
        """
        Saturation (et arrondi) d'une valeur exprimée en LSB, ADC 16 bits signé

        Args:
            counts: Valeur en LSB (non arrondie)
            quantize: Arrondir au LSB

        Returns:
            Valeur en LSB dans [-32768, 32767]
        """
        if quantize:
            counts = np.round(counts)
        return np.clip(counts, -32768, 32767)


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
