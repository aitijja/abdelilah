"""
Filtre de Mahony pour estimation d'orientation avec MPU6050
"""

import numpy as np
from typing import Tuple


class MahonyFilter:
    """
    Filtre de Mahony avec correction PI (Proportionnel-Intégral)

    Utilise une rétroaction pour corriger la dérive du gyroscope
    """

    def __init__(self,
                 kp: float = 1.0,
                 ki: float = 0.0,
                 sample_rate: float = 100.0,
                 gravity: float = 9.81,
                 initial_orientation: np.ndarray = None):
        """
        Initialise le filtre de Mahony

        Args:
            kp: Gain proportionnel (typiquement 0.5-5.0)
            ki: Gain intégral (typiquement 0.0-0.5, 0 = filtre P pur)
            sample_rate: Fréquence d'échantillonnage [Hz]
            gravity: Magnitude de la gravité [m/s²]
            initial_orientation: Quaternion initial [q₀, q₁, q₂, q₃]
        """
        self.kp = kp
        self.ki = ki
        self.dt = 1.0 / sample_rate
        self.g = gravity

        # Quaternion d'orientation
        if initial_orientation is None:
            self.q = np.array([1.0, 0.0, 0.0, 0.0])  # Identité
        else:
            self.q = initial_orientation / np.linalg.norm(initial_orientation)

        # Biais intégré du gyroscope
        self.bias_integral = np.zeros(3)

        # Seuils pour détection de mouvement
        self.accel_threshold_low = 0.8 * gravity
        self.accel_threshold_high = 1.2 * gravity

    @property
    def quaternion(self) -> np.ndarray:
        """Retourne le quaternion d'orientation"""
        return self.q

    @property
    def gyro_bias(self) -> np.ndarray:
        """Retourne l'estimation du biais du gyroscope"""
        return self.bias_integral.copy()

    @property
    def euler_angles(self) -> Tuple[float, float, float]:
        """
        Convertit le quaternion en angles d'Euler

        Returns:
            Tuple (roll, pitch, yaw) en radians
        """
        q0, q1, q2, q3 = self.q[0], self.q[1], self.q[2], self.q[3]

        roll = np.arctan2(2*(q0*q1 + q2*q3), 1 - 2*(q1**2 + q2**2))
        pitch = np.arcsin(np.clip(2*(q0*q2 - q3*q1), -1.0, 1.0))
        yaw = np.arctan2(2*(q0*q3 + q1*q2), 1 - 2*(q2**2 + q3**2))

        return roll, pitch, yaw

    def update(self,
               gyro_measurement: np.ndarray,
               accel_measurement: np.ndarray) -> bool:
        """
        Met à jour l'estimation d'orientation

        Args:
            gyro_measurement: Vitesse angulaire [ω_x, ω_y, ω_z] en rad/s
            accel_measurement: Accélération [a_x, a_y, a_z] en m/s²

        Returns:
            True si l'accéléromètre a été utilisé pour la correction
        """
        # 1. Normaliser l'accélération
        accel_norm = np.linalg.norm(accel_measurement)

        # Détection de mouvement dynamique
        use_accel = (self.accel_threshold_low <= accel_norm <= self.accel_threshold_high)

        # Vecteur d'erreur
        error = np.zeros(3)

        if use_accel:
            # 2. Normaliser la mesure de l'accéléromètre
            a_norm = accel_measurement / accel_norm

            # 3. Prédire la direction de la gravité dans le repère capteur
            g_pred = self._predict_gravity()

            # 4. Calculer l'erreur (produit vectoriel)
            error = np.cross(a_norm, g_pred)

            # 5. Mise à jour du terme intégral
            if self.ki > 0:
                self.bias_integral -= self.ki * error * self.dt

        # 6. Correction PI du gyroscope
        omega_corrected = gyro_measurement - self.bias_integral + self.kp * error

        # 7. Intégration du quaternion
        self._integrate_quaternion(omega_corrected)

        # 8. Normalisation
        self.q = self.q / np.linalg.norm(self.q)

        return use_accel

    def _predict_gravity(self) -> np.ndarray:
        """
        Prédit la direction de la gravité dans le repère capteur

        Returns:
            Vecteur gravité normalisé dans le repère capteur (3,)
        """
        q0, q1, q2, q3 = self.q[0], self.q[1], self.q[2], self.q[3]

        # R(q)ᵀ * [0, 0, 1]ᵀ (gravité normalisée)
        g_pred = np.array([
            2 * (q1*q3 - q0*q2),
            2 * (q0*q1 + q2*q3),
            q0**2 - q1**2 - q2**2 + q3**2
        ])

        return g_pred

    def _integrate_quaternion(self, omega: np.ndarray):
        """
        Intègre le quaternion avec la vitesse angulaire corrigée

        Args:
            omega: Vitesse angulaire corrigée [rad/s]
        """
        wx, wy, wz = omega[0], omega[1], omega[2]

        # Matrice Ω(ω)
        Omega = np.array([
            [0,   -wx,  -wy,  -wz],
            [wx,   0,    wz,  -wy],
            [wy,  -wz,   0,    wx],
            [wz,   wy,  -wx,   0]
        ])

        # Dérivée du quaternion
        q_dot = 0.5 * Omega @ self.q

        # Intégration d'Euler
        self.q = self.q + q_dot * self.dt

    def reset_bias(self):
        """Réinitialise l'estimation du biais intégré"""
        self.bias_integral = np.zeros(3)

    def set_gains(self, kp: float, ki: float):
        """
        Ajuste les gains du filtre

        Args:
            kp: Gain proportionnel
            ki: Gain intégral
        """
        self.kp = kp
        self.ki = ki

    def get_rotation_matrix(self) -> np.ndarray:
        """
        Convertit le quaternion en matrice de rotation

        Returns:
            Matrice de rotation R (3x3)
        """
        q0, q1, q2, q3 = self.q[0], self.q[1], self.q[2], self.q[3]

        R = np.array([
            [1 - 2*(q2**2 + q3**2), 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2)],
            [2*(q1*q2 + q0*q3), 1 - 2*(q1**2 + q3**2), 2*(q2*q3 - q0*q1)],
            [2*(q1*q3 - q0*q2), 2*(q2*q3 + q0*q1), 1 - 2*(q1**2 + q2**2)]
        ])

        return R
