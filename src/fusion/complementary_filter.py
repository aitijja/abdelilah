"""
Filtre complémentaire pour estimation d'orientation avec MPU6050
"""

import numpy as np
from typing import Tuple


class ComplementaryFilter:
    """
    Filtre complémentaire avec quaternions

    Fusionne gyroscope (haute fréquence) et accéléromètre (basse fréquence)
    """

    def __init__(self,
                 alpha: float = 0.98,
                 sample_rate: float = 100.0,
                 gravity: float = 9.81,
                 initial_orientation: np.ndarray = None):
        """
        Initialise le filtre complémentaire

        Args:
            alpha: Coefficient de pondération (0-1), typiquement 0.95-0.99
                  alpha = τ/(τ + Δt) où τ est la constante de temps
            sample_rate: Fréquence d'échantillonnage [Hz]
            gravity: Magnitude de la gravité [m/s²]
            initial_orientation: Quaternion initial [q₀, q₁, q₂, q₃]
        """
        self.alpha = alpha
        self.dt = 1.0 / sample_rate
        self.g = gravity

        # Quaternion d'orientation
        if initial_orientation is None:
            self.q = np.array([1.0, 0.0, 0.0, 0.0])  # Identité
        else:
            self.q = initial_orientation / np.linalg.norm(initial_orientation)

        # Calcul de la constante de temps
        self.tau = alpha * self.dt / (1 - alpha)

        # Seuils pour détection de mouvement
        self.accel_threshold_low = 0.8 * gravity
        self.accel_threshold_high = 1.2 * gravity

    @property
    def quaternion(self) -> np.ndarray:
        """Retourne le quaternion d'orientation"""
        return self.q

    @property
    def euler_angles(self) -> Tuple[float, float, float]:
        """
        Convertit le quaternion en angles d'Euler

        Returns:
            Tuple (roll, pitch, yaw) en radians
        """
        q0, q1, q2, q3 = self.q[0], self.q[1], self.q[2], self.q[3]

        roll = np.arctan2(2*(q0*q1 + q2*q3), 1 - 2*(q1**2 + q2**2))
        pitch = np.arcsin(2*(q0*q2 - q3*q1))
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
            True si l'accéléromètre a été utilisé
        """
        # 1. Prédiction avec gyroscope
        q_pred = self._integrate_gyro(gyro_measurement)

        # 2. Détection de mouvement
        accel_norm = np.linalg.norm(accel_measurement)
        use_accel = (self.accel_threshold_low <= accel_norm <= self.accel_threshold_high)

        if use_accel:
            # 3. Correction avec accéléromètre
            q_accel = self._quaternion_from_accel(accel_measurement)

            # 4. Fusion complémentaire (SLERP)
            self.q = self._slerp(q_pred, q_accel, 1 - self.alpha)
        else:
            # Utiliser uniquement le gyroscope
            self.q = q_pred

        # Normalisation
        self.q = self.q / np.linalg.norm(self.q)

        return use_accel

    def _integrate_gyro(self, omega: np.ndarray) -> np.ndarray:
        """
        Intègre le gyroscope pour prédire le quaternion

        Args:
            omega: Vitesse angulaire [rad/s]

        Returns:
            Quaternion prédit
        """
        # Matrice Ω(ω)
        wx, wy, wz = omega[0], omega[1], omega[2]
        Omega = np.array([
            [0,   -wx,  -wy,  -wz],
            [wx,   0,    wz,  -wy],
            [wy,  -wz,   0,    wx],
            [wz,   wy,  -wx,   0]
        ])

        # Intégration d'Euler: q_pred = q + (Δt/2) * Ω(ω) * q
        q_dot = 0.5 * Omega @ self.q
        q_pred = self.q + q_dot * self.dt

        return q_pred / np.linalg.norm(q_pred)

    def _quaternion_from_accel(self, accel: np.ndarray) -> np.ndarray:
        """
        Calcule un quaternion d'orientation depuis l'accéléromètre

        Suppose que l'accéléromètre mesure uniquement la gravité (statique)

        Args:
            accel: Mesure de l'accéléromètre normalisée

        Returns:
            Quaternion estimé
        """
        # Normaliser l'accélération
        a_norm = accel / np.linalg.norm(accel)

        # Vecteur gravité dans le repère monde
        g_world = np.array([0, 0, 1])

        # Calculer l'axe de rotation (produit vectoriel)
        v = np.cross(g_world, a_norm)
        v_norm = np.linalg.norm(v)

        if v_norm < 1e-6:
            # Vecteurs alignés ou anti-alignés
            if np.dot(g_world, a_norm) > 0:
                return np.array([1.0, 0.0, 0.0, 0.0])  # Pas de rotation
            else:
                return np.array([0.0, 1.0, 0.0, 0.0])  # Rotation de 180° autour de x

        # Normaliser l'axe
        axis = v / v_norm

        # Angle de rotation
        angle = np.arccos(np.clip(np.dot(g_world, a_norm), -1.0, 1.0))

        # Quaternion depuis axe-angle
        half_angle = angle / 2.0
        sin_half = np.sin(half_angle)

        q = np.array([
            np.cos(half_angle),
            axis[0] * sin_half,
            axis[1] * sin_half,
            axis[2] * sin_half
        ])

        return q

    def _slerp(self, q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
        """
        Interpolation sphérique linéaire (SLERP) entre deux quaternions

        Args:
            q1: Premier quaternion
            q2: Second quaternion
            t: Paramètre d'interpolation [0, 1]

        Returns:
            Quaternion interpolé
        """
        # Produit scalaire
        dot = np.dot(q1, q2)

        # Si dot < 0, inverser q2 pour prendre le chemin le plus court
        if dot < 0:
            q2 = -q2
            dot = -dot

        # Clamp pour éviter les erreurs numériques
        dot = np.clip(dot, -1.0, 1.0)

        # Angle entre les quaternions
        theta = np.arccos(dot)

        # Si l'angle est très petit, utiliser l'interpolation linéaire
        if theta < 1e-6:
            return (1 - t) * q1 + t * q2

        # SLERP
        sin_theta = np.sin(theta)
        w1 = np.sin((1 - t) * theta) / sin_theta
        w2 = np.sin(t * theta) / sin_theta

        return w1 * q1 + w2 * q2

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

    def set_alpha_from_time_constant(self, tau: float):
        """
        Ajuste alpha depuis une constante de temps désirée

        Args:
            tau: Constante de temps [s]
        """
        self.tau = tau
        self.alpha = tau / (tau + self.dt)
