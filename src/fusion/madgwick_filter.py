"""
Filtre de Madgwick pour estimation d'orientation avec MPU6050
"""

import numpy as np
from typing import Tuple


class MadgwickFilter:
    """
    Filtre de Madgwick (version IMU, sans magnétomètre)

    Corrige l'intégration du gyroscope par une descente de gradient
    sur l'erreur entre gravité prédite et gravité mesurée
    """

    def __init__(self,
                 beta: float = 0.1,
                 sample_rate: float = 100.0,
                 gravity: float = 9.81,
                 adaptive_gain: bool = False,
                 initial_orientation: np.ndarray = None):
        """
        Initialise le filtre de Madgwick

        Args:
            beta: Gain de convergence [rad/s] (typiquement 0.03-0.5)
                  β = √(3/4)·ω̃_β où ω̃_β est l'erreur moyenne du gyroscope
            sample_rate: Fréquence d'échantillonnage [Hz]
            gravity: Magnitude de la gravité [m/s²]
            adaptive_gain: Réduit β quand ||a_m|| s'écarte de g
            initial_orientation: Quaternion initial [q₀, q₁, q₂, q₃]
        """
        self.beta = beta
        self.dt = 1.0 / sample_rate
        self.g = gravity
        self.adaptive_gain = adaptive_gain

        # Quaternion d'orientation
        if initial_orientation is None:
            self.q = np.array([1.0, 0.0, 0.0, 0.0])  # Identité
        else:
            self.q = initial_orientation / np.linalg.norm(initial_orientation)

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
        q0, q1, q2, q3 = self.q

        # 1. Dérivée du quaternion due au gyroscope: q̇_ω = ½ · q ⊗ [0, ω]
        q_dot = 0.5 * self._omega_matrix(gyro_measurement) @ self.q

        # 2. Détection de mouvement dynamique
        accel_norm = np.linalg.norm(accel_measurement)
        use_accel = (self.accel_threshold_low <= accel_norm <= self.accel_threshold_high)

        if use_accel:
            a_norm = accel_measurement / accel_norm

            # 3. Fonction objectif f(q) = R(q)ᵀ·e_z - ã
            f = np.array([
                2 * (q1*q3 - q0*q2) - a_norm[0],
                2 * (q0*q1 + q2*q3) - a_norm[1],
                q0**2 - q1**2 - q2**2 + q3**2 - a_norm[2]
            ])

            # 4. Jacobienne J(q) = ∂f/∂q
            J = np.array([
                [-2*q2,  2*q3, -2*q0,  2*q1],
                [ 2*q1,  2*q0,  2*q3,  2*q2],
                [ 2*q0, -2*q1, -2*q2,  2*q3]
            ])

            # 5. Gradient normalisé
            gradient = J.T @ f
            grad_norm = np.linalg.norm(gradient)

            if grad_norm > 1e-12:
                beta = self.beta
                if self.adaptive_gain:
                    beta *= self._adaptive_factor(accel_norm)

                # 6. Fusion: q̇ = q̇_ω - β · ∇f / ||∇f||
                q_dot = q_dot - beta * gradient / grad_norm

        # 7. Intégration d'Euler et normalisation
        self.q = self.q + q_dot * self.dt
        self.q = self.q / np.linalg.norm(self.q)

        return use_accel

    def _adaptive_factor(self, accel_norm: float) -> float:
        """
        Facteur de pondération de β selon l'écart de ||a_m|| à g

        Vaut 1 quand ||a_m|| = g et décroît linéairement jusqu'à 0
        aux seuils de détection de mouvement.

        Args:
            accel_norm: Norme de l'accélération mesurée [m/s²]

        Returns:
            Facteur dans [0, 1]
        """
        deviation = abs(accel_norm / self.g - 1.0)
        max_deviation = self.accel_threshold_high / self.g - 1.0
        return float(np.clip(1.0 - deviation / max_deviation, 0.0, 1.0))

    def _omega_matrix(self, omega: np.ndarray) -> np.ndarray:
        """
        Construit la matrice Ω(ω) pour la dérivée du quaternion

        Args:
            omega: Vitesse angulaire [ω_x, ω_y, ω_z]

        Returns:
            Matrice Ω (4x4)
        """
        wx, wy, wz = omega[0], omega[1], omega[2]
        return np.array([
            [0,   -wx,  -wy,  -wz],
            [wx,   0,    wz,  -wy],
            [wy,  -wz,   0,    wx],
            [wz,   wy,  -wx,   0]
        ])

    def set_beta(self, beta: float):
        """
        Ajuste le gain de convergence

        Args:
            beta: Nouveau gain [rad/s]
        """
        self.beta = beta

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
