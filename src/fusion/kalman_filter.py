"""
Filtre de Kalman étendu (EKF) pour estimation d'orientation avec MPU6050
"""

import numpy as np
from typing import Tuple


class QuaternionEKF:
    """
    Extended Kalman Filter pour estimation d'orientation avec quaternions

    État: x = [q₀, q₁, q₂, q₃, b_gx, b_gy, b_gz]ᵀ
    - q: quaternion d'orientation (4)
    - b_g: biais du gyroscope (3)
    """

    def __init__(self,
                 initial_orientation: np.ndarray = None,
                 gyro_noise_density: float = 8.73e-5,  # rad/s/√Hz
                 gyro_bias_random_walk: float = 8.73e-8,  # rad/s²/√Hz
                 accel_noise_density: float = 3.924e-3,  # m/s²/√Hz
                 sample_rate: float = 100.0,  # Hz
                 gravity: float = 9.81):  # m/s²
        """
        Initialise l'EKF

        Args:
            initial_orientation: Quaternion initial [q₀, q₁, q₂, q₃] (None = identité)
            gyro_noise_density: Densité spectrale de bruit du gyroscope
            gyro_bias_random_walk: Random walk du biais du gyroscope
            accel_noise_density: Densité spectrale de bruit de l'accéléromètre
            sample_rate: Fréquence d'échantillonnage
            gravity: Magnitude de la gravité
        """
        self.dt = 1.0 / sample_rate
        self.g = gravity

        # Vecteur d'état: [q (4), b_g (3)]
        self.x = np.zeros(7)
        if initial_orientation is None:
            self.x[0] = 1.0  # Quaternion identité [1, 0, 0, 0]
        else:
            self.x[:4] = initial_orientation / np.linalg.norm(initial_orientation)

        # Matrice de covariance de l'état
        self.P = np.eye(7)
        self.P[:4, :4] *= 0.1  # Incertitude initiale sur l'orientation
        self.P[4:, 4:] *= 0.01  # Incertitude initiale sur le biais

        # Matrices de bruit
        # Bruit de processus Q
        sigma_n = gyro_noise_density
        sigma_wg = gyro_bias_random_walk

        self.Q = np.zeros((7, 7))
        self.Q[:4, :4] = (sigma_n ** 2) * self.dt * np.eye(4)
        self.Q[4:, 4:] = (sigma_wg ** 2) * self.dt * np.eye(3)

        # Bruit de mesure R (accéléromètre)
        sigma_a = accel_noise_density
        self.R = (sigma_a ** 2) * sample_rate * np.eye(3)

        # Seuils pour détection de mouvement
        self.accel_threshold_low = 0.8 * gravity
        self.accel_threshold_high = 1.2 * gravity

    @property
    def quaternion(self) -> np.ndarray:
        """Retourne le quaternion d'orientation estimé"""
        return self.x[:4]

    @property
    def gyro_bias(self) -> np.ndarray:
        """Retourne le biais du gyroscope estimé"""
        return self.x[4:]

    @property
    def euler_angles(self) -> Tuple[float, float, float]:
        """
        Convertit le quaternion en angles d'Euler (roll, pitch, yaw)

        Returns:
            Tuple (roll, pitch, yaw) en radians
        """
        q = self.quaternion
        q0, q1, q2, q3 = q[0], q[1], q[2], q[3]

        # Roll (φ)
        roll = np.arctan2(2 * (q0*q1 + q2*q3), 1 - 2 * (q1**2 + q2**2))

        # Pitch (θ)
        pitch = np.arcsin(np.clip(2 * (q0*q2 - q3*q1), -1.0, 1.0))

        # Yaw (ψ)
        yaw = np.arctan2(2 * (q0*q3 + q1*q2), 1 - 2 * (q2**2 + q3**2))

        return roll, pitch, yaw

    def predict(self, gyro_measurement: np.ndarray):
        """
        Étape de prédiction avec mesure du gyroscope

        Args:
            gyro_measurement: Mesure du gyroscope [ω_x, ω_y, ω_z] en rad/s
        """
        q = self.x[:4]
        b_g = self.x[4:]

        # Gyroscope corrigé
        omega = gyro_measurement - b_g

        # Intégration RK4 du quaternion
        q_new = self._integrate_quaternion_rk4(q, omega, self.dt)

        # Normalisation
        q_new = q_new / np.linalg.norm(q_new)

        # Mise à jour de l'état
        self.x[:4] = q_new
        # Le biais reste constant: x[4:] = x[4:]

        # Matrice jacobienne F
        F = self._compute_state_jacobian(omega)

        # Matrice de transition
        Phi = np.eye(7) + F * self.dt

        # Prédiction de la covariance
        self.P = Phi @ self.P @ Phi.T + self.Q

    def update(self, accel_measurement: np.ndarray) -> bool:
        """
        Étape de mise à jour avec mesure de l'accéléromètre

        Args:
            accel_measurement: Mesure de l'accéléromètre [a_x, a_y, a_z] en m/s²

        Returns:
            True si la mise à jour a été effectuée, False si rejetée
        """
        # Vérification de mouvement dynamique
        accel_norm = np.linalg.norm(accel_measurement)
        if accel_norm < self.accel_threshold_low or accel_norm > self.accel_threshold_high:
            # Mouvement dynamique détecté, ignorer l'accéléromètre
            return False

        q = self.x[:4]

        # Prédiction de la mesure (gravité dans le repère capteur)
        h_pred = self._predict_gravity_measurement(q)

        # Innovation
        y = accel_measurement - h_pred

        # Matrice jacobienne H
        H = self._compute_measurement_jacobian(q)

        # Matrice de covariance d'innovation
        S = H @ self.P @ H.T + self.R

        # Gain de Kalman
        K = self.P @ H.T @ np.linalg.inv(S)

        # Mise à jour de l'état
        self.x = self.x + K @ y

        # Normalisation du quaternion
        self.x[:4] = self.x[:4] / np.linalg.norm(self.x[:4])

        # Mise à jour de la covariance (forme Joseph pour stabilité numérique)
        I_KH = np.eye(7) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

        return True

    def _integrate_quaternion_rk4(self,
                                  q: np.ndarray,
                                  omega: np.ndarray,
                                  dt: float) -> np.ndarray:
        """
        Intègre le quaternion avec Runge-Kutta 4

        Args:
            q: Quaternion actuel (4,)
            omega: Vitesse angulaire (3,) [rad/s]
            dt: Pas de temps

        Returns:
            Quaternion intégré (4,)
        """
        def qdot(q_val, omega_val):
            """Dérivée du quaternion"""
            Omega = self._omega_matrix(omega_val)
            return 0.5 * Omega @ q_val

        k1 = qdot(q, omega)
        k2 = qdot(q + 0.5 * dt * k1, omega)
        k3 = qdot(q + 0.5 * dt * k2, omega)
        k4 = qdot(q + dt * k3, omega)

        q_new = q + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        return q_new

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

    def _compute_state_jacobian(self, omega: np.ndarray) -> np.ndarray:
        """
        Calcule la matrice jacobienne F = ∂f/∂x

        Args:
            omega: Vitesse angulaire corrigée

        Returns:
            Matrice F (7x7)
        """
        q = self.x[:4]
        F = np.zeros((7, 7))

        # F_qq = (1/2) * Ω(ω)
        F[:4, :4] = 0.5 * self._omega_matrix(omega)

        # F_qb = -(1/2) * Ξ(q)
        F[:4, 4:] = -0.5 * self._xi_matrix(q)

        # F_bb = 0 (déjà initialisé à zéro)

        return F

    def _xi_matrix(self, q: np.ndarray) -> np.ndarray:
        """
        Matrice Ξ(q) pour la dérivée par rapport au biais

        Args:
            q: Quaternion [q₀, q₁, q₂, q₃]

        Returns:
            Matrice Ξ (4x3)
        """
        q0, q1, q2, q3 = q[0], q[1], q[2], q[3]
        return np.array([
            [-q1, -q2, -q3],
            [ q0, -q3,  q2],
            [ q3,  q0, -q1],
            [-q2,  q1,  q0]
        ])

    def _predict_gravity_measurement(self, q: np.ndarray) -> np.ndarray:
        """
        Prédit la mesure de gravité dans le repère capteur

        Args:
            q: Quaternion d'orientation

        Returns:
            Gravité prédite dans le repère capteur (3,)
        """
        q0, q1, q2, q3 = q[0], q[1], q[2], q[3]

        # R(q)ᵀ * [0, 0, g]ᵀ
        h = self.g * np.array([
            2 * (q1*q3 - q0*q2),
            2 * (q0*q1 + q2*q3),
            q0**2 - q1**2 - q2**2 + q3**2
        ])

        return h

    def _compute_measurement_jacobian(self, q: np.ndarray) -> np.ndarray:
        """
        Calcule la matrice jacobienne H = ∂h/∂x

        Args:
            q: Quaternion d'orientation

        Returns:
            Matrice H (3x7)
        """
        q0, q1, q2, q3 = q[0], q[1], q[2], q[3]

        H = np.zeros((3, 7))

        # ∂h/∂q
        H[:, :4] = self.g * np.array([
            [-2*q2,  2*q3, -2*q0,  2*q1],
            [ 2*q1,  2*q0,  2*q3,  2*q2],
            [ 2*q0, -2*q1, -2*q2,  2*q3]
        ])

        # ∂h/∂b_g = 0 (colonnes 4:7 déjà à zéro)

        return H

    def get_rotation_matrix(self) -> np.ndarray:
        """
        Convertit le quaternion en matrice de rotation

        Returns:
            Matrice de rotation R (3x3) du repère capteur vers repère monde
        """
        q = self.quaternion
        q0, q1, q2, q3 = q[0], q[1], q[2], q[3]

        R = np.array([
            [1 - 2*(q2**2 + q3**2), 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2)],
            [2*(q1*q2 + q0*q3), 1 - 2*(q1**2 + q3**2), 2*(q2*q3 - q0*q1)],
            [2*(q1*q3 - q0*q2), 2*(q2*q3 + q0*q1), 1 - 2*(q1**2 + q2**2)]
        ])

        return R
