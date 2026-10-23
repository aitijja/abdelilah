"""
Trajectoires 6 DDL pour la simulation dynamique du MPU6050

Une trajectoire décrit l'attitude (angles d'Euler ZYX) et la position du capteur
en fonction du temps. Les grandeurs vraies vues par l'IMU s'en déduisent
analytiquement (voir docs/MATHEMATICAL_MODELS.md, §5.6) :

    ω_S = W(φ, θ) · [φ̇, θ̇, ψ̇]ᵀ        vitesse angulaire (repère capteur)
    f_S = R(q)ᵀ · (p̈_W + g·e_z)        force spécifique (repère capteur)

Conventions identiques aux filtres de src/fusion : quaternion scalaire en tête,
rotation capteur -> monde, axe z du repère monde vers le haut.
"""

import numpy as np
from typing import Optional, Sequence, Tuple

from ..utils.constants import GRAVITY


class Signal:
    """
    Signal scalaire analytique : s(t) = s₀ + ṡ₀·t + Σ Aᵢ·sin(2π·fᵢ·t + φᵢ)

    Les dérivées première et seconde sont calculées exactement, ce qui évite
    toute erreur de différentiation numérique dans les grandeurs vraies.
    """

    def __init__(self,
                 offset: float = 0.0,
                 rate: float = 0.0,
                 sines: Sequence[Tuple[float, float, float]] = ()):
        """
        Args:
            offset: Valeur initiale s₀
            rate: Pente ṡ₀ (unité par seconde)
            sines: Composantes sinusoïdales [(Aᵢ, fᵢ [Hz], φᵢ [rad]), ...]
        """
        self.offset = float(offset)
        self.rate = float(rate)
        sines = np.asarray(sines, dtype=float).reshape(-1, 3)
        self.amplitudes = sines[:, 0]
        self.omegas = 2 * np.pi * sines[:, 1]
        self.phases = sines[:, 2]

    def _arg(self, t: np.ndarray) -> np.ndarray:
        return np.multiply.outer(t, self.omegas) + self.phases

    def value(self, t: np.ndarray) -> np.ndarray:
        """s(t)"""
        t = np.asarray(t, dtype=float)
        return self.offset + self.rate * t + np.sin(self._arg(t)) @ self.amplitudes

    def derivative(self, t: np.ndarray) -> np.ndarray:
        """ds/dt"""
        t = np.asarray(t, dtype=float)
        return self.rate + np.cos(self._arg(t)) @ (self.amplitudes * self.omegas)

    def second_derivative(self, t: np.ndarray) -> np.ndarray:
        """d²s/dt²"""
        t = np.asarray(t, dtype=float)
        return -np.sin(self._arg(t)) @ (self.amplitudes * self.omegas**2)


class Trajectory:
    """
    Trajectoire du capteur : attitude (roll φ, pitch θ, yaw ψ en rad, ZYX)
    et position (x, y, z en m) dans le repère monde

    Exemple (oscillations d'attitude d'un multirotor en vol stationnaire) :
        traj = Trajectory(roll=Signal(sines=[(np.deg2rad(15), 0.5, 0.0)]),
                          yaw=Signal(rate=np.deg2rad(20)),
                          z=Signal(sines=[(0.2, 0.5, 0.0)]))
    """

    def __init__(self,
                 roll: Optional[Signal] = None,
                 pitch: Optional[Signal] = None,
                 yaw: Optional[Signal] = None,
                 x: Optional[Signal] = None,
                 y: Optional[Signal] = None,
                 z: Optional[Signal] = None,
                 gravity: float = GRAVITY):
        """
        Args:
            roll, pitch, yaw: Angles d'Euler ZYX [rad] (None = 0)
            x, y, z: Position dans le repère monde [m] (None = 0)
            gravity: Magnitude de la gravité [m/s²]
        """
        self.angles = [s if s is not None else Signal() for s in (roll, pitch, yaw)]
        self.axes = [s if s is not None else Signal() for s in (x, y, z)]
        self.g = gravity

    def euler(self, t: np.ndarray) -> np.ndarray:
        """Angles d'Euler [φ, θ, ψ] (N, 3) [rad]"""
        return np.stack([s.value(t) for s in self.angles], axis=-1)

    def euler_rates(self, t: np.ndarray) -> np.ndarray:
        """Dérivées des angles d'Euler [φ̇, θ̇, ψ̇] (N, 3) [rad/s]"""
        return np.stack([s.derivative(t) for s in self.angles], axis=-1)

    def position(self, t: np.ndarray) -> np.ndarray:
        """Position p_W (N, 3) [m]"""
        return np.stack([s.value(t) for s in self.axes], axis=-1)

    def velocity(self, t: np.ndarray) -> np.ndarray:
        """Vitesse ṗ_W (N, 3) [m/s]"""
        return np.stack([s.derivative(t) for s in self.axes], axis=-1)

    def acceleration(self, t: np.ndarray) -> np.ndarray:
        """Accélération cinématique p̈_W (N, 3) [m/s²]"""
        return np.stack([s.second_derivative(t) for s in self.axes], axis=-1)

    def quaternion(self, t: np.ndarray) -> np.ndarray:
        """Quaternion d'attitude q = q_z(ψ) ⊗ q_y(θ) ⊗ q_x(φ) (N, 4)"""
        e = self.euler(t) / 2.0
        cr, cp, cy = np.cos(e[..., 0]), np.cos(e[..., 1]), np.cos(e[..., 2])
        sr, sp, sy = np.sin(e[..., 0]), np.sin(e[..., 1]), np.sin(e[..., 2])
        return np.stack([
            cr*cp*cy + sr*sp*sy,
            sr*cp*cy - cr*sp*sy,
            cr*sp*cy + sr*cp*sy,
            cr*cp*sy - sr*sp*cy
        ], axis=-1)

    def rotation_matrix(self, t: np.ndarray) -> np.ndarray:
        """Matrice de rotation capteur -> monde R = R_z(ψ)·R_y(θ)·R_x(φ) (N, 3, 3)"""
        e = self.euler(t)
        cr, cp, cy = np.cos(e[..., 0]), np.cos(e[..., 1]), np.cos(e[..., 2])
        sr, sp, sy = np.sin(e[..., 0]), np.sin(e[..., 1]), np.sin(e[..., 2])
        return np.stack([
            np.stack([cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr], axis=-1),
            np.stack([sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr], axis=-1),
            np.stack([-sp,   cp*sr,            cp*cr], axis=-1)
        ], axis=-2)

    def angular_velocity(self, t: np.ndarray) -> np.ndarray:
        """
        Vitesse angulaire dans le repère capteur ω_S (N, 3) [rad/s]

        ω_x = φ̇ - ψ̇·sin θ
        ω_y = θ̇·cos φ + ψ̇·sin φ·cos θ
        ω_z = -θ̇·sin φ + ψ̇·cos φ·cos θ
        """
        e = self.euler(t)
        d = self.euler_rates(t)
        cr, sr = np.cos(e[..., 0]), np.sin(e[..., 0])
        cp, sp = np.cos(e[..., 1]), np.sin(e[..., 1])
        return np.stack([
            d[..., 0] - d[..., 2] * sp,
            d[..., 1] * cr + d[..., 2] * sr * cp,
            -d[..., 1] * sr + d[..., 2] * cr * cp
        ], axis=-1)

    def specific_force(self, t: np.ndarray) -> np.ndarray:
        """
        Force spécifique mesurée par un accéléromètre idéal (N, 3) [m/s²]

        f_S = R(q)ᵀ · (p̈_W + g·e_z) : au repos et à plat, f_S = [0, 0, g]
        """
        a = self.acceleration(t)
        a[..., 2] += self.g
        R = self.rotation_matrix(t)
        return np.einsum('...ji,...j->...i', R, a)
