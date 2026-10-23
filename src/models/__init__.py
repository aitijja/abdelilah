"""
Modèles mathématiques des capteurs MPU6050
"""

from .mpu6050_model import (
    MPU6050Model,
    MPU6050Config,
    SensorCalibration,
    DynamicMeasurements,
    create_scale_matrix_with_errors
)
from .trajectory import Signal, Trajectory

__all__ = [
    'MPU6050Model',
    'MPU6050Config',
    'SensorCalibration',
    'DynamicMeasurements',
    'Signal',
    'Trajectory',
    'create_scale_matrix_with_errors'
]
