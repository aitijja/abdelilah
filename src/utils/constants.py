"""
Constantes physiques et paramètres du MPU6050
"""

import numpy as np

# Constantes physiques
GRAVITY = 9.80665  # m/s² (gravité standard)

# Paramètres de conversion ADC de l'accéléromètre
# Sensibilité en LSB/g pour différentes plages
ACCEL_SCALE_FACTOR = {
    2: 16384.0,   # ±2g
    4: 8192.0,    # ±4g
    8: 4096.0,    # ±8g
    16: 2048.0    # ±16g
}

# Résolution en mg/LSB
ACCEL_RESOLUTION = {
    2: 0.061,
    4: 0.122,
    8: 0.244,
    16: 0.488
}

# Paramètres de conversion ADC du gyroscope
# Sensibilité en LSB/(°/s) pour différentes plages
GYRO_SCALE_FACTOR = {
    250: 131.0,    # ±250°/s
    500: 65.5,     # ±500°/s
    1000: 32.8,    # ±1000°/s
    2000: 16.4     # ±2000°/s
}

# Résolution en (°/s)/LSB
GYRO_RESOLUTION = {
    250: 0.00763,
    500: 0.01527,
    1000: 0.03049,
    2000: 0.06098
}

# Conversion température
TEMP_SENSITIVITY = 340.0  # LSB/°C
TEMP_OFFSET = 36.53       # °C

# Paramètres de bruit de l'accéléromètre
ACCEL_NOISE_DENSITY = 400e-6 * GRAVITY  # 400 µg/√Hz en m/s²/√Hz
ACCEL_NOISE_DENSITY_SI = 3.924e-3       # m/s²/√Hz

# Paramètres de bruit du gyroscope
GYRO_NOISE_DENSITY_DEG = 0.005          # °/s/√Hz
GYRO_NOISE_DENSITY_RAD = np.deg2rad(GYRO_NOISE_DENSITY_DEG)  # rad/s/√Hz

# Paramètres de biais de l'accéléromètre
ACCEL_BIAS_INITIAL_MG = 50.0            # mg (typical)
ACCEL_BIAS_INITIAL = ACCEL_BIAS_INITIAL_MG * 1e-3 * GRAVITY  # m/s²
ACCEL_BIAS_STABILITY = 0.01e-3 * GRAVITY  # 0.01 mg/√Hz en m/s²/√Hz

# Paramètres de biais du gyroscope
GYRO_BIAS_INITIAL_DPS = 20.0            # °/s (typical)
GYRO_BIAS_INITIAL = np.deg2rad(GYRO_BIAS_INITIAL_DPS)  # rad/s
GYRO_BIAS_INSTABILITY_DPS = 0.01        # °/s (Allan variance minimum)
GYRO_BIAS_INSTABILITY = np.deg2rad(GYRO_BIAS_INSTABILITY_DPS)  # rad/s

# Angle Random Walk (ARW)
GYRO_ARW_DEG = 0.1                      # °/√s
GYRO_ARW = np.deg2rad(GYRO_ARW_DEG)     # rad/√s

# Rate Random Walk (RRW)
GYRO_RRW_DEG = 0.001                    # °/s/√s
GYRO_RRW = np.deg2rad(GYRO_RRW_DEG)     # rad/s/√s

# Paramètres de température
TEMP_RANGE = (-40, 85)                  # °C (operating range)
TEMP_COEFF_GYRO_DPS = 0.03              # °/s/°C (bias temperature coefficient)
TEMP_COEFF_GYRO = np.deg2rad(TEMP_COEFF_GYRO_DPS)  # rad/s/°C

# Erreurs d'échelle et de non-orthogonalité
SCALE_ERROR_PERCENT = 2.0               # ±2% typical
CROSS_AXIS_SENSITIVITY_PERCENT = 2.0    # ±2% typical

# Sensibilité du gyroscope aux accélérations (g-sensitivity)
GYRO_G_SENSITIVITY_DPS_G = 0.2          # (°/s)/g
GYRO_G_SENSITIVITY = np.deg2rad(GYRO_G_SENSITIVITY_DPS_G)  # (rad/s)/(m/s²)

# Fréquences de coupure du filtre passe-bas (DLPF - Digital Low Pass Filter)
DLPF_CUTOFF_FREQ = {
    0: 260,    # Hz (Accel: 260Hz, Gyro: 256Hz)
    1: 184,    # Hz (Accel: 184Hz, Gyro: 188Hz)
    2: 94,     # Hz (Accel: 94Hz, Gyro: 98Hz)
    3: 44,     # Hz (Accel: 44Hz, Gyro: 42Hz)
    4: 21,     # Hz (Accel: 21Hz, Gyro: 20Hz)
    5: 10,     # Hz (Accel: 10Hz, Gyro: 10Hz)
    6: 5       # Hz (Accel: 5Hz, Gyro: 5Hz)
}

# Fréquences d'échantillonnage typiques
SAMPLE_RATE_HZ = 1000                   # Hz (typical for sensor fusion)
SAMPLE_PERIOD = 1.0 / SAMPLE_RATE_HZ    # s

# Plage ADC
ADC_RANGE = 32768                       # 16-bit signed: -32768 to +32767
ADC_BITS = 16
