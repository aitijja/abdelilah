# Guide d'implémentation

## Table des matières

1. [Installation](#installation)
2. [Utilisation basique](#utilisation-basique)
3. [Configuration du capteur](#configuration-du-capteur)
4. [Calibration](#calibration)
5. [Fusion de capteurs](#fusion-de-capteurs)
6. [Exemples avancés](#exemples-avancés)

---

## Installation

### Prérequis

```bash
# Python 3.8 ou supérieur
python --version

# Dépendances
pip install numpy scipy matplotlib
```

### Structure du projet

```
abdelilah/
├── src/
│   ├── models/
│   │   └── mpu6050_model.py
│   ├── fusion/
│   │   ├── kalman_filter.py
│   │   ├── complementary_filter.py
│   │   └── mahony_filter.py
│   └── utils/
│       └── constants.py
├── examples/
│   ├── basic_simulation.py
│   └── calibration_example.py
└── docs/
    ├── MATHEMATICAL_MODELS.md
    ├── SENSOR_FUSION.md
    ├── CALIBRATION.md
    └── IMPLEMENTATION.md
```

---

## Utilisation basique

### Création d'un modèle de capteur

```python
from src.models.mpu6050_model import MPU6050Model, MPU6050Config, SensorCalibration
import numpy as np

# Configuration du capteur
config = MPU6050Config(
    accel_range=2,        # ±2g
    gyro_range=250,       # ±250°/s
    sample_rate=100.0,    # 100 Hz
    dlpf_cfg=3            # Filtre passe-bas
)

# Paramètres de calibration (optionnel)
calibration = SensorCalibration()
calibration.b_a = np.array([0.05, -0.03, 0.02])  # Biais accéléromètre [m/s²]
calibration.b_g = np.deg2rad([0.5, -0.3, 0.2])   # Biais gyroscope [rad/s]

# Créer le modèle
sensor = MPU6050Model(config, calibration)
```

### Simulation de mesures

```python
# Accélération vraie dans le repère capteur
accel_true = np.array([0, 0, 9.81])  # Statique, +Z up

# Vitesse angulaire vraie
omega_true = np.deg2rad([0, 0, 10])  # 10°/s autour de Z

# Simuler les mesures
accel_meas = sensor.measure_accel(accel_true, temperature=25.0, add_noise=True)
gyro_meas = sensor.measure_gyro(omega_true, temperature=25.0, add_noise=True)

print(f"Accéléromètre: {accel_meas}")
print(f"Gyroscope: {np.rad2deg(gyro_meas)} °/s")
```

### Correction des mesures

```python
# Corriger avec les paramètres de calibration
accel_corrected = sensor.correct_measurement_accel(accel_meas)
gyro_corrected = sensor.correct_measurement_gyro(gyro_meas, temperature=25.0)

print(f"Accéléromètre corrigé: {accel_corrected}")
print(f"Gyroscope corrigé: {np.rad2deg(gyro_corrected)} °/s")
```

---

## Configuration du capteur

### Choix de la plage de mesure

**Accéléromètre:**

```python
# Plage ±2g: Haute résolution, faible plage (mouvements lents)
config_high_res = MPU6050Config(accel_range=2)

# Plage ±16g: Basse résolution, haute plage (chocs, vibrations)
config_high_range = MPU6050Config(accel_range=16)
```

**Gyroscope:**

```python
# Plage ±250°/s: Haute résolution (mouvement lent, précision)
config_slow = MPU6050Config(gyro_range=250)

# Plage ±2000°/s: Faible résolution (rotation rapide)
config_fast = MPU6050Config(gyro_range=2000)
```

### Filtre passe-bas digital (DLPF)

```python
# DLPF off: Bande passante maximale (260 Hz), bruit élevé
config_nofilter = MPU6050Config(dlpf_cfg=0)

# DLPF modéré: 42 Hz (bon compromis)
config_moderate = MPU6050Config(dlpf_cfg=3)

# DLPF fort: 5 Hz (très lissé, latence élevée)
config_smooth = MPU6050Config(dlpf_cfg=6)
```

### Fréquence d'échantillonnage

```python
# Haute fréquence pour contrôle rapide
config_fast = MPU6050Config(sample_rate=1000.0)

# Fréquence standard pour fusion
config_standard = MPU6050Config(sample_rate=100.0)

# Basse fréquence pour économie d'énergie
config_slow = MPU6050Config(sample_rate=10.0)
```

---

## Calibration

### Calibration rapide (statique)

```python
# Placer le capteur immobile sur surface plane (+Z up)
N = 1000
accel_samples = []
gyro_samples = []

for i in range(N):
    # Lire les capteurs
    accel_samples.append(read_accel())
    gyro_samples.append(read_gyro())

# Calculer les biais
b_a = np.mean(accel_samples, axis=0)
b_a[2] -= 9.81  # Compenser la gravité sur Z

b_g = np.mean(gyro_samples, axis=0)

# Appliquer
calibration = SensorCalibration()
calibration.b_a = b_a
calibration.b_g = b_g
```

### Calibration complète (6 positions)

Voir `examples/calibration_example.py` pour l'implémentation complète.

```python
from examples.calibration_example import calibrate_accelerometer_6_positions

# Exécuter la calibration
b_a, S_a = calibrate_accelerometer_6_positions()

# Utiliser les paramètres
calibration = SensorCalibration()
calibration.b_a = b_a
calibration.S_a = S_a
```

---

## Fusion de capteurs

### Filtre de Kalman étendu (EKF)

**Meilleur pour:** Applications critiques nécessitant des covariances

```python
from src.fusion.kalman_filter import QuaternionEKF

# Initialiser
ekf = QuaternionEKF(
    sample_rate=100.0,
    gyro_noise_density=8.73e-5,
    accel_noise_density=3.924e-3
)

# Boucle de mesure
while True:
    gyro = read_gyro()
    accel = read_accel()

    # Prédiction
    ekf.predict(gyro)

    # Mise à jour
    ekf.update(accel)

    # Récupérer l'orientation
    q = ekf.quaternion
    roll, pitch, yaw = ekf.euler_angles
    bias_gyro = ekf.gyro_bias

    print(f"Roll: {np.rad2deg(roll):.2f}°, Pitch: {np.rad2deg(pitch):.2f}°")
```

### Filtre complémentaire

**Meilleur pour:** Systèmes embarqués avec ressources limitées

```python
from src.fusion.complementary_filter import ComplementaryFilter

# Initialiser
comp_filter = ComplementaryFilter(
    alpha=0.98,  # Pondération (0.95-0.99)
    sample_rate=100.0
)

# Ou ajuster depuis une constante de temps
comp_filter.set_alpha_from_time_constant(tau=1.0)  # 1 seconde

# Boucle de mesure
while True:
    gyro = read_gyro()
    accel = read_accel()

    # Mise à jour
    used_accel = comp_filter.update(gyro, accel)

    # Récupérer l'orientation
    q = comp_filter.quaternion
    roll, pitch, yaw = comp_filter.euler_angles

    if not used_accel:
        print("Mouvement dynamique détecté - accéléromètre ignoré")
```

### Filtre de Mahony

**Meilleur pour:** Bon compromis performance/complexité

```python
from src.fusion.mahony_filter import MahonyFilter

# Initialiser
mahony = MahonyFilter(
    kp=2.0,      # Gain proportionnel
    ki=0.1,      # Gain intégral (0 = filtre P pur)
    sample_rate=100.0
)

# Boucle de mesure
while True:
    gyro = read_gyro()
    accel = read_accel()

    # Mise à jour
    mahony.update(gyro, accel)

    # Récupérer l'orientation et biais estimé
    q = mahony.quaternion
    roll, pitch, yaw = mahony.euler_angles
    bias_estimate = mahony.gyro_bias

    # Ajuster les gains si nécessaire
    if high_dynamics:
        mahony.set_gains(kp=5.0, ki=0.0)  # Plus réactif
    else:
        mahony.set_gains(kp=1.0, ki=0.1)  # Plus lissé
```

---

## Exemples avancés

### Simulation de trajectoire dynamique

```python
import numpy as np
from src.models.mpu6050_model import MPU6050Model, MPU6050Config
from src.fusion.kalman_filter import QuaternionEKF

# Configuration
config = MPU6050Config(sample_rate=200.0)
sensor = MPU6050Model(config)
ekf = QuaternionEKF(sample_rate=200.0)

# Générer une trajectoire de rotation
duration = 10.0
n_samples = int(duration * config.sample_rate)
dt = 1.0 / config.sample_rate

# Rotation sinusoïdale
freq = 0.5  # Hz
omega_amplitude = np.deg2rad(45)  # 45°/s

for i in range(n_samples):
    t = i * dt

    # Vitesse angulaire vraie
    omega_true = np.array([
        omega_amplitude * np.sin(2*np.pi*freq*t),
        omega_amplitude * np.cos(2*np.pi*freq*t),
        0
    ])

    # Calculer l'orientation vraie (intégration)
    # ... (omis pour brièveté)

    # Simuler les mesures
    accel_meas = sensor.measure_accel(accel_true)
    gyro_meas = sensor.measure_gyro(omega_true)

    # Fusion
    ekf.predict(gyro_meas)
    ekf.update(accel_meas)

    # Analyser les résultats
    roll, pitch, yaw = ekf.euler_angles
```

### Détection et gestion du mouvement dynamique

```python
def is_static(accel, threshold=0.2):
    """Détecte si le capteur est en mouvement"""
    magnitude = np.linalg.norm(accel)
    return abs(magnitude - 9.81) < threshold

# Dans la boucle
accel = read_accel()
gyro = read_gyro()

if is_static(accel):
    # Utiliser les deux capteurs
    filter.update(gyro, accel)
else:
    # Utiliser uniquement le gyroscope
    filter.predict(gyro)
    # Ne pas faire de update()
```

### Compensation thermique en temps réel

```python
# Charger les coefficients de calibration thermique
thermal_coeffs = load_thermal_calibration()

# Dans la boucle
temperature = read_temperature()
gyro_raw = read_gyro()

# Compenser le biais thermique
delta_T = temperature - thermal_coeffs['T_ref']
b_g_thermal = (thermal_coeffs['b0'] +
               thermal_coeffs['b1'] * delta_T +
               thermal_coeffs['b2'] * delta_T**2)

gyro_compensated = gyro_raw - b_g_thermal
```

### Matrice de rotation pour transformation de coordonnées

```python
# Obtenir la matrice de rotation
R = filter.get_rotation_matrix()

# Transformer un vecteur du repère capteur vers le repère monde
vector_sensor = np.array([1, 0, 0])
vector_world = R @ vector_sensor

# Transformer du repère monde vers le repère capteur
vector_world = np.array([0, 0, 1])
vector_sensor = R.T @ vector_world
```

---

## Optimisation des performances

### Réduction de la charge de calcul

```python
# Utiliser des filtres plus légers
# Au lieu de EKF, utiliser Mahony ou Complémentaire

# Réduire la fréquence de mise à jour
UPDATE_DECIMATION = 2
counter = 0

while True:
    gyro = read_gyro()
    accel = read_accel()

    # Toujours prédire avec le gyroscope
    filter.predict(gyro)

    # Mise à jour plus lente avec accéléromètre
    counter += 1
    if counter % UPDATE_DECIMATION == 0:
        filter.update(accel)
```

### Gestion de la mémoire

```python
# Préallouer les buffers
MAX_SAMPLES = 1000
accel_buffer = np.zeros((MAX_SAMPLES, 3))
gyro_buffer = np.zeros((MAX_SAMPLES, 3))

# Utiliser des vues au lieu de copies
accel_view = accel_buffer[:n_samples]
```

---

## Validation et tests

### Test de précision statique

```python
# Collecter 1000 échantillons statiques
samples = collect_static_samples(n=1000)

# Vérifier la magnitude
magnitudes = [np.linalg.norm(s) for s in samples]
mean_mag = np.mean(magnitudes)
std_mag = np.std(magnitudes)

assert abs(mean_mag - 9.81) < 0.1, "Erreur de calibration"
assert std_mag < 0.05, "Bruit trop élevé"
```

### Test de convergence du filtre

```python
# Vérifier que le filtre converge vers l'orientation vraie
for i in range(1000):
    filter.update(gyro, accel)

roll, pitch, yaw = filter.euler_angles
assert abs(roll - roll_true) < np.deg2rad(2), "Erreur roll > 2°"
assert abs(pitch - pitch_true) < np.deg2rad(2), "Erreur pitch > 2°"
```

---

## Références rapides

### Conversion d'unités

```python
# Degrés ↔ radians
rad = np.deg2rad(degrees)
deg = np.rad2deg(radians)

# g ↔ m/s²
accel_ms2 = accel_g * 9.81
accel_g = accel_ms2 / 9.81

# Quaternion ↔ matrice de rotation
R = quaternion_to_rotation_matrix(q)
q = rotation_matrix_to_quaternion(R)
```

### Constantes importantes

```python
GRAVITY = 9.80665  # m/s²
GYRO_NOISE_DENSITY = 8.73e-5  # rad/s/√Hz
ACCEL_NOISE_DENSITY = 3.924e-3  # m/s²/√Hz
SAMPLE_RATE_TYPICAL = 100.0  # Hz
```

---

Pour des exemples complets, voir:
- `examples/basic_simulation.py` - Simulations et comparaisons
- `examples/calibration_example.py` - Procédures de calibration
