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
│   │   ├── mpu6050_model.py      # Modèle de mesure + chaîne dynamique (DLPF, ADC)
│   │   └── trajectory.py         # Trajectoires 6 DDL (Signal, Trajectory)
│   ├── fusion/
│   │   ├── kalman_filter.py
│   │   ├── complementary_filter.py
│   │   ├── mahony_filter.py
│   │   └── madgwick_filter.py
│   └── utils/
│       └── constants.py
├── examples/
│   ├── basic_simulation.py
│   ├── calibration_example.py
│   └── dynamic_simulation.py
├── tests/
├── matlab/
├── arduino/
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
accel_meas = sensor.measure_accel(accel_true, add_noise=True)
gyro_meas = sensor.measure_gyro(omega_true, accel=accel_true, add_noise=True)

print(f"Accéléromètre: {accel_meas}")
print(f"Gyroscope: {np.rad2deg(gyro_meas)} °/s")
```

### Correction des mesures

```python
# Corriger avec les paramètres de calibration
accel_corrected = sensor.correct_measurement_accel(accel_meas)
gyro_corrected = sensor.correct_measurement_gyro(gyro_meas)

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

Le DLPF est appliqué par `simulate_dynamic_measurement` (voir [Simulation de trajectoire dynamique](#simulation-de-trajectoire-dynamique)) ; les méthodes `measure_accel` et `measure_gyro` donnent la mesure à un instant, sans filtrage.

### Fréquence d'échantillonnage

```python
# Haute fréquence pour contrôle rapide
config_fast = MPU6050Config(sample_rate=1000.0)

# Fréquence standard pour fusion
config_standard = MPU6050Config(sample_rate=100.0)

# Basse fréquence pour économie d'énergie
config_slow = MPU6050Config(sample_rate=10.0)
```

Pour `simulate_dynamic_measurement`, la fréquence doit valoir 1 kHz / (1 + SMPLRT_DIV) : 1000, 500, 250, 200, 100 Hz…

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

Une `Trajectory` décrit l'attitude (roll, pitch, yaw) et la position du capteur ; chaque composante est un `Signal` analytique `s₀ + ṡ₀·t + Σ A·sin(2πft + φ)`. `simulate_dynamic_measurement` en déduit les grandeurs vraies puis applique toute la chaîne de mesure (DLPF, décimation, saturation, quantification). Détails : `docs/MATHEMATICAL_MODELS.md`, §5.6 à §5.8.

```python
import numpy as np
from src.models import MPU6050Model, MPU6050Config, Signal, Trajectory
from src.fusion import MahonyFilter

# Trajectoire : oscillations d'attitude, lacet continu, déplacement horizontal
traj = Trajectory(
    roll=Signal(sines=[(np.deg2rad(15), 0.5, 0.0)]),    # ±15° à 0.5 Hz
    pitch=Signal(sines=[(np.deg2rad(10), 0.3, 0.5)]),   # ±10° à 0.3 Hz
    yaw=Signal(rate=np.deg2rad(20)),                     # 20 °/s
    x=Signal(sines=[(0.5, 0.25, 0.0)])                   # ±0.5 m à 0.25 Hz
)

# Capteur : 250 Hz (SMPLRT_DIV = 3), DLPF 44 Hz, ±4 g, ±500 °/s
config = MPU6050Config(accel_range=4, gyro_range=500, sample_rate=250.0, dlpf_cfg=3)
sensor = MPU6050Model(config)
data = sensor.simulate_dynamic_measurement(traj, duration=30.0)

# Fusion et comparaison à la vérité terrain
mahony = MahonyFilter(kp=2.0, ki=0.1, sample_rate=config.sample_rate)
errors = []
for k in range(len(data.time)):
    mahony.update(data.gyro[k], data.accel[k])
    roll, pitch, _ = mahony.euler_angles
    errors.append([roll, pitch] - data.euler_true[k, :2])

print("RMSE roll/pitch [°]:", np.rad2deg(np.sqrt(np.mean(np.square(errors), axis=0))))
```

Exemple complet (4 filtres, vibrations moteur) : `examples/dynamic_simulation.py`.

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
