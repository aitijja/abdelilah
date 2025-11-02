# Procédures de calibration du MPU6050

## Table des matières

1. [Introduction](#introduction)
2. [Calibration de l'accéléromètre](#calibration-de-laccéléromètre)
3. [Calibration du gyroscope](#calibration-du-gyroscope)
4. [Compensation thermique](#compensation-thermique)
5. [Validation de la calibration](#validation-de-la-calibration)

---

## Introduction

La calibration du MPU6050 est essentielle pour obtenir des mesures précises. Les principales sources d'erreurs à compenser sont:

1. **Biais (offset)**: Décalage constant de la mesure
2. **Erreur d'échelle**: Gain incorrect du capteur
3. **Non-orthogonalité**: Désalignement des axes
4. **Dépendance thermique**: Variation avec la température

---

## Calibration de l'accéléromètre

### 1.1 Calibration du biais (méthode 6 positions)

**Principe:**
Mesurer l'accélération dans 6 positions statiques alignées avec la gravité.

**Positions:**
1. +X vers le haut
2. -X vers le haut
3. +Y vers le haut
4. -Y vers le haut
5. +Z vers le haut
6. -Z vers le haut

**Procédure:**

```python
# Pour chaque position i:
# 1. Placer le capteur dans la position i
# 2. Attendre la stabilisation (1-2 secondes)
# 3. Enregistrer N échantillons (ex: N=1000)
# 4. Calculer la moyenne: a_i_mean = mean(a_i)

# Positions de référence (dans le repère capteur)
g_ref = [
    [+g,  0,  0],  # +X up
    [-g,  0,  0],  # -X up
    [ 0, +g,  0],  # +Y up
    [ 0, -g,  0],  # -Y up
    [ 0,  0, +g],  # +Z up
    [ 0,  0, -g]   # -Z up
]

# Mesures moyennes
a_meas = [a_0_mean, a_1_mean, ..., a_5_mean]
```

**Estimation du biais et du facteur d'échelle:**

Pour chaque axe k ∈ {x, y, z}:

```
# Mesures pour l'axe k dans les positions +k et -k
a_k_plus = a_meas[2*k][k]      # Position +k
a_k_minus = a_meas[2*k+1][k]   # Position -k

# Biais
b_a[k] = (a_k_plus + a_k_minus) / 2

# Facteur d'échelle
s_a[k] = (a_k_plus - a_k_minus) / (2 * g)
```

**Correction des mesures:**

```
a_corrected[k] = (a_raw[k] - b_a[k]) / s_a[k]
```

### 1.2 Calibration complète (avec non-orthogonalité)

**Modèle complet:**

```
a_raw = S_a · a_true + b_a
```

où S_a est la matrice de sensibilité (3×3).

**Estimation par moindres carrés:**

Avec M positions de référence (M ≥ 6):

```
# Matrice de mesures (M × 3)
A = [a_meas_1, a_meas_2, ..., a_meas_M]

# Matrice de références (M × 3)
G = [g_ref_1, g_ref_2, ..., g_ref_M]

# Formulation du problème
# A = G · S_a^T + B (où B répète b_a sur M lignes)

# Résolution
# 1. Estimer S_a et b_a par optimisation
# 2. Utiliser régression linéaire multiple
```

**Solution analytique simplifiée:**

```python
import numpy as np

# Centrer les mesures
A_centered = A - np.mean(A, axis=0)
G_centered = G - np.mean(G, axis=0)

# Estimation de S_a par moindres carrés
S_a = np.linalg.lstsq(G_centered, A_centered, rcond=None)[0].T

# Estimation de b_a
b_a = np.mean(A, axis=0) - S_a @ np.mean(G, axis=0)
```

### 1.3 Méthode de calibration simplifiée (statique sur surface plane)

**Position unique:**
Capteur posé à plat sur surface horizontale (+Z vers le haut).

**Hypothèses:**
- Surface parfaitement horizontale
- Seul le biais est estimé
- Facteur d'échelle supposé nominal

**Procédure:**

```python
# 1. Enregistrer N échantillons (ex: N=1000)
a_samples = [a_1, a_2, ..., a_N]

# 2. Calculer les moyennes
a_mean = np.mean(a_samples, axis=0)

# 3. Biais
b_a[0] = a_mean[0]  # Attendu: 0
b_a[1] = a_mean[1]  # Attendu: 0
b_a[2] = a_mean[2] - g  # Attendu: g
```

**Limitation:** Ne compense pas les erreurs d'échelle ni la non-orthogonalité.

---

## Calibration du gyroscope

### 2.1 Calibration du biais (méthode statique)

**Principe:**
En position statique, le gyroscope doit mesurer zéro. Le biais est la moyenne des mesures.

**Procédure:**

```python
# 1. Placer le capteur immobile sur surface stable
# 2. Attendre la stabilisation thermique (5-10 minutes)
# 3. Enregistrer N échantillons (ex: N=5000 pour 50s à 100Hz)
omega_samples = [omega_1, omega_2, ..., omega_N]

# 4. Calculer le biais
b_g = np.mean(omega_samples, axis=0)

# 5. Écart-type pour validation
sigma_g = np.std(omega_samples, axis=0)
```

**Critère de validation:**

```
sigma_g < 3 * gyro_noise_density * sqrt(sample_rate)
```

Pour MPU6050: σ_g < 3 × 0.005°/s × √100 ≈ 0.15°/s

**Correction:**

```
omega_corrected = omega_raw - b_g
```

### 2.2 Calibration du facteur d'échelle (table tournante)

**Équipement requis:**
Table tournante à vitesse contrôlée (rate table).

**Procédure:**

Pour chaque axe k et vitesse angulaire de référence ω_ref:

```python
# 1. Rotation à ω_ref autour de l'axe k
# 2. Enregistrer les mesures après stabilisation
omega_meas = mean(omega_samples)

# 3. Facteur d'échelle
s_g[k] = omega_meas[k] / omega_ref
```

**Plages de test typiques:**
- ±50°/s, ±100°/s, ±200°/s pour la plage ±250°/s

**Correction:**

```
omega_corrected = (omega_raw - b_g) / s_g
```

### 2.3 Caractérisation de la dérive (variance d'Allan)

**Objectif:**
Identifier les coefficients de bruit pour le modèle stochastique.

**Procédure:**

```python
# 1. Enregistrer un long jeu de données statique (≥ 1 heure)
# 2. Calculer la variance d'Allan pour différents τ

def allan_variance(data, sample_rate, tau_values):
    """
    Calcule la variance d'Allan

    Args:
        data: Série temporelle (N,)
        sample_rate: Fréquence d'échantillonnage [Hz]
        tau_values: Vecteur de temps d'intégration [s]

    Returns:
        allan_dev: Écart-type d'Allan pour chaque τ
    """
    dt = 1.0 / sample_rate
    allan_dev = []

    for tau in tau_values:
        m = int(tau / dt)  # Nombre d'échantillons par cluster
        if m == 0:
            continue

        # Intégrer sur m échantillons
        clusters = []
        for i in range(0, len(data) - m, m):
            cluster_mean = np.mean(data[i:i+m])
            clusters.append(cluster_mean)

        # Variance d'Allan
        if len(clusters) < 2:
            continue
        diff_squared = [(clusters[i+1] - clusters[i])**2
                       for i in range(len(clusters)-1)]
        avar = np.mean(diff_squared) / 2
        allan_dev.append(np.sqrt(avar))

    return np.array(allan_dev)
```

**Interprétation:**

Plot log-log de σ_Allan(τ) vs τ:

```
         |
         |      /  Bias instability (minimum)
σ_Allan  |     /
         |    /
         |___/
         |  /
         | / ARW (pente -1/2)
         |/___________________
              log(τ)
```

**Paramètres extraits:**

- **Angle Random Walk (ARW):** σ(τ) à τ = 1s, pente -1/2
- **Bias Instability:** Minimum de la courbe
- **Rate Random Walk (RRW):** Pente +1/2 à grands τ

---

## Compensation thermique

### 3.1 Caractérisation thermique

**Procédure:**

```python
# 1. Placer le capteur dans une chambre thermique
# 2. Pour chaque température T ∈ [T_min, T_max]:
#    - Stabiliser à température T (30 min)
#    - Mesurer le biais b_g(T) en statique
# 3. Ajuster un modèle polynomial

# Exemple: modèle quadratique
import numpy as np
from scipy.optimize import curve_fit

def bias_model(T, b0, b1, b2):
    return b0 + b1 * T + b2 * T**2

# Ajustement
T_samples = [20, 30, 40, 50, 60]  # °C
bias_samples = [...]  # Mesures correspondantes

params, _ = curve_fit(bias_model, T_samples, bias_samples)
b0, b1, b2 = params
```

**Correction en temps réel:**

```python
# Lire la température
T_current = read_temperature_sensor()

# Compenser le biais
b_g_thermal = bias_model(T_current, b0, b1, b2)
omega_compensated = omega_raw - b_g_thermal
```

### 3.2 Modèle linéaire simplifié

Pour des variations modérées (±20°C):

```
b_g(T) = b_g(T_ref) + k_T · (T - T_ref)
```

où k_T ≈ 0.03°/s/°C pour le MPU6050.

---

## Validation de la calibration

### 4.1 Test de précision de l'accéléromètre

**Test de la magnitude:**

En position statique, la magnitude doit être g:

```python
a_corrected = apply_calibration(a_raw)
magnitude = np.linalg.norm(a_corrected)

# Critère
assert abs(magnitude - g) < 0.1  # Erreur < 0.1 m/s²
```

**Test de répétabilité:**

```python
# Mesurer dans les 6 positions, calculer magnitude
magnitudes = []
for position in positions:
    a_mean = measure_in_position(position)
    a_corrected = apply_calibration(a_mean)
    magnitudes.append(np.linalg.norm(a_corrected))

# Écart-type doit être faible
std_magnitude = np.std(magnitudes)
assert std_magnitude < 0.05  # < 0.05 m/s²
```

### 4.2 Test de stabilité du gyroscope

**Test de biais statique:**

```python
# 10 secondes de mesure statique
omega_samples = measure_static(duration=10)
omega_corrected = omega_samples - b_g

# Moyenne proche de zéro
mean_corrected = np.mean(omega_corrected, axis=0)
assert np.all(np.abs(mean_corrected) < 0.1)  # < 0.1°/s
```

**Test de dérive intégrée:**

```python
# Intégrer pendant 60 secondes
angle_drift = np.cumsum(omega_corrected) * dt

# Dérive finale
final_drift = np.abs(angle_drift[-1])
assert final_drift < 5.0  # < 5° de dérive en 1 min
```

### 4.3 Test de fusion de capteurs

**Test d'orientation statique:**

Utiliser un filtre de fusion (EKF, Mahony, etc.) et vérifier la convergence:

```python
# Initialiser le filtre
filter = MahonyFilter()

# Mesures statiques
for i in range(1000):
    gyro = read_gyro()
    accel = read_accel()
    filter.update(gyro, accel)

# Angles d'Euler finaux
roll, pitch, yaw = filter.euler_angles

# Si le capteur est à plat (+Z up)
assert abs(roll) < 2.0 * np.pi/180   # < 2°
assert abs(pitch) < 2.0 * np.pi/180  # < 2°
# yaw est non observable sans magnétomètre
```

---

## Stockage des paramètres de calibration

### Format JSON recommandé

```json
{
  "calibration_date": "2024-01-15T10:30:00Z",
  "serial_number": "MPU6050-001",
  "temperature_reference": 25.0,
  "accelerometer": {
    "bias": [0.05, -0.03, 0.02],
    "scale_matrix": [
      [1.02, 0.01, 0.01],
      [0.01, 0.98, 0.01],
      [0.01, 0.01, 1.00]
    ],
    "misalignment": [0.0, 0.0, 0.0]
  },
  "gyroscope": {
    "bias": [0.15, -0.10, 0.05],
    "scale_matrix": [
      [1.01, 0.00, 0.00],
      [0.00, 1.00, 0.00],
      [0.00, 0.00, 0.99]
    ],
    "thermal_coefficients": {
      "b0": [0.15, -0.10, 0.05],
      "b1": [0.03, 0.02, 0.03],
      "b2": [0.0, 0.0, 0.0]
    }
  },
  "noise_parameters": {
    "accel_noise_density": 0.003924,
    "gyro_noise_density": 8.73e-5,
    "gyro_bias_instability": 0.01,
    "gyro_arw": 0.1
  }
}
```

---

## Références

1. **Frosio, I., et al.** (2009). *Autocalibration of MEMS accelerometers*. IEEE Transactions on Instrumentation and Measurement, 58(6), 2034-2041.

2. **Tedaldi, D., Pretto, A., & Menegatti, E.** (2014). *A robust and easy to implement method for IMU calibration without external equipments*. IEEE International Conference on Robotics and Automation.

3. **Skog, I., & Handel, P.** (2006). *Calibration of a MEMS inertial measurement unit*. XVII IMEKO World Congress, 1–6.

4. **IEEE Std 952-1997.** *IEEE Standard Specification Format Guide and Test Procedure for Single-Axis Interferometric Fiber Optic Gyros*.
