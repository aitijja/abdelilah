# Modèles Mathématiques du Capteur MPU6050

## Table des matières

1. [Introduction](#introduction)
2. [Modèle de l'accéléromètre](#modèle-de-laccéléromètre)
3. [Modèle du gyroscope](#modèle-du-gyroscope)
4. [Modélisation du bruit](#modélisation-du-bruit)
5. [Modélisation du biais](#modélisation-du-biais)
6. [Modélisation dynamique](#modélisation-dynamique)
7. [Compensation thermique](#compensation-thermique)
8. [Références](#références)

---

## Introduction

Le MPU6050 est une unité de mesure inertielle (IMU) à 6 degrés de liberté intégrant:
- Un accéléromètre MEMS 3 axes
- Un gyroscope MEMS 3 axes
- Un capteur de température

Ce document présente les modèles mathématiques rigoureux de ces capteurs pour applications en fusion de capteurs et estimation d'état.

---

## Modèle de l'accéléromètre

### 1.1 Principe de mesure

L'accéléromètre mesure l'accélération spécifique **a** (accélération moins la gravité) dans le repère du capteur.

### 1.2 Équation de mesure idéale

L'accélération mesurée dans le repère du capteur {S} est donnée par:

```
a_S = R_WS^T · (a_W - g_W)
```

où:
- **a_S** ∈ ℝ³ : accélération spécifique mesurée dans {S}
- **a_W** ∈ ℝ³ : accélération vraie dans le repère monde {W}
- **g_W** = [0, 0, -g]^T : vecteur gravité (g ≈ 9.81 m/s²)
- **R_WS** ∈ SO(3) : matrice de rotation de {S} vers {W}

### 1.3 Modèle de mesure complet

Le modèle réaliste incluant imperfections est:

```
y_acc(t) = S_a · (a_S(t) + b_a(t) + n_a(t)) + m_a
```

**Forme développée:**

```
y_acc(t) = S_a · R_WS^T(t) · (a_W(t) - g_W) + S_a · b_a(t) + S_a · n_a(t) + m_a
```

où:
- **y_acc** ∈ ℝ³ : mesure brute (sortie ADC convertie)
- **S_a** ∈ ℝ³ˣ³ : matrice de sensibilité et non-orthogonalité
- **b_a** ∈ ℝ³ : vecteur de biais déterministe
- **n_a** ∈ ℝ³ : bruit de mesure blanc gaussien
- **m_a** ∈ ℝ³ : erreur de désalignement (misalignment)

### 1.4 Conversion ADC vers unités physiques

Le MPU6050 fournit des valeurs sur 16 bits. La conversion est:

```
a_i [m/s²] = (ADC_i / S_LSB) · g
```

où S_LSB dépend de la plage configurée:

| Plage | S_LSB [LSB/g] | Résolution [mg/LSB] |
|-------|---------------|---------------------|
| ±2g   | 16384         | 0.061               |
| ±4g   | 8192          | 0.122               |
| ±8g   | 4096          | 0.244               |
| ±16g  | 2048          | 0.488               |

### 1.5 Matrice de sensibilité

La matrice **S_a** capture:
- Erreurs d'échelle (scale factors) sur la diagonale
- Non-orthogonalité des axes (cross-axis sensitivity) hors diagonale

```
S_a = [s_ax    α_yx    α_zx  ]
      [α_xy    s_ay    α_zy  ]
      [α_xz    α_yz    s_az  ]
```

Typiquement:
- Erreur d'échelle: ±2% (s_i ≈ 1 ± 0.02)
- Sensibilité croisée: ±2% (|α_ij| < 0.02)

### 1.6 Modèle simplifié (axes découplés)

Pour des applications moins exigeantes:

```
y_acc,i(t) = s_i · a_S,i(t) + b_a,i(t) + n_a,i(t)
```

pour i ∈ {x, y, z}

---

## Modèle du gyroscope

### 2.1 Principe de mesure

Le gyroscope mesure la vitesse angulaire **ω** du capteur dans son propre repère.

### 2.2 Équation de mesure idéale

La vitesse angulaire dans le repère capteur {S} est:

```
ω_S = R_WS^T · ω_W
```

où:
- **ω_S** ∈ ℝ³ : vitesse angulaire dans {S} [rad/s]
- **ω_W** ∈ ℝ³ : vitesse angulaire dans {W} [rad/s]

### 2.3 Modèle de mesure complet

```
y_gyro(t) = S_g · (ω_S(t) + b_g(t) + n_g(t)) + m_g
```

**Forme développée:**

```
y_gyro(t) = S_g · R_WS^T(t) · ω_W(t) + S_g · b_g(t) + S_g · n_g(t) + m_g
```

où:
- **y_gyro** ∈ ℝ³ : mesure brute du gyroscope
- **S_g** ∈ ℝ³ˣ³ : matrice de sensibilité et g-sensibilité
- **b_g** ∈ ℝ³ : biais variable dans le temps
- **n_g** ∈ ℝ³ : bruit de mesure blanc gaussien
- **m_g** ∈ ℝ³ : erreur de désalignement

### 2.4 Conversion ADC vers unités physiques

```
ω_i [rad/s] = (ADC_i / S_LSB) · (π/180)
```

| Plage      | S_LSB [LSB/(°/s)] | Résolution [(°/s)/LSB] |
|------------|-------------------|------------------------|
| ±250°/s    | 131.0             | 0.00763                |
| ±500°/s    | 65.5              | 0.01527                |
| ±1000°/s   | 32.8              | 0.03049                |
| ±2000°/s   | 16.4              | 0.06098                |

### 2.5 Matrice de sensibilité du gyroscope

```
S_g = [s_gx    α_gyx    α_gzx  ]
      [α_gxy   s_gy     α_gzy  ]
      [α_gxz   α_gyz    s_gz   ]
```

Spécifications typiques:
- Erreur d'échelle: ±2%
- Sensibilité croisée: ±2%
- Sensibilité aux accélérations linéaires: ±0.2 (°/s)/g

### 2.6 G-sensibilité

Le gyroscope est affecté par les accélérations linéaires:

```
ω_measured = ω_true + K_ga · a_S
```

où **K_ga** ∈ ℝ³ˣ³ est la matrice de g-sensibilité (typiquement diagonale avec valeurs ~0.002 (°/s)/g).

---

## Modélisation du bruit

### 3.1 Bruit de l'accéléromètre

Le bruit **n_a(t)** est modélisé comme un processus blanc gaussien:

```
n_a(t) ~ N(0, σ_a² · I₃)
```

**Densité spectrale de puissance (PSD):**

```
PSD_a = σ_a² [m²/s⁴/Hz]
```

Valeurs typiques du MPU6050:
- σ_a ≈ 400 µg/√Hz = 0.0039 m/s²/√Hz
- Variance discrète (à f_s = 1000 Hz): σ_a,d² = σ_a² · f_s

**Variance discrète:**

```
Var(n_a[k]) = σ_a² · f_s = (400 × 10⁻⁶ · 9.81)² · 1000 ≈ 1.51 × 10⁻⁵ (m/s²)²
```

### 3.2 Bruit du gyroscope

Le bruit **n_g(t)** est également blanc gaussien:

```
n_g(t) ~ N(0, σ_g² · I₃)
```

**Densité spectrale de puissance:**

```
PSD_g = σ_g² [(rad/s)²/Hz]
```

Valeurs typiques:
- σ_g ≈ 0.005 °/s/√Hz = 8.73 × 10⁻⁵ rad/s/√Hz
- Variance discrète (à f_s = 1000 Hz): σ_g,d² = σ_g² · f_s

**Variance discrète:**

```
Var(n_g[k]) = σ_g² · f_s = (8.73 × 10⁻⁵)² · 1000 ≈ 7.62 × 10⁻⁹ (rad/s)²
```

### 3.3 Matrice de covariance du bruit de mesure

Pour le filtre de Kalman, la matrice de covariance du bruit de mesure est:

```
R = [σ_a,d² · I₃      0        ]
    [    0        σ_g,d² · I₃  ] ∈ ℝ⁶ˣ⁶
```

### 3.4 Autocorrélation et bande passante

Le bruit blanc a une autocorrélation:

```
R_n(τ) = σ² · δ(τ)
```

La bande passante effective du capteur (filtre passe-bas interne) affecte le spectre du bruit:

```
H(f) = 1 / (1 + j·f/f_c)
```

où f_c est la fréquence de coupure (configurable: 5, 10, 20, 42, 98, 188, 260 Hz).

---

## Modélisation du biais

### 4.1 Biais de l'accéléromètre

Le biais **b_a(t)** évolue lentement et est modélisé comme une marche aléatoire (random walk):

```
db_a(t)/dt = w_a(t)
```

où **w_a(t)** ~ N(0, σ_wa² · I₃) est un bruit blanc de processus.

**Forme discrète:**

```
b_a[k+1] = b_a[k] + w_a[k]
w_a[k] ~ N(0, σ_wa² · Δt · I₃)
```

**Stabilité du biais:**
- Stabilité sur température: ±50 mg typique
- Répétabilité power-on à power-on: ±40 mg typique
- Dérive à long terme: négligeable pour applications courtes

### 4.2 Biais du gyroscope

Le biais du gyroscope est plus critique:

```
db_g(t)/dt = w_g(t)
```

où **w_g(t)** ~ N(0, σ_wg² · I₃).

**Forme discrète:**

```
b_g[k+1] = b_g[k] + w_g[k]
w_g[k] ~ N(0, σ_wg² · Δt · I₃)
```

**Caractéristiques de dérive:**
- Biais initial: ±20 °/s
- Instabilité du biais (bias instability): ~0.01 °/s (Allan variance minimum)
- Dérive angulaire (angle random walk - ARW): ~0.1 °/√s
- Sensibilité thermique: ~0.03 °/s/°C

### 4.3 Variance Allan et caractérisation du biais

La variance d'Allan caractérise la stabilité du biais:

```
σ²_Allan(τ) = (1/2) · E[(Ω(t+τ) - Ω(t))²]
```

où Ω(t) est l'intégration de la vitesse angulaire sur l'intervalle de temps τ.

**Paramètres du bruit identifiés depuis Allan variance:**

Pour le gyroscope:
- Quantization noise: N ~ 0.0001 °/√h
- Angle random walk (ARW): K ~ 0.1 °/√h
- Bias instability: B ~ 0.01 °/h
- Rate random walk (RRW): R ~ 0.001 °/h^(3/2)

**Relation avec les paramètres du modèle:**

```
σ_g² = N²/τ₀
σ_wg² = K²
```

### 4.4 Modèle de biais dépendant de la température

Le biais varie avec la température T:

```
b_g(T) = b_g0 + k_T · (T - T_0) + b_g_drift(t)
```

où:
- b_g0 : biais à température de référence T_0
- k_T : coefficient thermique (~0.03 °/s/°C)
- b_g_drift(t) : dérive résiduelle (random walk)

---

## Modélisation dynamique

### 5.1 Modèle cinématique de l'orientation

L'orientation du capteur est décrite par une matrice de rotation **R(t)** ∈ SO(3) ou un quaternion **q(t)**.

**Équation différentielle avec quaternions:**

```
dq/dt = (1/2) · q ⊗ ω_S
```

où ⊗ est le produit quaternionique et **ω_S** = [0, ω_x, ω_y, ω_z]^T.

**Forme développée:**

```
dq/dt = (1/2) · Ω(ω_S) · q
```

où:

```
Ω(ω_S) = [   0      -ω_x    -ω_y    -ω_z  ]
          [  ω_x       0      ω_z    -ω_y  ]
          [  ω_y     -ω_z      0      ω_x  ]
          [  ω_z      ω_y    -ω_x      0   ]
```

### 5.2 Modèle cinématique de la position

L'accélération spécifique mesurée est reliée à la position par:

```
a_W = d²p/dt² + g_W
a_S = R_WS^T · (d²p/dt² + g_W)
```

**Système d'état complet (9 DDL - position, vitesse, accélération):**

```
dx/dt = [  v  ]
        [  a  ]
        [  0  ]

x = [ p ]  ∈ ℝ⁹
    [ v ]
    [ a ]
```

où:
- **p** ∈ ℝ³ : position
- **v** ∈ ℝ³ : vitesse
- **a** ∈ ℝ³ : accélération

### 5.3 Modèle d'état complet pour filtre de Kalman

**Vecteur d'état (15 DDL):**

```
x = [ p ]   ∈ ℝ¹⁵
    [ v ]
    [ q ]
    [b_g]
```

où:
- **p** ∈ ℝ³ : position
- **v** ∈ ℝ³ : vitesse
- **q** ∈ ℝ⁴ : quaternion d'orientation (||q|| = 1)
- **b_g** ∈ ℝ³ : biais du gyroscope

**Modèle de prédiction (continuous-time):**

```
dp/dt = v
dv/dt = R(q) · a_S - g
dq/dt = (1/2) · Ω(ω_S - b_g) · q
db_g/dt = w_g
```

**Équations de mesure:**

```
y_acc = R^T(q) · (a_W + g) + b_a + n_a
y_gyro = ω_S + b_g + n_g
```

### 5.4 Linéarisation pour EKF (Extended Kalman Filter)

La matrice jacobienne de la fonction d'état f(x,u) par rapport à x est:

```
F = ∂f/∂x = [ 0₃     I₃      0₃ₓ₄        0₃     ]
            [ 0₃     0₃    ∂a/∂q        0₃     ]
            [ 0₄ₓ₃   0₄ₓ₃  ∂q̇/∂q   -(1/2)·Ξ(q) ]
            [ 0₃     0₃      0₃ₓ₄        0₃     ]
```

où:
- ∂a/∂q : jacobien de l'accélération par rapport au quaternion
- ∂q̇/∂q = (1/2)·Ω(ω_S - b_g)
- Ξ(q) : matrice de conversion quaternion-vitesse angulaire

**Matrice jacobienne de mesure pour l'accéléromètre:**

```
H_acc = ∂h_acc/∂x = [ 0₃  0₃  ∂(R^T·g)/∂q  0₃ ]
```

**Matrice jacobienne de mesure pour le gyroscope:**

```
H_gyro = [ 0₃  0₃  0₃ₓ₄  I₃ ]
```

### 5.5 Discrétisation pour implémentation numérique

**Intégration d'Euler (1er ordre):**

```
x[k+1] = x[k] + f(x[k], u[k]) · Δt
```

**Intégration Runge-Kutta 4 (RK4):**

```
k₁ = f(x[k], u[k])
k₂ = f(x[k] + Δt/2·k₁, u[k])
k₃ = f(x[k] + Δt/2·k₂, u[k])
k₄ = f(x[k] + Δt·k₃, u[k])

x[k+1] = x[k] + (Δt/6)·(k₁ + 2k₂ + 2k₃ + k₄)
```

Pour le quaternion, normalisation après chaque intégration:

```
q[k+1] = q[k+1] / ||q[k+1]||
```

---

## Compensation thermique

### 6.1 Mesure de température

Le MPU6050 intègre un capteur de température:

```
T [°C] = (ADC_temp / 340) + 36.53
```

### 6.2 Compensation du biais du gyroscope

**Modèle polynomial:**

```
b_g(T) = b₀ + b₁·T + b₂·T²
```

Les coefficients [b₀, b₁, b₂] sont déterminés par calibration sur plage de température.

**Modèle linéaire simplifié:**

```
b_g(T) = b_ref + k_T · (T - T_ref)
```

où k_T ≈ 0.03 °/s/°C.

### 6.3 Compensation du facteur d'échelle

Le facteur d'échelle varie également avec la température:

```
S(T) = S_ref · [1 + α_T · (T - T_ref)]
```

où α_T est le coefficient thermique du facteur d'échelle (typiquement ±0.02%/°C).

### 6.4 Algorithme de compensation en temps réel

```python
# Pseudo-code
T_current = read_temperature()
ΔT = T_current - T_calibration

# Compensation du biais
b_g_compensated = b_g_calibrated + k_T * ΔT

# Compensation du facteur d'échelle
S_g_compensated = S_g_calibrated * (1 + α_T * ΔT)

# Application
ω_corrected = S_g_compensated^(-1) * (y_gyro - b_g_compensated)
```

---

## Résumé des paramètres du modèle

### Accéléromètre

| Paramètre | Symbole | Valeur typique | Unité |
|-----------|---------|----------------|-------|
| Densité de bruit | σ_a | 400 µg/√Hz | m/s²/√Hz |
| Biais initial | b_a0 | ±50 | mg |
| Stabilité du biais | σ_wa | 0.01 | mg/√Hz |
| Erreur d'échelle | ε_s | ±2 | % |
| Non-orthogonalité | α_ij | ±2 | % |

### Gyroscope

| Paramètre | Symbole | Valeur typique | Unité |
|-----------|---------|----------------|-------|
| Densité de bruit | σ_g | 0.005 °/s/√Hz | rad/s/√Hz |
| Biais initial | b_g0 | ±20 | °/s |
| Instabilité du biais | B | 0.01 | °/s |
| ARW | K | 0.1 | °/√s |
| Coeff. thermique | k_T | 0.03 | °/s/°C |
| Erreur d'échelle | ε_s | ±2 | % |

---

## Références

### Spécifications techniques
1. InvenSense (2013). *MPU-6000 and MPU-6050 Product Specification Revision 3.4*
2. InvenSense (2013). *MPU-6000 and MPU-6050 Register Map and Descriptions Revision 4.2*

### Littérature scientifique
3. Woodman, O. J. (2007). *An introduction to inertial navigation*. University of Cambridge, Computer Laboratory, Tech. Rep. UCAMCL-TR-696.
4. Sola, J. (2017). *Quaternion kinematics for the error-state Kalman filter*. arXiv preprint arXiv:1711.02508.
5. Farhangian, F., & Landry, R. (2020). *Accuracy improvement of attitude determination systems using EKF-based error prediction filter and PI controller*. Sensors, 20(4), 1055.
6. IEEE Std 952-1997. *IEEE Standard Specification Format Guide and Test Procedure for Single-Axis Interferometric Fiber Optic Gyros*.
7. El-Sheimy, N., Hou, H., & Niu, X. (2008). *Analysis and modeling of inertial sensors using Allan variance*. IEEE Transactions on instrumentation and measurement, 57(1), 140-149.

### Algorithmes de fusion
8. Madgwick, S. O. (2010). *An efficient orientation filter for inertial and inertial/magnetic sensor arrays*. Report x-io and University of Bristol (UK), 25, 113-118.
9. Mahony, R., Hamel, T., & Pflimlin, J. M. (2008). *Nonlinear complementary filters on the special orthogonal group*. IEEE Transactions on automatic control, 53(5), 1203-1218.
10. Sabatini, A. M. (2006). *Quaternion-based extended Kalman filter for determining orientation by inertial and magnetic sensing*. IEEE Transactions on Biomedical Engineering, 53(7), 1346-1356.
