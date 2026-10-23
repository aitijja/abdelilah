# Algorithmes de fusion de capteurs pour MPU6050

## Table des matières

1. [Introduction](#introduction)
2. [Filtre de Kalman étendu (EKF)](#filtre-de-kalman-étendu-ekf)
3. [Filtre complémentaire](#filtre-complémentaire)
4. [Filtre de Mahony](#filtre-de-mahony)
5. [Filtre de Madgwick](#filtre-de-madgwick)
6. [Comparaison des algorithmes](#comparaison-des-algorithmes)

---

## Introduction

La fusion de capteurs combine les mesures de l'accéléromètre et du gyroscope pour estimer l'orientation du MPU6050. Chaque capteur a ses forces et faiblesses:

**Accéléromètre:**
- ✓ Pas de dérive à long terme
- ✓ Mesure absolue de l'inclinaison (par rapport à la gravité)
- ✗ Sensible aux accélérations dynamiques
- ✗ Bruyant à court terme

**Gyroscope:**
- ✓ Précis à court terme
- ✓ Insensible aux accélérations linéaires
- ✗ Dérive du biais
- ✗ Intégration = accumulation d'erreurs

La fusion exploite les forces des deux capteurs.

---

## Filtre de Kalman étendu (EKF)

### 1.1 Représentation d'état pour l'orientation

**Vecteur d'état (erreur minimal):**

```
x = [q₀, q₁, q₂, q₃, b_gx, b_gy, b_gz]ᵀ ∈ ℝ⁷
```

où:
- **q** = [q₀, q₁, q₂, q₃]ᵀ : quaternion d'orientation (||q|| = 1)
- **b_g** = [b_gx, b_gy, b_gz]ᵀ : biais du gyroscope [rad/s]

**Contrainte:** q₀² + q₁² + q₂² + q₃² = 1

### 1.2 Modèle de prédiction (continuous-time)

**Équations différentielles:**

```
dq/dt = (1/2) · Ω(ω_m - b_g) · q
db_g/dt = w_g
```

où:
- **ω_m** : mesure du gyroscope [rad/s]
- **w_g** ~ N(0, Q_g) : bruit de processus du biais

**Matrice Ω(ω):**

```
Ω(ω) = [   0     -ω_x   -ω_y   -ω_z ]
       [  ω_x      0     ω_z   -ω_y ]
       [  ω_y    -ω_z     0     ω_x ]
       [  ω_z     ω_y   -ω_x     0  ]
```

### 1.3 Discrétisation (Runge-Kutta 4)

Pour Δt = 1/f_s:

```
q[k+1] = q[k] + (Δt/6) · (k₁ + 2k₂ + 2k₃ + k₄)
b_g[k+1] = b_g[k] + w_g[k]

k₁ = (1/2) · Ω(ω_m[k] - b_g[k]) · q[k]
k₂ = (1/2) · Ω(ω_m[k] - b_g[k]) · (q[k] + Δt/2 · k₁)
k₃ = (1/2) · Ω(ω_m[k] - b_g[k]) · (q[k] + Δt/2 · k₂)
k₄ = (1/2) · Ω(ω_m[k] - b_g[k]) · (q[k] + Δt · k₃)
```

**Normalisation:** q[k+1] ← q[k+1] / ||q[k+1]||

### 1.4 Matrice jacobienne F (linéarisation)

Pour le vecteur d'état x = [q, b_g]ᵀ:

```
F = ∂f/∂x = [  F_qq    F_qb ]
            [   0₃ₓ₄    0₃   ]
```

où:

```
F_qq = (1/2) · Ω(ω_m - b_g)

F_qb = -(1/2) · Ξ(q)
```

**Matrice Ξ(q) (conversion quaternion → vitesse angulaire):**

```
Ξ(q) = [ -q₁   -q₂   -q₃ ]
       [  q₀   -q₃    q₂ ]
       [  q₃    q₀   -q₁ ]
       [ -q₂    q₁    q₀ ]
```

### 1.5 Matrice de transition d'état Φ

Approximation d'Euler:

```
Φ = I₇ + F · Δt
```

ou intégration exacte pour le quaternion.

### 1.6 Prédiction de la covariance

```
P[k+1|k] = Φ · P[k|k] · Φᵀ + Q
```

**Matrice de bruit de processus Q:**

```
Q = [ Q_q      0₄ₓ₃  ]
    [ 0₃ₓ₄    Q_bg   ]
```

où:
- Q_q = σ_n² · Δt · I₄ (bruit du gyroscope propagé au quaternion)
- Q_bg = σ_wg² · Δt · I₃ (random walk du biais)

Typiquement:
```
σ_n = 8.73 × 10⁻⁵ rad/s (bruit du gyroscope)
σ_wg = 8.73 × 10⁻⁸ rad/s² (dérive du biais)
```

### 1.7 Modèle de mesure de l'accéléromètre

L'accéléromètre mesure la gravité dans le repère capteur:

```
y_acc = R(q)ᵀ · g + n_a
```

où:
- **g** = [0, 0, 9.81]ᵀ [m/s²] : gravité dans repère monde
- **n_a** ~ N(0, R_a) : bruit de mesure

**Prédiction de mesure (statique):**

```
ŷ_acc = R(q)ᵀ · g = [ 2(q₁q₃ - q₀q₂)       ]
                    [ 2(q₀q₁ + q₂q₃)       ] · g
                    [ q₀² - q₁² - q₂² + q₃² ]
```

### 1.8 Matrice jacobienne H de mesure

```
H = ∂h/∂x = [ H_q   0₃ₓ₃ ] ∈ ℝ³ˣ⁷
```

où:

```
H_q = ∂(Rᵀ·g)/∂q = g · [ -2q₂   2q₃  -2q₀   2q₁ ]
                        [  2q₁   2q₀   2q₃   2q₂ ]
                        [  2q₀  -2q₁  -2q₂   2q₃ ]
```

### 1.9 Mise à jour (correction)

**Innovation:**

```
y = a_m - ŷ_acc
```

**Matrice de covariance d'innovation:**

```
S = H · P[k+1|k] · Hᵀ + R_a
```

**Gain de Kalman:**

```
K = P[k+1|k] · Hᵀ · S⁻¹
```

**Mise à jour de l'état:**

```
x[k+1|k+1] = x[k+1|k] + K · y
```

**Normalisation du quaternion:**

```
q ← q / ||q||
```

**Mise à jour de la covariance:**

```
P[k+1|k+1] = (I - K·H) · P[k+1|k]
```

### 1.10 Détection de mouvement dynamique

Pour éviter de corriger avec l'accéléromètre pendant le mouvement:

```
||a_m|| ∈ [0.8g, 1.2g] → utiliser l'accéléromètre
sinon → ignorer la mise à jour
```

### 1.11 Conversion quaternion → angles d'Euler

**Roll (φ), Pitch (θ), Yaw (ψ):**

```
φ = atan2(2(q₀q₁ + q₂q₃), 1 - 2(q₁² + q₂²))
θ = asin(2(q₀q₂ - q₃q₁))
ψ = atan2(2(q₀q₃ + q₁q₂), 1 - 2(q₂² + q₃²))
```

---

## Filtre complémentaire

### 2.1 Principe

Le filtre complémentaire combine linéairement les deux sources d'information dans le domaine fréquentiel:

```
Orientation = α · (Gyroscope intégré) + (1-α) · (Accéléromètre)
```

où α est le coefficient de pondération (typiquement 0.95-0.99).

### 2.2 Implémentation avec angles d'Euler

**Prédiction (intégration du gyroscope):**

```
φ_pred[k] = φ[k-1] + ω_x[k] · Δt
θ_pred[k] = θ[k-1] + ω_y[k] · Δt
```

**Mesure de l'accéléromètre:**

```
φ_acc = atan2(a_y, a_z)
θ_acc = atan2(-a_x, √(a_y² + a_z²))
```

**Fusion:**

```
φ[k] = α · φ_pred[k] + (1-α) · φ_acc
θ[k] = α · θ_pred[k] + (1-α) · θ_acc
```

### 2.3 Choix du coefficient α

La constante de temps τ du filtre passe-haut est:

```
τ = α · Δt / (1 - α)
```

Pour τ = 1 s et Δt = 0.01 s:

```
α = τ / (τ + Δt) = 1 / (1 + 0.01) ≈ 0.99
```

### 2.4 Fonction de transfert

En transformée de Laplace:

```
G_gyro(s) = τ·s / (τ·s + 1)         (passe-haut)
G_accel(s) = 1 / (τ·s + 1)          (passe-bas)
```

Vérification: G_gyro(s) + G_accel(s) = 1 (complémentaires)

### 2.5 Limitations

- Problème de blocage gimbal (gimbal lock) avec angles d'Euler à θ = ±90°
- Pas d'estimation du biais du gyroscope
- Pas optimal statistiquement

### 2.6 Filtre complémentaire avec quaternions

**Prédiction:**

```
q_pred = q · exp((Δt/2) · ω)
```

où exp est l'exponentielle quaternionique.

**Correction avec accéléromètre (inclinaison uniquement):**

Avec ĥ = R(q_pred)ᵀ · e_z la gravité prédite et ã_m = a_m / ||a_m|| la mesure normalisée,
on cherche la rotation δq (repère capteur) telle que R(q_pred ⊗ δq)ᵀ · e_z = ã_m,
c'est-à-dire la rotation de plus court chemin de ã_m vers ĥ :

```
δq = quat_between(ã_m, ĥ)        axe ∝ ã_m × ĥ,  angle = arccos(ã_m · ĥ)
δq_α = SLERP(1, δq, 1 - α)
q = q_pred ⊗ δq_α
```

L'axe de δq est orthogonal à la gravité : seule l'inclinaison (roll/pitch) est
corrigée et le lacet reste celui intégré par le gyroscope. Un SLERP direct vers
un quaternion « absolu » calculé depuis l'accéléromètre ramènerait au contraire
le lacet vers une valeur arbitraire.

---

## Filtre de Mahony

### 3.1 Principe

Le filtre de Mahony utilise une rétroaction proportionnelle-intégrale (PI) pour corriger la dérive du gyroscope.

### 3.2 Erreur d'orientation

**Vecteur d'erreur (produit vectoriel):**

```
e = ã_m × (Rᵀ · g)
```

où ã_m = a_m / ||a_m|| (accélération normalisée).

### 3.3 Correction PI

**Gyroscope corrigé:**

```
ω_corrected = ω_m - b_g + K_P · e + K_I · ∫e dt
```

**Mise à jour du biais (intégrale):**

```
b_g[k+1] = b_g[k] - K_I · e[k] · Δt
```

**Paramètres typiques:**
- K_P = 1.0 à 5.0 (gain proportionnel)
- K_I = 0.0 à 0.5 (gain intégral, 0 = filtre proportionnel pur)

### 3.4 Intégration du quaternion

```
dq/dt = (1/2) · Ω(ω_corrected) · q
```

Discrétisation:

```
q[k+1] = q[k] + (Δt/2) · Ω(ω_corrected[k]) · q[k]
q[k+1] ← q[k+1] / ||q[k+1]||
```

### 3.5 Algorithme complet

```
1. Normaliser a_m → ã_m
2. Prédire gravité: ĝ = R(q)ᵀ · [0, 0, g]ᵀ
3. Calculer erreur: e = ã_m × ĝ
4. Correction PI: ω_corr = ω_m - b_g + K_P·e + K_I·∫e
5. Intégrer quaternion: q[k+1] = integrate(q[k], ω_corr, Δt)
6. Normaliser: q ← q / ||q||
7. Mettre à jour biais: b_g[k+1] = b_g[k] - K_I·e·Δt
```

---

## Filtre de Madgwick

### 4.1 Principe

Le filtre de Madgwick utilise une descente de gradient pour minimiser l'erreur entre la gravité mesurée et prédite.

### 4.2 Fonction objectif

Minimiser:

```
f(q) = R(q)ᵀ · g - ã_m
```

### 4.3 Gradient de la fonction objectif

```
∇f(q) = Jᵀ(q) · f(q)
```

où J(q) est la matrice jacobienne:

```
J(q) = ∂f/∂q = g · [ -2q₂   2q₃  -2q₀   2q₁ ]
                    [  2q₁   2q₀   2q₃   2q₂ ]
                    [  2q₀  -2q₁  -2q₂   2q₃ ]
```

### 4.4 Mise à jour du quaternion

**Prédiction gyroscope:**

```
q̇_ω = (1/2) · Ω(ω_m) · q
```

**Correction gradient:**

```
q̇_∇f = -β · ∇f(q) / ||∇f(q)||
```

où β est le gain de convergence (typiquement 0.1 à 1.0).

**Fusion:**

```
q̇ = q̇_ω + q̇_∇f
```

**Intégration:**

```
q[k+1] = q[k] + q̇ · Δt
q[k+1] ← q[k+1] / ||q[k+1]||
```

### 4.5 Ajustement adaptatif du gain β

Pour réduire l'influence de l'accéléromètre pendant les mouvements, β est pondéré
par l'écart de ||a_m|| à g, normalisé par le seuil de détection de mouvement (ici 20 %) :

```
β_adaptive = β₀ · max(0, 1 - |‖a_m‖/g - 1| / 0.2)
```

β_adaptive vaut β₀ au repos et s'annule quand ||a_m|| atteint 0.8 g ou 1.2 g
(option `adaptive_gain=True` de `MadgwickFilter`).

**Choix de β:** Madgwick propose β = √(3/4) · ω̃_β où ω̃_β est l'erreur moyenne
du gyroscope (biais résiduel). Avec un biais résiduel de ~5 °/s, β ≈ 0.08 rad/s.

### 4.6 Version avec magnétomètre

Le MPU6050 n'a pas de magnétomètre, mais si disponible:

```
f_mag(q) = R(q)ᵀ · b - m̃
```

où **b** est le champ magnétique de référence et **m̃** la mesure normalisée.

---

## Comparaison des algorithmes

### 5.1 Tableau comparatif

| Critère | EKF | Complémentaire | Mahony | Madgwick |
|---------|-----|----------------|--------|----------|
| **Complexité** | Élevée | Faible | Moyenne | Moyenne |
| **Charge calcul** | Élevée | Faible | Moyenne | Moyenne |
| **Optimalité** | Optimal (MMSE) | Sous-optimal | Sous-optimal | Sous-optimal |
| **Estimation biais** | ✓ | ✗ | ✓ | ✗ |
| **Covariance** | ✓ | ✗ | ✗ | ✗ |
| **Tuning** | Q, R | α | K_P, K_I | β |
| **Robustesse** | Élevée | Moyenne | Élevée | Élevée |
| **Temps réel** | Possible | Excellent | Excellent | Excellent |

### 5.2 Fréquence d'exécution typique

| Algorithme | f_min [Hz] | f_typ [Hz] | f_max [Hz] |
|------------|------------|------------|------------|
| EKF        | 50         | 100-200    | 1000       |
| Complémentaire | 50     | 100-500    | 2000       |
| Mahony     | 50         | 100-500    | 2000       |
| Madgwick   | 50         | 100-500    | 2000       |

### 5.3 Précision typique (RMS)

| Algorithme | Roll/Pitch [°] | Yaw [°] | Commentaires |
|------------|----------------|---------|--------------|
| EKF        | 0.5-1.0        | N/A*    | Optimal, coûteux |
| Complémentaire | 1.0-2.0    | N/A*    | Simple, rapide |
| Mahony     | 0.5-1.5        | N/A*    | Bon compromis |
| Madgwick   | 0.5-1.5        | N/A*    | Populaire, efficace |

*Le yaw n'est pas observable sans magnétomètre (dérive libre)

### 5.4 Recommandations d'utilisation

**Utiliser EKF si:**
- Besoin de covariance pour fusion ultérieure
- Estimation du biais critique
- Ressources de calcul disponibles
- Application critique (navigation, contrôle)

**Utiliser Complémentaire si:**
- Ressources très limitées (microcontrôleur 8-bit)
- Application simple (niveau à bulle électronique)
- Pas besoin d'estimation du biais

**Utiliser Mahony si:**
- Bon compromis performance/complexité
- Besoin d'estimation du biais
- Temps réel strict avec ressources moyennes

**Utiliser Madgwick si:**
- Bonne performance requise
- Pas besoin de covariance
- Code ouvert populaire souhaité

### 5.5 Implémentations du dépôt

| Algorithme | Python | MATLAB / Octave | Arduino (C++) |
|------------|--------|-----------------|---------------|
| EKF | `src/fusion/kalman_filter.py` | `matlab/QuaternionEKF.m` | — |
| Complémentaire | `src/fusion/complementary_filter.py` | — | — |
| Mahony | `src/fusion/mahony_filter.py` | — | `arduino/MPU6050_Fusion/Fusion.h` |
| Madgwick | `src/fusion/madgwick_filter.py` | — | `arduino/MPU6050_Fusion/Fusion.h` |

`matlab/QuaternionEKF.m` reproduit la version Python à 1e-9 près
(`tests/test_matlab_parity.py`). Le sketch Arduino tourne à 250 Hz sur un
ATmega328P (≈13 Ko de flash) et lit le MPU6050 directement via `Wire`.

Résultats en simulation (`examples/basic_simulation.py`, rampe de 45° en pitch
sur 5 s à 100 Hz) : RMSE pitch de 0.03 à 0.15° pour les quatre filtres.

**En mouvement** (`examples/dynamic_simulation.py`, 250 Hz, DLPF 44 Hz, mêmes
réglages que ci-dessus, RMSE roll après 2 s de convergence) :

| Scénario | EKF | Complémentaire | Mahony | Madgwick |
|----------|-----|----------------|--------|----------|
| Rotation seule (±15°, lacet 20 °/s) | 0.15° | 0.19° | 0.14° | 0.12° |
| Rotation + accélérations linéaires (jusqu'à 0.2 g) | 14.5° | 4.6° | 3.7° | 4.0° |

Les accélérations linéaires faussent la direction de gravité vue par
l'accéléromètre : les filtres complémentaire, Mahony et Madgwick perdent un
facteur 25 à 30. L'EKF, dont la matrice R ne représente que le bruit du capteur,
attribue ces accélérations au biais du gyroscope (biais estimé jusqu'à 388 °/s)
et diverge. Multiplier R par 100 ramène l'erreur à 3.5° mais le biais dérive
encore (74 °/s) ; sans estimation du biais, l'EKF descend à 0.6–1.2°. Une
correction de fond (EKF à état d'erreur, R adaptatif) reste à faire.

Vibrations moteur à 150 Hz (0.5 g) : avec DLPF_CFG = 0 (260 Hz), le résidu
vibratoire atteint 0.34 g et 80 % des mesures de l'accéléromètre sont rejetées
par la détection de mouvement ; avec DLPF_CFG = 3 (44 Hz), il tombe à 0.03 g et
toutes les mesures sont utilisées.

---

## Références

### Articles fondateurs

1. **Kalman, R. E.** (1960). *A new approach to linear filtering and prediction problems*. Journal of Basic Engineering, 82(1), 35-45.

2. **Welch, G., & Bishop, G.** (2006). *An introduction to the Kalman filter*. University of North Carolina at Chapel Hill, Chapel Hill, NC.

3. **Mahony, R., Hamel, T., & Pflimlin, J. M.** (2008). *Nonlinear complementary filters on the special orthogonal group*. IEEE Transactions on Automatic Control, 53(5), 1203-1218.

4. **Madgwick, S. O., Harrison, A. J., & Vaidyanathan, R.** (2011). *Estimation of IMU and MARG orientation using a gradient descent algorithm*. IEEE International Conference on Rehabilitation Robotics.

### Applications MPU6050

5. **Colton, S.** (2007). *The balance filter: A simple solution for integrating accelerometer and gyroscope measurements for a balancing platform*. MIT.

6. **Valenti, R. G., Dryanovski, I., & Xiao, J.** (2015). *Keeping a good attitude: A quaternion-based orientation filter for IMUs and MARGs*. Sensors, 15(8), 19302-19330.

7. **Sabatini, A. M.** (2006). *Quaternion-based extended Kalman filter for determining orientation by inertial and magnetic sensing*. IEEE Transactions on Biomedical Engineering, 53(7), 1346-1356.

### Livres de référence

8. **Farrell, J. A.** (2008). *Aided navigation: GPS with high rate sensors*. McGraw-Hill.

9. **Groves, P. D.** (2013). *Principles of GNSS, inertial, and multisensor integrated navigation systems* (2nd ed.). Artech House.

10. **Titterton, D., & Weston, J. L.** (2004). *Strapdown inertial navigation technology* (2nd ed.). IET.
