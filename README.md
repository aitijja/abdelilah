# Modèle Mathématique du Capteur MPU6050

## Vue d'ensemble

Ce projet fournit un modèle mathématique complet et rigoureux du capteur inertiel MPU6050 (accéléromètre 3 axes + gyroscope 3 axes) destiné à la recherche scientifique en fusion de capteurs.

## Contenu

### Documentation

- **[MATHEMATICAL_MODELS.md](docs/MATHEMATICAL_MODELS.md)** - Modèles mathématiques complets du MPU6050
  - Équations de mesure de l'accéléromètre
  - Équations de mesure du gyroscope
  - Modélisation du biais (offset)
  - Modélisation du bruit
  - Modélisation dynamique (trajectoires, filtre DLPF, échantillonnage, saturation)

- **[SENSOR_FUSION.md](docs/SENSOR_FUSION.md)** - Algorithmes de fusion de capteurs
  - Filtre de Kalman étendu (EKF)
  - Filtre complémentaire
  - Filtre de Mahony
  - Filtre de Madgwick

- **[CALIBRATION.md](docs/CALIBRATION.md)** - Procédures de calibration
  - Calibration de l'accéléromètre
  - Calibration du gyroscope
  - Compensation du biais

- **[IMPLEMENTATION.md](docs/IMPLEMENTATION.md)** - Guide d'implémentation

### Code Source

- **src/models/** - Implémentations Python des modèles mathématiques (mesure, trajectoires, simulation dynamique)
- **src/fusion/** - Algorithmes de fusion de capteurs (EKF, complémentaire, Mahony, Madgwick)
- **src/utils/** - Utilitaires et constantes
- **matlab/** - `QuaternionEKF.m` (EKF, compatible MATLAB et GNU Octave) et `demo_ekf.m`
- **arduino/MPU6050_Fusion/** - Sketch Arduino temps réel (Mahony ou Madgwick, 250 Hz, sans bibliothèque externe)

### Exemples

- **examples/** - Exemples d'utilisation pratique

## Démarrage rapide

```bash
pip install -r requirements.txt
python examples/basic_simulation.py      # Simulation + comparaison des 4 filtres
python examples/calibration_example.py   # Calibration du gyroscope
python examples/dynamic_simulation.py    # Capteur en mouvement : filtres et vibrations (DLPF)

pip install pytest
python -m pytest tests                   # Tests de convergence (+ parité MATLAB si Octave est installé)
```

MATLAB / Octave :

```matlab
cd matlab
demo_ekf
```

Arduino : ouvrir `arduino/MPU6050_Fusion/MPU6050_Fusion.ino` dans l'IDE Arduino
(câblage SDA→A4, SCL→A5 sur Uno/Nano), téléverser, puis ouvrir le traceur série
à 115200 bauds. Laisser le capteur immobile ~2 s au démarrage (calibration du gyroscope).
`USE_MADGWICK` choisit le filtre. Test hôte des filtres C++ :

```bash
g++ -std=c++11 -O2 -Iarduino/MPU6050_Fusion arduino/test/test_fusion_host.cpp -o test_fusion_host && ./test_fusion_host
```

## Caractéristiques du MPU6050

### Accéléromètre
- **Plages programmables**: ±2g, ±4g, ±8g, ±16g
- **Résolution**: 16 bits
- **Bruit**: ~400 µg/√Hz
- **Sensibilité croisée**: < 2%

### Gyroscope
- **Plages programmables**: ±250°/s, ±500°/s, ±1000°/s, ±2000°/s
- **Résolution**: 16 bits
- **Bruit**: 0.005 °/s/√Hz
- **Dérive de biais**: ±20 °/s

### Filtre passe-bas et échantillonnage
- **DLPF**: 7 réglages, de 260 Hz à 5 Hz de bande passante
- **Cadence de sortie**: 1 kHz / (1 + SMPLRT_DIV) avec DLPF actif

## Applications

- Navigation inertielle
- Estimation d'attitude et d'orientation
- Détection de mouvement
- Stabilisation d'image
- Robotique mobile
- Systèmes embarqués

## Références

Ce travail s'appuie sur les spécifications officielles et la littérature scientifique en fusion de capteurs:

1. InvenSense. (2013). *MPU-6000 and MPU-6050 Product Specification Revision 3.4*
2. InvenSense. (2013). *MPU-6000 and MPU-6050 Register Map and Descriptions Revision 4.2*
3. Woodman, O. J. (2007). *An introduction to inertial navigation*. University of Cambridge.
4. Sola, J. (2017). *Quaternion kinematics for the error-state Kalman filter*. arXiv:1711.02508

## Licence

Ce projet est destiné à la recherche scientifique et à l'enseignement.

## Auteur

Documentation créée pour la recherche en fusion de capteurs inertiels.
