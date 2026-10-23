/*
 * MPU6050_Fusion.ino - Estimation d'attitude temps réel avec un MPU6050
 *
 * Câblage (Arduino Uno/Nano) : VCC -> 5V, GND -> GND, SDA -> A4, SCL -> A5
 *                              (AD0 à GND : adresse 0x68)
 *
 * Au démarrage, laisser le capteur immobile ~2 s (calibration du gyroscope).
 * Sortie série (115200 bauds) : roll, pitch, yaw en degrés, une ligne par
 * période d'affichage, lisible avec le traceur série de l'IDE Arduino.
 *
 * Aucune bibliothèque externe : accès direct aux registres via Wire.
 */

#include <Wire.h>
#include "Fusion.h"

// ---------------------------------------------------------------- Configuration
#define USE_MADGWICK 0          // 0 = Mahony (estime le biais), 1 = Madgwick

const uint8_t  MPU_ADDR        = 0x68;
const float    LOOP_HZ         = 250.0f;  // = 1 kHz / (1 + SMPLRT_DIV)
const uint8_t  SMPLRT_DIV      = 3;
const uint8_t  DLPF_CFG        = 3;       // Filtre passe-bas ~44 Hz
const uint8_t  GYRO_FS_SEL     = 1;       // 0:±250 1:±500 2:±1000 3:±2000 °/s
const uint8_t  ACCEL_FS_SEL    = 1;       // 0:±2g  1:±4g  2:±8g   3:±16g
const uint16_t CALIB_SAMPLES   = 500;
const uint8_t  PRINT_DIVIDER   = 10;      // Affiche 1 échantillon sur 10 (25 Hz)

// Registres (MPU-6000/6050 Register Map rev 4.2)
const uint8_t REG_SMPLRT_DIV   = 0x19;
const uint8_t REG_CONFIG       = 0x1A;
const uint8_t REG_GYRO_CONFIG  = 0x1B;
const uint8_t REG_ACCEL_CONFIG = 0x1C;
const uint8_t REG_ACCEL_XOUT_H = 0x3B;
const uint8_t REG_PWR_MGMT_1   = 0x6B;
const uint8_t REG_WHO_AM_I     = 0x75;

// Sensibilités [LSB/g] et [LSB/(°/s)]
const float ACCEL_LSB[] = {16384.0f, 8192.0f, 4096.0f, 2048.0f};
const float GYRO_LSB[]  = {131.0f, 65.5f, 32.8f, 16.4f};

const float DEG2RAD = 0.017453292519943295f;
const float RAD2DEG = 57.29577951308232f;

// ---------------------------------------------------------------- État
#if USE_MADGWICK
MadgwickAHRS filter(0.1f);
#else
MahonyAHRS filter(2.0f, 0.1f);
#endif

float gyroBias[3] = {0.0f, 0.0f, 0.0f};  // [rad/s], mesuré au démarrage
float accelScale, gyroScale;
uint32_t lastMicros;
uint8_t printCount = 0;

// ---------------------------------------------------------------- I2C
void writeReg(uint8_t reg, uint8_t value) {
    Wire.beginTransmission(MPU_ADDR);
    Wire.write(reg);
    Wire.write(value);
    Wire.endTransmission();
}

uint8_t readReg(uint8_t reg) {
    Wire.beginTransmission(MPU_ADDR);
    Wire.write(reg);
    Wire.endTransmission(false);
    Wire.requestFrom(MPU_ADDR, (uint8_t)1);
    return Wire.available() ? Wire.read() : 0xFF;
}

// Lit accéléromètre [g] et gyroscope [rad/s] en une seule rafale de 14 octets
bool readImu(float accel[3], float gyro[3]) {
    Wire.beginTransmission(MPU_ADDR);
    Wire.write(REG_ACCEL_XOUT_H);
    if (Wire.endTransmission(false) != 0) return false;
    if (Wire.requestFrom(MPU_ADDR, (uint8_t)14) != 14) return false;

    int16_t raw[7];
    for (uint8_t i = 0; i < 7; i++) {
        raw[i] = (int16_t)((Wire.read() << 8) | Wire.read());
    }
    // raw[3] = température, non utilisée ici
    for (uint8_t i = 0; i < 3; i++) {
        accel[i] = raw[i] * accelScale;
        gyro[i]  = raw[i + 4] * gyroScale;
    }
    return true;
}

// ---------------------------------------------------------------- Initialisation
bool initMpu() {
    if (readReg(REG_WHO_AM_I) != 0x68) return false;

    writeReg(REG_PWR_MGMT_1, 0x01);               // Réveil, horloge PLL gyro X
    delay(100);
    writeReg(REG_SMPLRT_DIV, SMPLRT_DIV);
    writeReg(REG_CONFIG, DLPF_CFG);
    writeReg(REG_GYRO_CONFIG, GYRO_FS_SEL << 3);
    writeReg(REG_ACCEL_CONFIG, ACCEL_FS_SEL << 3);

    accelScale = 1.0f / ACCEL_LSB[ACCEL_FS_SEL];
    gyroScale  = DEG2RAD / GYRO_LSB[GYRO_FS_SEL];
    return true;
}

// Biais statique du gyroscope + attitude initiale depuis l'accéléromètre
void calibrate() {
    float accel[3], gyro[3];
    float sumA[3] = {0, 0, 0}, sumG[3] = {0, 0, 0};
    uint16_t n = 0;

    while (n < CALIB_SAMPLES) {
        if (readImu(accel, gyro)) {
            for (uint8_t i = 0; i < 3; i++) { sumA[i] += accel[i]; sumG[i] += gyro[i]; }
            n++;
        }
        delayMicroseconds((uint32_t)(1e6f / LOOP_HZ));
    }
    for (uint8_t i = 0; i < 3; i++) {
        gyroBias[i] = sumG[i] / n;
        sumA[i] /= n;
    }

    // Initialiser q sur l'inclinaison mesurée (évite le transitoire de convergence)
    float roll  = atan2f(sumA[1], sumA[2]);
    float pitch = atan2f(-sumA[0], sqrtf(sumA[1]*sumA[1] + sumA[2]*sumA[2]));
    float cr = cosf(roll / 2), sr = sinf(roll / 2);
    float cp = cosf(pitch / 2), sp = sinf(pitch / 2);
    filter.q.q0 = cr * cp;
    filter.q.q1 = sr * cp;
    filter.q.q2 = cr * sp;
    filter.q.q3 = -sr * sp;
}

void setup() {
    Serial.begin(115200);
    Wire.begin();
    Wire.setClock(400000);

    if (!initMpu()) {
        Serial.println(F("MPU6050 introuvable (verifier cablage / adresse)"));
        while (true) {}
    }

    Serial.println(F("Calibration : ne pas bouger le capteur..."));
    calibrate();
    Serial.print(F("Biais gyro [deg/s] : "));
    for (uint8_t i = 0; i < 3; i++) { Serial.print(gyroBias[i] * RAD2DEG, 3); Serial.print(' '); }
    Serial.println();
    Serial.println(F("roll pitch yaw"));

    lastMicros = micros();
}

// ---------------------------------------------------------------- Boucle
void loop() {
    const uint32_t period = (uint32_t)(1e6f / LOOP_HZ);
    uint32_t now = micros();
    if ((uint32_t)(now - lastMicros) < period) return;

    float dt = (now - lastMicros) * 1e-6f;   // dt réel (robuste aux retards)
    lastMicros = now;

    float accel[3], gyro[3];
    if (!readImu(accel, gyro)) return;

    filter.update(gyro[0] - gyroBias[0], gyro[1] - gyroBias[1], gyro[2] - gyroBias[2],
                  accel[0], accel[1], accel[2], dt);

    if (++printCount >= PRINT_DIVIDER) {
        printCount = 0;
        Serial.print(filter.q.roll() * RAD2DEG, 2);  Serial.print(' ');
        Serial.print(filter.q.pitch() * RAD2DEG, 2); Serial.print(' ');
        Serial.println(filter.q.yaw() * RAD2DEG, 2);
    }
}
