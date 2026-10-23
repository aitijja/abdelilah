/*
 * Fusion.h - Filtres d'orientation Mahony et Madgwick pour MPU6050
 *
 * Code C++ sans dépendance Arduino (compilable aussi sur PC pour les tests).
 * Mêmes conventions que les filtres Python de src/fusion :
 *   - quaternion q = [q0 q1 q2 q3], scalaire en tête, repère capteur -> monde
 *   - gyroscope en rad/s, accéléromètre dans n'importe quelle unité (normalisé)
 *   - angles d'Euler ZYX (roll, pitch, yaw) en radians
 */

#ifndef MPU6050_FUSION_H
#define MPU6050_FUSION_H

#include <math.h>

struct Quaternion {
    float q0 = 1.0f, q1 = 0.0f, q2 = 0.0f, q3 = 0.0f;

    void normalize() {
        float n = sqrtf(q0*q0 + q1*q1 + q2*q2 + q3*q3);
        if (n > 0.0f) {
            float inv = 1.0f / n;
            q0 *= inv; q1 *= inv; q2 *= inv; q3 *= inv;
        }
    }

    float roll() const  { return atan2f(2.0f*(q0*q1 + q2*q3), 1.0f - 2.0f*(q1*q1 + q2*q2)); }
    float pitch() const {
        float s = 2.0f*(q0*q2 - q3*q1);
        if (s > 1.0f) s = 1.0f;
        if (s < -1.0f) s = -1.0f;
        return asinf(s);
    }
    float yaw() const   { return atan2f(2.0f*(q0*q3 + q1*q2), 1.0f - 2.0f*(q2*q2 + q3*q3)); }
};

// Accéléromètre accepté si 0.8 g <= ||a|| <= 1.2 g (rapport ||a||² / g²)
static inline bool accelIsStatic(float normSq, float gSq) {
    return normSq >= 0.64f * gSq && normSq <= 1.44f * gSq;
}

// Intégration d'Euler de dq/dt = 0.5 * q ⊗ [0, w]
static inline void integrateGyro(Quaternion &q, float wx, float wy, float wz, float dt) {
    float h = 0.5f * dt;
    float q0 = q.q0, q1 = q.q1, q2 = q.q2, q3 = q.q3;
    q.q0 += h * (-q1*wx - q2*wy - q3*wz);
    q.q1 += h * ( q0*wx + q2*wz - q3*wy);
    q.q2 += h * ( q0*wy - q1*wz + q3*wx);
    q.q3 += h * ( q0*wz + q1*wy - q2*wx);
    q.normalize();
}

/*
 * Filtre de Mahony (correction PI), voir docs/SENSOR_FUSION.md §3
 */
class MahonyAHRS {
public:
    Quaternion q;
    float kp, ki;
    float bx = 0.0f, by = 0.0f, bz = 0.0f;  // Biais estimé [rad/s]

    explicit MahonyAHRS(float kp_ = 2.0f, float ki_ = 0.1f) : kp(kp_), ki(ki_) {}

    // g : magnitude de référence dans l'unité de l'accéléromètre (1.0 en g, 9.81 en m/s²)
    void update(float gx, float gy, float gz,
                float ax, float ay, float az, float dt, float g = 1.0f) {
        float ex = 0.0f, ey = 0.0f, ez = 0.0f;
        float normSq = ax*ax + ay*ay + az*az;

        if (accelIsStatic(normSq, g*g)) {
            float inv = 1.0f / sqrtf(normSq);
            ax *= inv; ay *= inv; az *= inv;

            // Gravité prédite R(q)ᵀ·e_z
            float vx = 2.0f*(q.q1*q.q3 - q.q0*q.q2);
            float vy = 2.0f*(q.q0*q.q1 + q.q2*q.q3);
            float vz = q.q0*q.q0 - q.q1*q.q1 - q.q2*q.q2 + q.q3*q.q3;

            // Erreur e = ã × v
            ex = ay*vz - az*vy;
            ey = az*vx - ax*vz;
            ez = ax*vy - ay*vx;

            if (ki > 0.0f) {
                bx -= ki * ex * dt;
                by -= ki * ey * dt;
                bz -= ki * ez * dt;
            }
        }

        integrateGyro(q, gx - bx + kp*ex, gy - by + kp*ey, gz - bz + kp*ez, dt);
    }
};

/*
 * Filtre de Madgwick (descente de gradient), voir docs/SENSOR_FUSION.md §4
 */
class MadgwickAHRS {
public:
    Quaternion q;
    float beta;

    explicit MadgwickAHRS(float beta_ = 0.1f) : beta(beta_) {}

    void update(float gx, float gy, float gz,
                float ax, float ay, float az, float dt, float g = 1.0f) {
        float q0 = q.q0, q1 = q.q1, q2 = q.q2, q3 = q.q3;

        // q̇_ω = 0.5 * q ⊗ [0, ω]
        float d0 = 0.5f * (-q1*gx - q2*gy - q3*gz);
        float d1 = 0.5f * ( q0*gx + q2*gz - q3*gy);
        float d2 = 0.5f * ( q0*gy - q1*gz + q3*gx);
        float d3 = 0.5f * ( q0*gz + q1*gy - q2*gx);

        float normSq = ax*ax + ay*ay + az*az;
        if (accelIsStatic(normSq, g*g)) {
            float inv = 1.0f / sqrtf(normSq);
            ax *= inv; ay *= inv; az *= inv;

            // f(q) = R(q)ᵀ·e_z - ã
            float f0 = 2.0f*(q1*q3 - q0*q2) - ax;
            float f1 = 2.0f*(q0*q1 + q2*q3) - ay;
            float f2 = q0*q0 - q1*q1 - q2*q2 + q3*q3 - az;

            // ∇f = Jᵀ·f
            float s0 = -2.0f*q2*f0 + 2.0f*q1*f1 + 2.0f*q0*f2;
            float s1 =  2.0f*q3*f0 + 2.0f*q0*f1 - 2.0f*q1*f2;
            float s2 = -2.0f*q0*f0 + 2.0f*q3*f1 - 2.0f*q2*f2;
            float s3 =  2.0f*q1*f0 + 2.0f*q2*f1 + 2.0f*q3*f2;

            float sn = sqrtf(s0*s0 + s1*s1 + s2*s2 + s3*s3);
            if (sn > 1e-9f) {
                float k = beta / sn;
                d0 -= k*s0; d1 -= k*s1; d2 -= k*s2; d3 -= k*s3;
            }
        }

        q.q0 += d0 * dt; q.q1 += d1 * dt; q.q2 += d2 * dt; q.q3 += d3 * dt;
        q.normalize();
    }
};

#endif  // MPU6050_FUSION_H
