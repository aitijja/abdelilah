// Test hôte de Fusion.h (compilé sur PC, sans Arduino)
//   g++ -std=c++11 -O2 -I../MPU6050_Fusion test_fusion_host.cpp -o test_fusion_host && ./test_fusion_host

#include <cstdio>
#include <random>
#include "Fusion.h"

static const float PI_F = 3.14159265358979f;
static const float DEG = PI_F / 180.0f;

struct Result { float rollErr, pitchErr, rmseRamp; };

template <typename Filter>
Result run(Filter &f) {
    std::mt19937 rng(0);
    std::normal_distribution<float> na(0.0f, 0.004f);       // bruit accel [g]
    std::normal_distribution<float> ng(0.0f, 0.05f * DEG);  // bruit gyro [rad/s]
    const float dt = 1.0f / 250.0f;
    Result r;

    // 1. Convergence statique depuis l'identité vers roll 20°, pitch 25°
    float roll = 20 * DEG, pitch = 25 * DEG;
    for (int k = 0; k < 20 * 250; k++) {
        f.update(ng(rng), ng(rng), ng(rng),
                 -sinf(pitch) + na(rng),
                 sinf(roll) * cosf(pitch) + na(rng),
                 cosf(roll) * cosf(pitch) + na(rng), dt);
    }
    r.rollErr = fabsf(f.q.roll() - roll) / DEG;
    r.pitchErr = fabsf(f.q.pitch() - pitch) / DEG;

    // 2. Suivi d'une rampe de 45° en pitch sur 5 s, depuis le niveau
    f.q = Quaternion();
    const int n = 5 * 250;
    const float rate = 45 * DEG / (n * dt);
    double se = 0;
    for (int k = 0; k < n; k++) {
        float th = rate * k * dt;
        f.update(ng(rng), rate + ng(rng), ng(rng),
                 -sinf(th) + na(rng), na(rng), cosf(th) + na(rng), dt);
        float e = (f.q.pitch() - (th + rate * dt)) / DEG;
        se += e * e;
    }
    r.rmseRamp = sqrtf(se / n);
    return r;
}

int main() {
    MahonyAHRS mahony(2.0f, 0.1f);
    MadgwickAHRS madgwick(0.1f);
    Result a = run(mahony), b = run(madgwick);

    printf("Mahony   : statique roll %.3f° pitch %.3f°, rampe RMSE %.3f°\n", a.rollErr, a.pitchErr, a.rmseRamp);
    printf("Madgwick : statique roll %.3f° pitch %.3f°, rampe RMSE %.3f°\n", b.rollErr, b.pitchErr, b.rmseRamp);

    bool ok = a.rollErr < 1 && a.pitchErr < 1 && a.rmseRamp < 1 &&
              b.rollErr < 1 && b.pitchErr < 1 && b.rmseRamp < 1;
    puts(ok ? "OK" : "ECHEC");
    return ok ? 0 : 1;
}
