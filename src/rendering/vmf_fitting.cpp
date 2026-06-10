#include "vmf_fitting.h"
#include <algorithm>
#include <cmath>

namespace spectra {

namespace vmf_fitting {

static constexpr float PI = 3.14159265358979323846f;
static constexpr float INV_4PI = 1.0f / (4.0f * PI);

float logLikelihoodSingleLobe(float sumX, float sumY, float sumZ, float sumW)
{
    if (sumW < 1e-9f) return 0.0f;
    float R = std::sqrt(sumX*sumX + sumY*sumY + sumZ*sumZ);
    float Rbar = std::min(R / sumW, 0.9999f);

    // Estimate kappa from R_bar (Banerjee et al. / Sra approximation) —
    // identical to the device refit kernel so host decisions match the
    // device's fitted lobes.
    float denom = std::max(1.0f - Rbar*Rbar, 0.01f);
    float kappa = (Rbar * (3.0f - Rbar*Rbar)) / denom;
    kappa = std::min(kappa, 300.0f);

    // C3(kappa) = kappa / (4*pi*sinh(kappa))
    // log(C3) = log(kappa) - log(4*pi) - log(sinh(kappa))
    // For numerical stability: log(sinh(kappa)) = kappa + log(1 - exp(-2*kappa)) - log(2)
    // For large kappa: log(sinh(kappa)) ≈ kappa - log(2)
    float logC3;
    if (kappa > 20.0f) {
        logC3 = std::log(kappa) - std::log(4.0f * PI) - kappa + std::log(2.0f);
    } else if (kappa > 1e-6f) {
        logC3 = std::log(kappa) - std::log(4.0f * PI) - std::log(std::sinh(kappa));
    } else {
        logC3 = std::log(INV_4PI);  // uniform: C3 → 1/(4π)
    }

    // LL = N * log(C3(kappa)) + kappa * R
    return sumW * logC3 + kappa * R;
}

} // namespace vmf_fitting
} // namespace spectra
