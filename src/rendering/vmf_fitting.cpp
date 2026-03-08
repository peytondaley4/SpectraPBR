#include "vmf_fitting.h"
#include <algorithm>
#include <cmath>

namespace spectra {

namespace vmf_fitting {

static constexpr float PI = 3.14159265358979323846f;
static constexpr float TWO_PI = 6.28318530718f;
static constexpr float INV_4PI = 1.0f / (4.0f * PI);

bool fitFromSums(float sumX, float sumY, float sumZ, float sumW,
                 float& out_theta, float& out_phi, float& out_kappa)
{
    if (sumW < 1e-9f) return false;
    float len = std::sqrt(sumX*sumX + sumY*sumY + sumZ*sumZ);
    if (len < 1e-9f) return false;

    float nx = sumX / len;
    float ny = sumY / len;
    float nz = sumZ / len;

    float Rbar = len / sumW;
    Rbar = std::min(Rbar, 0.9999f);

    // Sra (2012): kappa from R_bar for vMF
    float denom = std::max(1.0f - Rbar*Rbar, 0.01f);
    float kappa = (Rbar * (3.0f - Rbar*Rbar)) / denom;
    // Clamp κ: exp(-2κ) underflows at large κ, but Wood/Ulrich handles this
    // gracefully (arg clamps to 1e-10 → w≈1). Allow tighter fits for converged scenes.
    kappa = std::min(kappa, 300.0f);

    float theta = std::acos(std::max(-1.0f, std::min(1.0f, ny)));
    float phi = std::atan2(nz, nx);
    if (phi < 0.0f) phi += TWO_PI;

    out_theta = theta;
    out_phi = phi;
    out_kappa = kappa;
    return true;
}

float vmfPdfCpu(float kappa, float cosTheta)
{
    if (kappa <= 1e-6f) return INV_4PI;
    // Numerically stable form: kappa/(2pi) * exp(kappa*(cosTheta-1)) / (1-exp(-2*kappa))
    float exp_neg2k = std::exp(-2.0f * kappa);
    float denom = 1.0f - exp_neg2k;
    if (denom < 1e-10f) denom = 1.0f;
    float pdf = (kappa / TWO_PI) * std::exp(kappa * (cosTheta - 1.0f)) / denom;
    return std::max(pdf, 0.0f);
}

bool fitTwoLobes(
    const float* dirX, const float* dirY, const float* dirZ,
    const float* weights, uint32_t count,
    TwoLobeFitResult& out)
{
    out = {};
    if (count == 0) return false;

    // Pass 1: accumulate all weighted directions → fit lobe 0
    float s0x = 0, s0y = 0, s0z = 0, s0w = 0;
    for (uint32_t i = 0; i < count; i++) {
        float w = weights[i];
        if (w <= 0.0f) continue;
        s0x += dirX[i] * w;
        s0y += dirY[i] * w;
        s0z += dirZ[i] * w;
        s0w += w;
    }

    if (!fitFromSums(s0x, s0y, s0z, s0w, out.theta0, out.phi0, out.kappa0)) {
        return false;
    }

    // Single-lobe defaults
    out.theta1 = 0; out.phi1 = 0; out.kappa1 = 0;
    out.pi0 = 1.0f;

    // Need at least 20 samples for a meaningful 2-lobe fit
    if (count < 20) return true;

    // Compute lobe 0 mean direction in Cartesian
    float st0 = std::sin(out.theta0);
    float mu0x = st0 * std::cos(out.phi0);
    float mu0y = std::cos(out.theta0);
    float mu0z = st0 * std::sin(out.phi0);

    // Pass 2: soft assignment — compute residual responsibilities for lobe 1
    float s1x = 0, s1y = 0, s1z = 0, s1w = 0;
    float sumPi0W = 0;  // for computing pi_0

    for (uint32_t i = 0; i < count; i++) {
        float w = weights[i];
        if (w <= 0.0f) continue;

        float cosAngle0 = dirX[i] * mu0x + dirY[i] * mu0y + dirZ[i] * mu0z;
        float pdf0 = vmfPdfCpu(out.kappa0, cosAngle0);
        float uniform = INV_4PI;

        // Responsibility of lobe 1 (residual) for this sample
        float r1 = uniform / (pdf0 + uniform);

        s1x += dirX[i] * w * r1;
        s1y += dirY[i] * w * r1;
        s1z += dirZ[i] * w * r1;
        s1w += w * r1;
        sumPi0W += w * (1.0f - r1);
    }

    // Fit lobe 1 from residual sums
    float theta1, phi1, kappa1;
    if (!fitFromSums(s1x, s1y, s1z, s1w, theta1, phi1, kappa1)) {
        return true;  // lobe 1 degenerate, single-lobe is fine
    }

    // Compute mixture weight
    float pi0 = sumPi0W / s0w;

    // Collapse to single lobe if lobe 1 is too weak
    if (kappa1 < 0.5f || (1.0f - pi0) < 0.05f) {
        return true;  // keep single-lobe defaults
    }

    out.theta1 = theta1;
    out.phi1 = phi1;
    out.kappa1 = kappa1;
    out.pi0 = std::max(0.05f, std::min(0.95f, pi0));  // clamp to avoid degenerate weights

    return true;
}

} // namespace vmf_fitting
} // namespace spectra
