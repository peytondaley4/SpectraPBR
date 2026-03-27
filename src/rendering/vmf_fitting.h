#pragma once

//------------------------------------------------------------------------------
// von Mises–Fisher fitting (host-side)
// Fit a single vMF lobe from weighted direction sums: (sumX, sumY, sumZ, sumW)
// -> mean direction mu = sum/|sum|, concentration kappa via R_bar (Sra 2012).
// Output (theta, phi, kappa) matches device convention: theta from +Y, phi [0,2*pi].
//------------------------------------------------------------------------------

#include <cstdint>

namespace spectra {

namespace vmf_fitting {

// Fit vMF from weighted sums. Returns true if fit is valid and out* were written.
// sumX, sumY, sumZ = sum of (direction * weight), sumW = sum of weights.
bool fitFromSums(float sumX, float sumY, float sumZ, float sumW,
                 float& out_theta, float& out_phi, float& out_kappa);

// CPU-side vMF PDF: C3(kappa)*exp(kappa*cosTheta)
float vmfPdfCpu(float kappa, float cosTheta);

// Result of two-lobe EM fitting
struct TwoLobeFitResult {
    float theta0, phi0, kappa0;
    float theta1, phi1, kappa1;
    float pi0;  // mixture weight for lobe 0
};

// Fit a 2-lobe vMF mixture from per-sample direction/weight arrays via EM.
// Returns true if at least one lobe was fitted. Requires count >= 1.
// If count < 20 or lobe 1 is degenerate, collapses to single-lobe (kappa1=0, pi0=1).
bool fitTwoLobes(
    const float* dirX, const float* dirY, const float* dirZ,
    const float* weights, uint32_t count,
    TwoLobeFitResult& out);

// Compute log-likelihood of data under a single vMF lobe fitted from aggregate sums.
// Uses: LL = N * log(C3(kappa)) + kappa * R, where R = ||(sumX, sumY, sumZ)||.
float logLikelihoodSingleLobe(float sumX, float sumY, float sumZ, float sumW);

// Bayesian Information Criterion for model selection.
// BIC = -2 * LL + numParams * ln(N).
// Lower BIC = better model (accounting for complexity).
// For vMF: 1 lobe = 3 params, 2 lobes = 7 params (3+3+1 weight).
float computeBIC(float logLikelihood, uint32_t numParams, float effectiveSampleCount);

// Compare 1-lobe vs 2-lobe fit using BIC.
// Returns true if 2-lobe fit is justified by the data.
// cumSum = cumulative (lifetime) aggregates, intSum = current interval aggregates.
bool shouldSplitLobe(
    float cumSumX, float cumSumY, float cumSumZ, float cumSumW,
    float intSumX, float intSumY, float intSumZ, float intSumW);

} // namespace vmf_fitting
} // namespace spectra
