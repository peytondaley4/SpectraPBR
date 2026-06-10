#pragma once

//------------------------------------------------------------------------------
// von Mises–Fisher statistics (host-side)
//
// Lobe FITTING now happens on the GPU (path_guide_kernels.cu::refitCellsKernel,
// same Banerjee/Sra kappa approximation). What remains here is the
// log-likelihood used by the refinement pass to judge single-lobe fit quality
// from a cell's aggregate sums.
//------------------------------------------------------------------------------

#include <cstdint>

namespace spectra {

namespace vmf_fitting {

// Log-likelihood of data under a single vMF lobe fitted from aggregate sums.
// Uses: LL = N * log(C3(kappa)) + kappa * R, where R = ||(sumX, sumY, sumZ)||
// and N is the total weight sumW. A low LL per unit weight means one lobe
// explains the directional distribution poorly at this spatial resolution.
float logLikelihoodSingleLobe(float sumX, float sumY, float sumZ, float sumW);

} // namespace vmf_fitting
} // namespace spectra
