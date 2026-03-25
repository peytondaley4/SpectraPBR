#pragma once

//------------------------------------------------------------------------------
// von Mises–Fisher (vMF) distribution — device-side only
// Self-contained: spherical/cartesian conversion, PDF, sampling.
// Convention: theta = polar angle from +Y [0, pi], phi = azimuth [0, 2*pi].
// mu = (sin(theta)*cos(phi), cos(theta), sin(theta)*sin(phi)).
// PDF: C3(kappa)*exp(kappa*dot(mu,omega)), C3(kappa) = kappa/(4*pi*sinh(kappa)).
// Sampling: Wood/Ulrich (Wikipedia; see also Müller et al. EGSR 2017 path guiding).
//------------------------------------------------------------------------------

#include <cuda_runtime.h>

__forceinline__ __device__ void vmfSphericalToCartesian(float theta, float phi, float& mx, float& my, float& mz) {
    float sinTheta, cosTheta, sinPhi, cosPhi;
    sincosf(theta, &sinTheta, &cosTheta);
    sincosf(phi, &sinPhi, &cosPhi);
    mx = sinTheta * cosPhi;
    my = cosTheta;
    mz = sinTheta * sinPhi;
}

__forceinline__ __device__ void vmfCartesianToSpherical(float mx, float my, float mz, float& theta, float& phi) {
    float len = sqrtf(mx*mx + my*my + mz*mz);
    if (len < 1e-8f) { theta = 0.0f; phi = 0.0f; return; }
    theta = acosf(fminf(fmaxf(my / len, -1.0f), 1.0f));
    phi = atan2f(mz, mx);
    if (phi < 0.0f) phi += 6.28318530718f;
}

// vMF PDF in 3D: C3(kappa)*exp(kappa*cos_theta) where cos_theta = dot(mu, omega)
// Numerically stable form: kappa/(2pi) * exp(kappa*(cos_theta-1)) / (1-exp(-2*kappa))
// Since cos_theta-1 <= 0, the exp never overflows. For large kappa, denominator -> 1.
__forceinline__ __device__ float vmfPdf(float kappa, float cos_theta) {
    if (kappa <= 1e-6f) return 0.07957747154f;  // 1/(4*pi)
    float exp_neg2k = expf(-2.0f * kappa);
    float denom = 1.0f - exp_neg2k;
    if (denom < 1e-10f) denom = 1.0f;  // large kappa: exp(-2k) underflows to 0
    float pdf = (kappa / 6.28318530718f) * expf(kappa * (cos_theta - 1.0f)) / denom;
    return fmaxf(pdf, 0.0f);
}

// Sample direction from vMF(mu, kappa). Wood/Ulrich: w = 1 + ln(u1+(1-u1)*exp(-2κ))/κ, then omega = sqrt(1-w^2)*v + w*mu.
// mu must be unit length. u1, u2 in [0,1) from caller.
__forceinline__ __device__ void vmfSample(
    float mx, float my, float mz,
    float kappa,
    float u1, float u2,
    float& ox, float& oy, float& oz)
{
    float w;
    if (kappa <= 1e-6f) {
        w = 2.0f * u1 - 1.0f;  // uniform on sphere
    } else {
        float exp2k = expf(-2.0f * kappa);
        float arg = u1 + (1.0f - u1) * exp2k;
        if (arg < 1e-10f) arg = 1e-10f;
        w = 1.0f + logf(arg) / kappa;
    }
    w = fminf(fmaxf(w, -1.0f), 1.0f);

    // Orthonormal basis: t, b perpendicular to mu
    float tx, ty, tz;
    if (fabsf(my) < 0.9f) {
        tx = -mz; ty = 0.0f; tz = mx;
    } else {
        tx = my; ty = -mx; tz = 0.0f;
    }
    float tlen = sqrtf(tx*tx + ty*ty + tz*tz);
    if (tlen < 1e-8f) { tx = 1.0f; ty = 0.0f; tz = 0.0f; tlen = 1.0f; }
    tx /= tlen; ty /= tlen; tz /= tlen;
    float bx = my*tz - mz*ty, by = mz*tx - mx*tz, bz = mx*ty - my*tx;
    float blen = sqrtf(bx*bx + by*by + bz*bz);
    if (blen < 1e-8f) blen = 1.0f;
    bx /= blen; by /= blen; bz /= blen;

    float angle = 6.28318530718f * u2;
    float r = sqrtf(fmaxf(1.0f - w*w, 0.0f));
    float sinA, cosA;
    sincosf(angle, &sinA, &cosA);
    float vx = r * cosA, vy = r * sinA;
    ox = vx*tx + vy*bx + w*mx;
    oy = vx*ty + vy*by + w*my;
    oz = vx*tz + vy*bz + w*mz;
    float olen = sqrtf(ox*ox + oy*oy + oz*oz);
    if (olen > 1e-8f) { ox /= olen; oy /= olen; oz /= olen; }
}
