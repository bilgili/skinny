// Epidermis layer: melanin-driven absorption, σ_s passes through.
//
// Mirrors `buildEpidermis()` in shaders/skin_bssrdf.slang. The melanin fit
// follows Donner & Jensen 2006 (RGB sample at 611, 549, 464 nm).

void skinny_skin_epidermis(
    float  melanin,
    float  thickness,
    float3 scattering_coeff,
    float  anisotropy,
    float  ior,
    out float3 absorption,
    out float3 scattering,
    out float  anisotropy_out,
    out float  thickness_out,
    out float  ior_out)
{
    // Eumelanin coefficients in mm^-1 at the RGB centres, plus a baseline
    // tissue absorption of 0.015 mm^-1 covering water + cell components.
    float3 eumelanin = float3(6.6, 11.0, 24.0);
    absorption       = melanin * eumelanin + float3(0.015);
    scattering       = scattering_coeff;
    anisotropy_out   = anisotropy;
    thickness_out    = thickness;
    ior_out          = ior;
}
