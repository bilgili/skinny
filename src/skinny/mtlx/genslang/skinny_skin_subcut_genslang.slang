// Subcutaneous fat layer. Defaults reproduce the legacy hard-coded values
// (sigmaA = 0.05/0.06/0.08 mm^-1, sigmaS = 1.2/1.1/0.95 mm^-1, g = 0.8,
// ior = 1.4). All five fields are pass-through so a user can override the
// fat optics by binding to the inputs.
//
// Mirrors `buildSubcutaneous()` in shaders/skin_bssrdf.slang.

void skinny_skin_subcut(
    float  thickness,
    float3 absorption,
    float3 scattering,
    float  anisotropy,
    float  ior,
    out float3 absorption_out,
    out float3 scattering_out,
    out float  anisotropy_out,
    out float  thickness_out,
    out float  ior_out)
{
    absorption_out = absorption;
    scattering_out = scattering;
    anisotropy_out = anisotropy;
    thickness_out  = thickness;
    ior_out        = ior;
}
