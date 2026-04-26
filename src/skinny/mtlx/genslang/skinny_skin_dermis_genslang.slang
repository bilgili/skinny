// Dermis layer: hemoglobin/oxygenation drive absorption; pigment overlay
// (replaces the legacy tattoo path) adds extra σ_a / σ_s in proportion to
// alpha density.
//
// Mirrors `buildDermisInked()` in shaders/skin_bssrdf.slang.

void skinny_skin_dermis(
    float  hemoglobin,
    float  blood_oxygenation,
    float  thickness,
    float3 scattering_coeff,
    float  anisotropy,
    float  ior,
    float4 pigment,                 // rgb = ink colour, a = density 0..1
    out float3 absorption,
    out float3 scattering,
    out float  anisotropy_out,
    out float  thickness_out,
    out float  ior_out)
{
    // Oxy/deoxy hemoglobin RGB approximations.
    float3 hbO2  = float3(0.3, 7.0, 1.0);
    float3 hb    = float3(1.6, 8.5, 3.2);
    float3 hbAbs = hemoglobin * (blood_oxygenation * hbO2 + (1.0 - blood_oxygenation) * hb);

    float3 baseAbsorption = hbAbs + float3(0.015);
    float3 baseScattering = scattering_coeff;

    // Pigment overlay: complement-channel absorption + slight extra scatter.
    float density = clamp(pigment.a, 0.0, 1.0);
    if (density > 1e-6)
    {
        float3 inkAbsContrib = 22.0 * (1.0 - pigment.rgb) + 1.2 * pigment.rgb;
        float  inkLum        = dot(pigment.rgb, float3(0.299, 0.587, 0.114));
        float3 inkScatter    = 3.5 * (1.0 + 0.5 * inkLum);

        baseAbsorption += density * inkAbsContrib;
        baseScattering += density * inkScatter;
    }

    absorption     = baseAbsorption;
    scattering     = baseScattering;
    anisotropy_out = anisotropy;
    thickness_out  = thickness;
    ior_out        = ior;
}
