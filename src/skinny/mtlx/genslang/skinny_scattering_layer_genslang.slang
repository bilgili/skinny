// Generic scattering layer escape hatch. Pure pass-through so a non-skin
// layered material (wax, marble, frosted plastic) can author σ_a / σ_s
// directly and still feed into the skin BSDF/VDF nodedefs.

void skinny_scattering_layer(
    float3 absorption,
    float3 scattering,
    float  thickness,
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
