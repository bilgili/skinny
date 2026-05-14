// MaterialX standard_surface evaluation for path-tracer NEE.
// Evaluates the full closure tree (coat → metal → specular → transmission →
// sheen → subsurface → diffuse) using the MaterialX pbrlib BSDF functions.
// The closure tree structure matches NG_standard_surface_surfaceshader_100
// emitted by MaterialXGenSlang.

import mtlx_closures;

// All standard_surface inputs packed into std430 layout.
// Renderer packs this from Material.parameter_overrides via
// pack_std_surface_params(). Field order matches the Python packer.
struct StdSurfaceParams
{
    float3 base_color;        // 0..12
    float  base;              // 12..16
    float  diffuse_roughness; // 16..20
    float  metalness;         // 20..24
    float  specular;          // 24..28
    float  specular_roughness;// 28..32
    float3 specular_color;    // 32..44
    float  specular_IOR;      // 44..48
    float  specular_anisotropy;// 48..52
    float  specular_rotation; // 52..56
    float  transmission;      // 56..60
    float  transmission_depth;// 60..64
    float3 transmission_color;// 64..76
    float  transmission_scatter_anisotropy; // 76..80
    float3 transmission_scatter; // 80..92
    float  transmission_dispersion; // 92..96
    float  transmission_extra_roughness; // 96..100
    float  subsurface;        // 100..104
    float  subsurface_scale;  // 104..108
    float  subsurface_anisotropy; // 108..112
    float3 subsurface_color;  // 112..124
    float  _pad0;             // 124..128
    float3 subsurface_radius; // 128..140
    float  sheen;             // 140..144
    float3 sheen_color;       // 144..156
    float  sheen_roughness;   // 156..160
    float  coat;              // 160..164
    float  coat_roughness;    // 164..168
    float  coat_anisotropy;   // 168..172
    float  coat_rotation;     // 172..176
    float  coat_IOR;          // 176..180
    float  coat_affect_color; // 180..184
    float  coat_affect_roughness; // 184..188
    float  _pad1;             // 188..192
    float3 coat_color;        // 192..204
    float  thin_film_thickness; // 204..208
    float  thin_film_IOR;     // 208..212
    float  emission;          // 212..216
    float3 emission_color;    // 216..228
    float  _pad2;             // 228..232
    float3 opacity;           // 232..244
    uint   thin_walled;       // 244..248
    float  _pad3;             // 248..252
    float  _pad4;             // 252..256
};

// Evaluate the standard_surface BSDF for a given light direction.
// Returns bsdf.response = f(L,V) * NdotL (matches MaterialX convention).
// cd.closureType should be CLOSURE_TYPE_REFLECTION for NEE.
BSDF evalStdSurfaceBSDF(ClosureData cd, StdSurfaceParams p,
                         float3 normal, float3 tangent)
{
    // --- Derived quantities (mirrors NG_standard_surface_surfaceshader_100) ---

    float3 base_color = max(p.base_color, 0.0);
    float3 specular_color = p.specular_color;
    float specular_IOR = p.specular_IOR;
    float coat = clamp(p.coat, 0.0, 1.0);
    float coat_IOR = p.coat_IOR;

    // Coat roughness
    float2 coat_roughness_vector;
    mx_roughness_anisotropy(p.coat_roughness, p.coat_anisotropy, coat_roughness_vector);

    // Coat tangent (rotation)
    float3 coat_tangent_rotated;
    mx_rotate_vector3(tangent, p.coat_rotation * 360.0, normal, coat_tangent_rotated);
    float3 coat_tangent = (p.coat_anisotropy > 0.0) ? normalize(coat_tangent_rotated) : tangent;

    // Coat normal — for now just use the surface normal
    float3 coat_normal = normal;

    // Metal reflectivity → artistic IOR
    float3 metal_reflectivity = base_color * p.base;
    float3 metal_edgecolor = specular_color * p.specular;
    float3 artistic_ior_ior;
    float3 artistic_ior_extinction;
    mx_artistic_ior(metal_reflectivity, metal_edgecolor, artistic_ior_ior, artistic_ior_extinction);

    // Coat affect roughness
    float coat_affect_roughness_mul = p.coat_affect_roughness * coat * p.coat_roughness;
    float coat_affected_roughness = lerp(p.specular_roughness, 1.0, coat_affect_roughness_mul);

    // Main roughness
    float2 main_roughness;
    mx_roughness_anisotropy(coat_affected_roughness, p.specular_anisotropy, main_roughness);

    // Main tangent (rotation)
    float3 main_tangent_rotated;
    mx_rotate_vector3(tangent, p.specular_rotation * 360.0, normal, main_tangent_rotated);
    float3 main_tangent = (p.specular_anisotropy > 0.0) ? normalize(main_tangent_rotated) : tangent;

    // Transmission roughness
    float transmission_roughness_raw = clamp(p.specular_roughness + p.transmission_extra_roughness, 0.0, 1.0);
    float coat_affected_tr_roughness = lerp(transmission_roughness_raw, 1.0, coat_affect_roughness_mul);
    float2 transmission_roughness;
    mx_roughness_anisotropy(coat_affected_tr_roughness, p.specular_anisotropy, transmission_roughness);

    // Subsurface selector (thin_walled → translucent, otherwise subsurface scattering)
    float subsurface_selector = float(p.thin_walled);
    float selected_subsurface_fg_weight = subsurface_selector;
    float selected_subsurface_bg_weight = 1.0 - subsurface_selector;

    // Coat attenuation + affect color
    float3 coat_attenuation = lerp(float3(1.0), p.coat_color, coat);
    float coat_gamma = coat * p.coat_affect_color + 1.0;
    float3 coat_affected_subsurface_color = pow(max(p.subsurface_color, 0.0), float3(coat_gamma));
    float3 coat_affected_diffuse_color = pow(base_color, float3(coat_gamma));

    // Subsurface radius scaled
    float3 subsurface_radius_scaled = p.subsurface_radius * p.subsurface_scale;

    // Metalness / transmission / subsurface mix weights
    float metalness_fg_weight = p.metalness;
    float metalness_inv = 1.0 - p.metalness;
    float transmission_inv = 1.0 - p.transmission;
    float subsurface_mix_bg_weight = p.base * (1.0 - p.subsurface);

    // --- Closure tree evaluation ---

    // Coat
    BSDF coat_bsdf = makeBSDF(float3(0.0), float3(1.0));
    mx_dielectric_bsdf(cd, coat, float3(1.0), coat_IOR,
                        coat_roughness_vector, false,
                        0.0, 1.5,
                        coat_normal, coat_tangent,
                        0, 0, coat_bsdf);

    // Metal
    BSDF metal_bsdf = makeBSDF(float3(0.0), float3(1.0));
    mx_conductor_bsdf(cd, metalness_fg_weight,
                       artistic_ior_ior, artistic_ior_extinction,
                       main_roughness, false,
                       p.thin_film_thickness, p.thin_film_IOR,
                       normal, main_tangent, 0, metal_bsdf);

    // Specular
    BSDF specular_bsdf = makeBSDF(float3(0.0), float3(1.0));
    mx_dielectric_bsdf(cd, p.specular, specular_color, specular_IOR,
                        main_roughness, false,
                        p.thin_film_thickness, p.thin_film_IOR,
                        normal, main_tangent,
                        0, 0, specular_bsdf);

    // Transmission
    BSDF transmission_bsdf = makeBSDF(float3(0.0), float3(1.0));
    mx_dielectric_bsdf(cd, p.transmission,
                        p.transmission_color, specular_IOR,
                        transmission_roughness, false,
                        0.0, 1.5,
                        normal, main_tangent,
                        0, 1, transmission_bsdf);

    // Sheen
    BSDF sheen_bsdf = makeBSDF(float3(0.0), float3(1.0));
    mx_sheen_bsdf(cd, p.sheen, p.sheen_color, p.sheen_roughness,
                   normal, 0, sheen_bsdf);

    // Translucent (thin-walled subsurface path)
    BSDF translucent_bsdf = makeBSDF(float3(0.0), float3(1.0));
    mx_translucent_bsdf(cd, selected_subsurface_fg_weight,
                         coat_affected_subsurface_color,
                         normal, translucent_bsdf);

    // Subsurface scattering
    BSDF subsurface_bsdf = makeBSDF(float3(0.0), float3(1.0));
    mx_subsurface_bsdf(cd, selected_subsurface_bg_weight,
                        coat_affected_subsurface_color,
                        subsurface_radius_scaled,
                        p.subsurface_anisotropy,
                        normal, subsurface_bsdf);

    // Subsurface mix: translucent + subsurface
    BSDF selected_subsurface;
    mx_add_bsdf(cd, translucent_bsdf, subsurface_bsdf, selected_subsurface);

    // Scale by subsurface weight
    BSDF subsurface_scaled;
    mx_multiply_bsdf_float(cd, selected_subsurface, p.subsurface, subsurface_scaled);

    // Diffuse
    BSDF diffuse_bsdf = makeBSDF(float3(0.0), float3(1.0));
    mx_oren_nayar_diffuse_bsdf(cd, subsurface_mix_bg_weight,
                                coat_affected_diffuse_color,
                                p.diffuse_roughness,
                                normal, false, diffuse_bsdf);

    // Diffuse + subsurface mix
    BSDF subsurface_mix;
    mx_add_bsdf(cd, subsurface_scaled, diffuse_bsdf, subsurface_mix);

    // Sheen layer over subsurface mix
    BSDF sheen_layer;
    mx_layer_bsdf(cd, sheen_bsdf, subsurface_mix, sheen_layer);

    // Transmission mix: transmission + sheen_layer * (1 - transmission)
    BSDF sheen_layer_scaled;
    mx_multiply_bsdf_float(cd, sheen_layer, transmission_inv, sheen_layer_scaled);
    BSDF transmission_mix;
    mx_add_bsdf(cd, transmission_bsdf, sheen_layer_scaled, transmission_mix);

    // Specular layer over transmission mix
    BSDF specular_layer;
    mx_layer_bsdf(cd, specular_bsdf, transmission_mix, specular_layer);

    // Metalness mix: metal + specular_layer * (1 - metalness)
    BSDF specular_layer_scaled;
    mx_multiply_bsdf_float(cd, specular_layer, metalness_inv, specular_layer_scaled);
    BSDF metalness_mix;
    mx_add_bsdf(cd, metal_bsdf, specular_layer_scaled, metalness_mix);

    // Coat attenuation
    BSDF attenuated;
    mx_multiply_bsdf_color3(cd, metalness_mix, coat_attenuation, attenuated);

    // Coat layer over attenuated base
    BSDF coat_layer;
    mx_layer_bsdf(cd, coat_bsdf, attenuated, coat_layer);

    // Guard against NaN / negative response from energy-compensation
    // overestimates or degenerate half-vector configurations.
    bool3 bad = isnan(coat_layer.response) | isinf(coat_layer.response);
    coat_layer.response = select(bad, float3(0.0), coat_layer.response);
    coat_layer.response = max(coat_layer.response, float3(0.0));

    return coat_layer;
}

// Emission from the material (independent of light direction).
float3 evalStdSurfaceEmission(StdSurfaceParams p)
{
    return p.emission_color * p.emission;
}

float stdSurfaceOpacity(StdSurfaceParams p)
{
    return dot(p.opacity, float3(0.272229, 0.674082, 0.053689));
}
