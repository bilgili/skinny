// Hero-wavelength spectral path tracer (change spectral-rendering, Group 5).
//
// Compiled ONLY in the spectral megakernel variant (`-DSKINNY_SPECTRAL`); the RGB
// build never imports this module, so its SPIR-V is byte-unchanged. This is a
// deliberately SEPARATE integrator from PathTracer (integrators/path.slang) —
// widening the shared BSDFSample/BounceResult/LightSample carriers to Spectrum
// would break every float3 assignment across the flat/skin/python/bdpt tree, so
// instead this module carries Spectrum (float4) throughput/radiance itself and
// reuses the RGB flat machinery for the wavelength-INDEPENDENT geometry (the
// sampled direction `wi` and the solid-angle pdf), recoloring per wavelength via
// flatBsdfResponseSpectral. v1 scope is path + megakernel + FLAT materials only
// (skin/subsurface/volume/python/BDPT/SPPM are refused at startup), so the loop
// handles the flat case and treats anything else as a terminal miss.
//
// The estimator is standard hero-wavelength MC (Wilkie et al. 2014): draw 4
// rotated wavelengths at path start, transport per wavelength, resolve
// L(λ_i)/pdf(λ_i) → XYZ → linear sRGB at termination (spectrum.slang), feeding
// the SAME RGBA32F accumulation as the RGB path — so exposure/tonemap/readback
// are untouched. RR / MIS stay scalar (they weight the whole Spectrum).

import common;
import spectrum;        // SampledWavelengths, sampleWavelengths, spectrumResolveToLinearSRGB
import bindings;        // fc, spectral buffers + upsampleReflectanceBound/upsampleIlluminantBound
import interfaces;
import scene_trace;
import scene_lights;
import environment;
import materials.flat.flat_shading;    // FlatHitMat
import materials.flat.flat_material;   // FlatMaterial, loadFlatMaterial
import materials.flat.flat_lobes;      // flatBsdfResponseSpectral
import lights.sphere_light;
import lights.emissive_triangle_light;
import lights.directional_light;
import sampling.proposal;              // ProposalContext, mixtureProposalPdf (scalar MIS companion)
import integrators.spectral_flat_common; // SpectralFlatColors, upsampleFlatColors, flatResponseS, spectralAllLightsNEE (hoisted, shared with bdpt_spectral)

static const uint SPECTRAL_MAX_BOUNCES = 6;

// ── Bounce loop ─────────────────────────────────────────────────

struct SpectralPathTracer
{
    float3 estimateRadiance(Ray ray, HitInfo firstHit, inout RNG rng)
    {
        // Hero-wavelength draw: the FIRST consumption of the pixel RNG in the
        // spectral build (the RGB build has no such draw ⇒ its stream is
        // unchanged and its SPIR-V byte-identical). Path-lifetime local.
        SampledWavelengths sw = sampleWavelengths(rng.next());

        Spectrum radiance   = Spectrum(0.0);
        Spectrum throughput = Spectrum(1.0);
        HitInfo  h          = firstHit;
        Ray      r          = ray;

        float misBsdfPdf       = -1.0;
        bool  spawnedBySpecular = false;
        float prevBsdfPdf      = -1.0;

        for (uint skip = 0u; skip < 32u && h.hit && isCutoutTransparent(h); skip++)
        {
            r.origin = h.position + r.direction * 0.002;
            h = traceScene(fc, r);
        }

        for (uint bounce = 0u; bounce < SPECTRAL_MAX_BOUNCES; bounce++)
        {
            if (!h.hit)
            {
                float3   envRgb = sceneEnvRadianceCell(
                    r.direction, fc.envIntensity, effectiveFurnaceMode(h.materialId));
                Spectrum envR = upsampleIlluminantBound(envRgb, sw);
                float w = 1.0;
                if (misBsdfPdf > 0.0)
                    w = powerHeuristic(misBsdfPdf, envPdf(r.direction));
                radiance += throughput * envR * w;
                break;
            }

            // v1 spectral is FLAT-only; any non-flat hit is out-of-envelope
            // (refused at startup) — terminate rather than mis-shade.
            uint matType = materialTypes[h.materialId] & 0xFFu;
            if (matType != MATERIAL_TYPE_FLAT)
                break;

            FlatMaterial mat = loadFlatMaterial(h);
            uint metalId = flatMaterials[h.materialId].conductorMetalId;
            SpectralFlatColors cols = upsampleFlatColors(mat, sw, metalId);

            float3 N = mat.data.N;
            float3 T, B;
            if (h.hasTangent)
            {
                T = normalize(h.tangent - N * dot(h.tangent, N));
                B = cross(N, T);
            }
            else
            {
                buildBasis(N, T, B);
            }
            float3 V  = -r.direction;
            float3 wo = worldToTangent(V, T, B, N);

            // Direct lighting (spectral NEE).
            radiance += throughput * spectralAllLightsNEE(mat, cols, h, N, T, B, wo, sw, rng);

            // BSDF sample: reuse the RGB flat draw for the λ-independent geometry
            // (wi, pdf, delta-ness, transmission). Emission is the surface's own.
            BSDFSample bs = mat.sample(wo, rng);

            // Emission at this hit (emissive flat), MIS-gated exactly as the RGB
            // path (emissive-triangle-bsdf-hit-mis). lum uses the RGB emission.
            // Group 6.1 follow-up: a blackbody material (per-material T>0) emits the
            // exact Planck SPD here too — luminance-matched by the host scale — so a
            // camera-visible / BSDF-hit blackbody agrees with its NEE contribution
            // instead of showing the RGB metamer. Non-blackbody keeps cols.emission.
            Spectrum emis = cols.emission;
            float2   matBB = materialBlackbody(h.materialId);
            if (matBB.x > 0.0)
                emis = planckSpectrum(sw, matBB.x) * matBB.y;
            if (bounce == 0u || fc.numEmissiveTriangles == 0u || spawnedBySpecular)
            {
                radiance += throughput * emis;
            }
            else if (any(bs.emission > 0.0))
            {
                float cosLight = max(dot(h.normal, -r.direction), 0.0);
                float lum      = dot(bs.emission, float3(0.2126, 0.7152, 0.0722));
                float pdfLightSA = (cosLight > 1e-6 && fc.emissiveTotalPower > 0.0)
                    ? lum * (h.t * h.t) / (fc.emissiveTotalPower * cosLight)
                    : 1e30;
                float wBSDF = powerHeuristic(prevBsdfPdf, pdfLightSA);
                radiance += throughput * emis * wBSDF;
            }

            if (!bs.valid)
                break;

            // ── Dispersive dielectric (Group 6.4) ──────────────────────────
            // The RGB sample() refracted at the scalar ior (λ-independent). For a
            // named glass, recompute the delta refraction at the HERO-λ Cauchy
            // index n(λ0) = ior + B/λ0² and collapse to the hero wavelength — the
            // secondary λ would refract along different directions, so pbrt
            // terminates them (terminateSecondary: pdf.x/=4, y/z/w=0). Only the
            // REFRACTED delta branch disperses; reflection off glass is achromatic.
            // A refraction is detected by wi/wo landing on OPPOSITE sides of the
            // interface (wo.z·wi.z < 0) — this catches BOTH the entering AND the
            // EXITING refraction (the exit has wi.z > 0 ⇒ bs.transmitted is false,
            // so gating on transmitted would leave exit refractions undispersed).
            // Constant-IOR glass (glassCauchyB == 0) never enters ⇒ keeps all 4 λ
            // ⇒ unbiased. Reflect fallback on hero-λ TIR is achromatic.
            float glassB = flatMaterials[h.materialId].glassCauchyB;
            if (bs.pdf == 0.0 && glassB > 0.0 && wo.z * bs.wi.z < 0.0)
            {
                float  lam0um   = sw.lambda.x * 1e-3;                 // nm → µm
                float  nHero    = mat.data.mat.ior + glassB / (lam0um * lam0um);
                bool   entering = wo.z > 0.0;
                float3 faceN    = entering ? float3(0, 0, 1) : float3(0, 0, -1);
                float  etaH     = entering ? (1.0 / nHero) : nHero;  // flat_material.slang:39
                float3 nd       = refract(-wo, faceN, etaH);
                if (all(nd == float3(0.0)))                          // hero-λ TIR ⇒ reflect
                    nd = reflect(-wo, faceN);
                bs.wi           = nd;
                bs.transmitted  = nd.z < 0.0;
                sw              = terminateSecondary(sw);
            }

            // Spectral throughput update. Non-delta lobe: response/pdf reproduces
            // the RGB sample weight per wavelength. Delta glass (pdf 0): the RGB
            // weight is transmissionColor — upsample it as a reflectance.
            Spectrum weightS;
            if (bs.pdf > 0.0)
                weightS = flatResponseS(mat, cols, wo, bs.wi) / bs.pdf;
            else
                weightS = cols.transmission;
            throughput *= weightS;

            if (bounce > 0u)
            {
                float mx = max(max(throughput.x, throughput.y),
                               max(throughput.z, throughput.w));
                float p = clamp(mx, 0.05, 0.95);
                if (rng.next() > p)
                    break;
                throughput /= p;
            }

            bool envNeeActive = (effectiveFurnaceMode(h.materialId) == 0u);
            misBsdfPdf = (envNeeActive && !bs.transmitted && bs.pdf > 0.0)
                       ? bs.pdf : -1.0;
            spawnedBySpecular = (bs.pdf <= 0.0);
            prevBsdfPdf = spawnedBySpecular ? -1.0 : bs.pdf;

            float3 wiWorld = tangentToWorld(bs.wi, T, B, N);
            float3 spawnOffset = bs.transmitted ? -h.normal * 0.002 : h.normal * 0.002;
            r.origin    = h.position + spawnOffset;
            r.direction = wiWorld;

            NearestSphereHit sph = intersectSphereLights(r, 1e30, fc.numSphereLights);
            HitInfo hNext = traceScene(fc, r);
            float geom_t = hNext.hit ? hNext.t : 1e30;
            if (sph.idx != 0xFFFFFFFFu && sph.t < geom_t)
            {
                SphereLight light = sphereLights[sph.idx];
                float w_bsdf = 1.0;
                if (!bs.transmitted && bs.pdf > 0.0)
                {
                    float area = 4.0 * PI * light.radius * light.radius;
                    float pdf_light_sa = (sph.t * sph.t) / max(sph.cos_y * area, 1e-12);
                    w_bsdf = powerHeuristic(bs.pdf, pdf_light_sa);
                }
                radiance += throughput * upsampleIlluminantBound(light.radiance, sw) * w_bsdf;
                break;
            }
            h = hNext;

            for (uint skip = 0u; skip < 32u && h.hit && isCutoutTransparent(h); skip++)
            {
                r.origin = h.position + r.direction * 0.002;
                h = traceScene(fc, r);
            }
        }

        return spectrumResolveToLinearSRGB(sw, radiance);
    }
}
