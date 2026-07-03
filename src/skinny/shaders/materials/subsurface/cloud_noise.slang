// pbrt classic Perlin noise + procedural cloud density (pbrt-cloud-procedural-medium).
//
// Line-for-line port of pbrt-v4 `util/noise.cpp` (`Noise`/`DNoise`, the 256->512
// duplicated `NoisePerm` permutation table, `Grad`, quintic `NoiseWeight`) and
// `CloudMedium::Density` (`media.h:493`): the 5-octave fBm, the 2-iteration
// wispiness domain warp, and the altitude falloff. The constants MUST stay
// bit-identical to pbrt's -- the parity gate compares against pbrt's own render,
// and a look-alike fBm passes visual inspection while failing on structure.
// Mirrored in numpy by tests/test_cloud_noise.py (the practical oracle).
//
// Purely analytic: no texture, no binding -- consumed only by densityAt's
// MEDIUM_CLOUD case (materials/subsurface/medium.slang).

// pbrt util/noise.cpp NoisePerm[2 * NoisePermSize], duplicated 256 -> 512 so
// Grad's nested lookups never wrap (extracted verbatim from the pbrt source).
static const int NOISE_PERM[512] = {
    151, 160, 137, 91, 90, 15, 131, 13, 201, 95, 96, 53, 194, 233, 7, 225,
    140, 36, 103, 30, 69, 142, 8, 99, 37, 240, 21, 10, 23, 190, 6, 148,
    247, 120, 234, 75, 0, 26, 197, 62, 94, 252, 219, 203, 117, 35, 11, 32,
    57, 177, 33, 88, 237, 149, 56, 87, 174, 20, 125, 136, 171, 168, 68, 175,
    74, 165, 71, 134, 139, 48, 27, 166, 77, 146, 158, 231, 83, 111, 229, 122,
    60, 211, 133, 230, 220, 105, 92, 41, 55, 46, 245, 40, 244, 102, 143, 54,
    65, 25, 63, 161, 1, 216, 80, 73, 209, 76, 132, 187, 208, 89, 18, 169,
    200, 196, 135, 130, 116, 188, 159, 86, 164, 100, 109, 198, 173, 186, 3, 64,
    52, 217, 226, 250, 124, 123, 5, 202, 38, 147, 118, 126, 255, 82, 85, 212,
    207, 206, 59, 227, 47, 16, 58, 17, 182, 189, 28, 42, 223, 183, 170, 213,
    119, 248, 152, 2, 44, 154, 163, 70, 221, 153, 101, 155, 167, 43, 172, 9,
    129, 22, 39, 253, 19, 98, 108, 110, 79, 113, 224, 232, 178, 185, 112, 104,
    218, 246, 97, 228, 251, 34, 242, 193, 238, 210, 144, 12, 191, 179, 162, 241,
    81, 51, 145, 235, 249, 14, 239, 107, 49, 192, 214, 31, 181, 199, 106, 157,
    184, 84, 204, 176, 115, 121, 50, 45, 127, 4, 150, 254, 138, 236, 205, 93,
    222, 114, 67, 29, 24, 72, 243, 141, 128, 195, 78, 66, 215, 61, 156, 180,
    151, 160, 137, 91, 90, 15, 131, 13, 201, 95, 96, 53, 194, 233, 7, 225,
    140, 36, 103, 30, 69, 142, 8, 99, 37, 240, 21, 10, 23, 190, 6, 148,
    247, 120, 234, 75, 0, 26, 197, 62, 94, 252, 219, 203, 117, 35, 11, 32,
    57, 177, 33, 88, 237, 149, 56, 87, 174, 20, 125, 136, 171, 168, 68, 175,
    74, 165, 71, 134, 139, 48, 27, 166, 77, 146, 158, 231, 83, 111, 229, 122,
    60, 211, 133, 230, 220, 105, 92, 41, 55, 46, 245, 40, 244, 102, 143, 54,
    65, 25, 63, 161, 1, 216, 80, 73, 209, 76, 132, 187, 208, 89, 18, 169,
    200, 196, 135, 130, 116, 188, 159, 86, 164, 100, 109, 198, 173, 186, 3, 64,
    52, 217, 226, 250, 124, 123, 5, 202, 38, 147, 118, 126, 255, 82, 85, 212,
    207, 206, 59, 227, 47, 16, 58, 17, 182, 189, 28, 42, 223, 183, 170, 213,
    119, 248, 152, 2, 44, 154, 163, 70, 221, 153, 101, 155, 167, 43, 172, 9,
    129, 22, 39, 253, 19, 98, 108, 110, 79, 113, 224, 232, 178, 185, 112, 104,
    218, 246, 97, 228, 251, 34, 242, 193, 238, 210, 144, 12, 191, 179, 162, 241,
    81, 51, 145, 235, 249, 14, 239, 107, 49, 192, 214, 31, 181, 199, 106, 157,
    184, 84, 204, 176, 115, 121, 50, 45, 127, 4, 150, 254, 138, 236, 205, 93,
    222, 114, 67, 29, 24, 72, 243, 141, 128, 195, 78, 66, 215, 61, 156, 180,
};

// pbrt NoiseWeight: quintic fade 6t^5 - 15t^4 + 10t^3.
float noiseWeight(float t)
{
    float t3 = t * t * t;
    float t4 = t3 * t;
    return 6.0 * t4 * t - 15.0 * t4 + 10.0 * t3;
}

// pbrt Grad: gradient selector via the permutation table. `x/y/z` are lattice
// coords already wrapped to [0, 255] (+1 neighbours reach 256 -- in range for
// the duplicated table); `dx/dy/dz` are the fractional offsets.
float pbrtGrad(int x, int y, int z, float dx, float dy, float dz)
{
    int h = NOISE_PERM[NOISE_PERM[NOISE_PERM[x] + y] + z];
    h &= 15;
    float u = (h < 8 || h == 12 || h == 13) ? dx : dy;
    float v = (h < 4 || h == 12 || h == 13) ? dy : dz;
    return ((h & 1) != 0 ? -u : u) + ((h & 2) != 0 ? -v : v);
}

// pbrt Noise(Point3f): classic Perlin over the integer lattice.
float pbrtNoise(float3 p)
{
    // Avoid int32 overflow for huge coordinates, exactly as pbrt does. C fmod
    // semantics (result has the sign of the dividend) match Slang's fmod.
    float x = fmod(p.x, float(1 << 30));
    float y = fmod(p.y, float(1 << 30));
    float z = fmod(p.z, float(1 << 30));
    int ix = int(floor(x)), iy = int(floor(y)), iz = int(floor(z));
    float dx = x - float(ix), dy = y - float(iy), dz = z - float(iz);

    ix &= 255;
    iy &= 255;
    iz &= 255;
    float w000 = pbrtGrad(ix, iy, iz, dx, dy, dz);
    float w100 = pbrtGrad(ix + 1, iy, iz, dx - 1.0, dy, dz);
    float w010 = pbrtGrad(ix, iy + 1, iz, dx, dy - 1.0, dz);
    float w110 = pbrtGrad(ix + 1, iy + 1, iz, dx - 1.0, dy - 1.0, dz);
    float w001 = pbrtGrad(ix, iy, iz + 1, dx, dy, dz - 1.0);
    float w101 = pbrtGrad(ix + 1, iy, iz + 1, dx - 1.0, dy, dz - 1.0);
    float w011 = pbrtGrad(ix, iy + 1, iz + 1, dx, dy - 1.0, dz - 1.0);
    float w111 = pbrtGrad(ix + 1, iy + 1, iz + 1, dx - 1.0, dy - 1.0, dz - 1.0);

    float wx = noiseWeight(dx), wy = noiseWeight(dy), wz = noiseWeight(dz);
    float x00 = lerp(w000, w100, wx);
    float x10 = lerp(w010, w110, wx);
    float x01 = lerp(w001, w101, wx);
    float x11 = lerp(w011, w111, wx);
    float y0 = lerp(x00, x10, wy);
    float y1 = lerp(x01, x11, wy);
    return lerp(y0, y1, wz);
}

// pbrt DNoise: forward-difference noise gradient (delta = 0.01), used by the
// cloud wispiness domain warp.
float3 pbrtDNoise(float3 p)
{
    const float delta = 0.01;
    float n = pbrtNoise(p);
    float3 noiseDelta = float3(
        pbrtNoise(p + float3(delta, 0.0, 0.0)),
        pbrtNoise(p + float3(0.0, delta, 0.0)),
        pbrtNoise(p + float3(0.0, 0.0, delta)));
    return (noiseDelta - float3(n, n, n)) / delta;
}

// pbrt CloudMedium::Density. `p` is the point in medium-local space (== the
// packed worldToUvw output; for pbrt's default bounds [0,1]^3 medium-local and
// normalized coords coincide). The caller (densityAt's MEDIUM_CLOUD case)
// clips to the [0,1]^3 medium bounds -- pbrt clips in SampleRay's bounds
// intersection, so Density itself stays unclipped, exactly like this port.
float cloudDensity(float3 p, float density, float wispiness, float frequency)
{
    float3 pp = frequency * p;
    if (wispiness > 0.0)
    {
        // Perturb the lookup point with 2 octaves of DNoise.
        float vomega = 0.05 * wispiness, vlambda = 10.0;
        for (int i = 0; i < 2; ++i)
        {
            pp += vomega * pbrtDNoise(vlambda * pp);
            vomega *= 0.5;
            vlambda *= 1.99;
        }
    }
    // Sum 5 octaves of noise to approximate the cloud density.
    float d = 0.0;
    float omega = 0.5, lambda = 1.0;
    for (int i = 0; i < 5; ++i)
    {
        d += omega * pbrtNoise(lambda * pp);
        omega *= 0.5;
        lambda *= 1.99;
    }

    // Density decreases with altitude; the low-half floor fills the cloud base.
    d = clamp((1.0 - p.y) * 4.5 * density * d, 0.0, 1.0);
    d += 2.0 * max(0.0, 0.5 - p.y);
    return clamp(d, 0.0, 1.0);
}
