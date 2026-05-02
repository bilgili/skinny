// MaterialX noise primitives — Perlin, fractal3d, and supporting hash
// functions.  Extracted from the MaterialX stdlib GenSlang output so
// procedural material evaluation (marble, wood, etc.) can run in our
// compute pipeline without pulling in the full GenSlang runtime.
//
// References:
//   [1] Perlin, "An Image Synthesizer", SIGGRAPH 1985.
//   [2] Musgrave, "Procedural Fractal Terrains", in Texturing and Modelling:
//       A Procedural Approach, 3rd ed., Morgan Kaufmann 2002.

#define mx_sin sin
#define mx_float_bits_to_int asint

float mx_select(bool b, float t, float f) { return b ? t : f; }
float mx_negate_if(float val, bool b) { return b ? -val : val; }
int   mx_floor(float x) { return int(floor(x)); }

float mx_floorfrac(float x, out int i)
{
    i = mx_floor(x);
    return x - float(i);
}

float mx_bilerp(float v0, float v1, float v2, float v3, float s, float t)
{
    float s1 = 1.0 - s;
    return (1.0 - t) * (v0 * s1 + v1 * s) + t * (v2 * s1 + v3 * s);
}

float mx_trilerp(float v0, float v1, float v2, float v3,
                 float v4, float v5, float v6, float v7,
                 float s, float t, float r)
{
    float s1 = 1.0 - s;
    float t1 = 1.0 - t;
    float r1 = 1.0 - r;
    return (r1 * (t1 * (v0 * s1 + v1 * s) + t * (v2 * s1 + v3 * s)) +
             r * (t1 * (v4 * s1 + v5 * s) + t * (v6 * s1 + v7 * s)));
}

// ── Gradient functions ─────────────────────────────────────────

float mx_gradient_float(uint hash, float x, float y)
{
    uint h = hash & 7u;
    float u = mx_select(h < 4u, x, y);
    float v = 2.0 * mx_select(h < 4u, y, x);
    return mx_negate_if(u, bool(h & 1u)) + mx_negate_if(v, bool(h & 2u));
}

float mx_gradient_float(uint hash, float x, float y, float z)
{
    uint h = hash & 15u;
    float u = mx_select(h < 8u, x, y);
    float v = mx_select(h < 4u, y, mx_select((h == 12u) || (h == 14u), x, z));
    return mx_negate_if(u, bool(h & 1u)) + mx_negate_if(v, bool(h & 2u));
}

float mx_gradient_scale2d(float v) { return 0.6616 * v; }
float mx_gradient_scale3d(float v) { return 0.9820 * v; }

// ── Bob Jenkins hash (mix / final) ─────────────────────────────

uint mx_rotl32(uint x, int k) { return (x << k) | (x >> (32 - k)); }

void mx_bjmix(inout uint a, inout uint b, inout uint c)
{
    a -= c; a ^= mx_rotl32(c, 4);  c += b;
    b -= a; b ^= mx_rotl32(a, 6);  a += c;
    c -= b; c ^= mx_rotl32(b, 8);  b += a;
    a -= c; a ^= mx_rotl32(c, 16); c += b;
    b -= a; b ^= mx_rotl32(a, 19); a += c;
    c -= b; c ^= mx_rotl32(b, 4);  b += a;
}

uint mx_bjfinal(uint a, uint b, uint c)
{
    c ^= b; c -= mx_rotl32(b, 14);
    a ^= c; a -= mx_rotl32(c, 11);
    b ^= a; b -= mx_rotl32(a, 25);
    c ^= b; c -= mx_rotl32(b, 16);
    a ^= c; a -= mx_rotl32(c, 4);
    b ^= a; b -= mx_rotl32(a, 14);
    c ^= b; c -= mx_rotl32(b, 24);
    return c;
}

float mx_bits_to_01(uint bits) { return float(bits) / float(uint(0xffffffff)); }

float mx_fade(float t) { return t * t * t * (t * (t * 6.0 - 15.0) + 10.0); }

// ── Hash functions (1–5 integer inputs → uint) ────────────────

uint mx_hash_int(int x)
{
    uint len = 1u;
    uint seed = uint(0xdeadbeef) + (len << 2u) + 13u;
    return mx_bjfinal(seed + uint(x), seed, seed);
}

uint mx_hash_int(int x, int y)
{
    uint len = 2u;
    uint a, b, c;
    a = b = c = uint(0xdeadbeef) + (len << 2u) + 13u;
    a += uint(x);
    b += uint(y);
    return mx_bjfinal(a, b, c);
}

uint mx_hash_int(int x, int y, int z)
{
    uint len = 3u;
    uint a, b, c;
    a = b = c = uint(0xdeadbeef) + (len << 2u) + 13u;
    a += uint(x);
    b += uint(y);
    c += uint(z);
    return mx_bjfinal(a, b, c);
}

uint mx_hash_int(int x, int y, int z, int xx)
{
    uint len = 4u;
    uint a, b, c;
    a = b = c = uint(0xdeadbeef) + (len << 2u) + 13u;
    a += uint(x);
    b += uint(y);
    c += uint(z);
    mx_bjmix(a, b, c);
    a += uint(xx);
    return mx_bjfinal(a, b, c);
}

uint mx_hash_int(int x, int y, int z, int xx, int yy)
{
    uint len = 5u;
    uint a, b, c;
    a = b = c = uint(0xdeadbeef) + (len << 2u) + 13u;
    a += uint(x);
    b += uint(y);
    c += uint(z);
    mx_bjmix(a, b, c);
    a += uint(xx);
    b += uint(yy);
    return mx_bjfinal(a, b, c);
}

// ── Perlin noise ───────────────────────────────────────────────

/// Classic Perlin gradient noise [1] — trilinear interpolation of random gradients.
///
///   n(p) = Σ_{i∈corners} w(p−iᵢ) · g(h(iᵢ)) · (p − iᵢ)
///
/// where  iᵢ = integer lattice corners of the unit cell containing p
///        w   = Fade(t) = t³·(6t²−15t+10)   (quintic smoothstep)
///        g   = gradient from Bob Jenkins hash h(iᵢ)
/// Result is scaled by 0.9820 to approximately normalise output to [−1, 1].
float mx_perlin_noise_float(float3 p)
{
    int X, Y, Z;
    float fx = mx_floorfrac(p.x, X);
    float fy = mx_floorfrac(p.y, Y);
    float fz = mx_floorfrac(p.z, Z);
    float u = mx_fade(fx);
    float v = mx_fade(fy);
    float w = mx_fade(fz);
    float result = mx_trilerp(
        mx_gradient_float(mx_hash_int(X,   Y,   Z  ), fx,     fy,     fz    ),
        mx_gradient_float(mx_hash_int(X+1, Y,   Z  ), fx-1.0, fy,     fz    ),
        mx_gradient_float(mx_hash_int(X,   Y+1, Z  ), fx,     fy-1.0, fz    ),
        mx_gradient_float(mx_hash_int(X+1, Y+1, Z  ), fx-1.0, fy-1.0, fz    ),
        mx_gradient_float(mx_hash_int(X,   Y,   Z+1), fx,     fy,     fz-1.0),
        mx_gradient_float(mx_hash_int(X+1, Y,   Z+1), fx-1.0, fy,     fz-1.0),
        mx_gradient_float(mx_hash_int(X,   Y+1, Z+1), fx,     fy-1.0, fz-1.0),
        mx_gradient_float(mx_hash_int(X+1, Y+1, Z+1), fx-1.0, fy-1.0, fz-1.0),
        u, v, w);
    return mx_gradient_scale3d(result);
}

// ── Fractal noise (octaved Perlin / fBm) ─────────────────────

/// Fractional Brownian motion — summed octaves of Perlin noise [2].
///
///   fBm(p) = Σ_{i=0}^{octaves−1} amplitude_i · noise(p · lacunarity^i)
///
/// where  amplitude₀ = 1,  amplitudeᵢ = amplitudeᵢ₋₁ · diminish
///        lacunarity  = frequency multiplier per octave  (typically 2)
///        diminish    = amplitude decay per octave       (typically 0.5)
/// Each octave adds finer detail at higher frequency with lower amplitude.
float mx_fractal3d_noise_float(float3 p, int octaves, float lacunarity, float diminish)
{
    float result = 0.0;
    float amplitude = 1.0;
    for (int i = 0; i < octaves; ++i)
    {
        result += amplitude * mx_perlin_noise_float(p);
        amplitude *= diminish;
        p *= lacunarity;
    }
    return result;
}

void mx_fractal3d_float(float amplitude, int octaves, float lacunarity,
                         float diminish, float3 position, out float result)
{
    float value = mx_fractal3d_noise_float(position, octaves, lacunarity, diminish);
    result = value * amplitude;
}
