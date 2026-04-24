// SDF-based human head using Loomis proportions.
//
// Conventions — one unit = approx. half a head height.
// The head's canonical bounds are roughly:
//   x:  [-0.85, +0.85]   (sides, including ears)
//   y:  [-0.97, +1.05]   (chin to crown; neck extends further down)
//   z:  [-0.90, +1.00]   (back of skull to nose tip)
// Eye line sits at y≈+0.10, mouth at y≈-0.47, so the visual centroid
// of the face is near the origin (which is what the orbit camera targets).
//
// This is still an SDF approximation — for true scanned-mesh fidelity
// (GPU Gems 3 Ch. 14 uses a laser-scanned head), we'd need a triangle
// mesh + BVH + ray-triangle intersection pipeline.

import common;

// ── SDF primitives ──────────────────────────────────────────────

float sdSphere(float3 p, float radius)
{
    return length(p) - radius;
}

float sdEllipsoid(float3 p, float3 radii)
{
    float k0 = length(p / radii);
    float k1 = length(p / (radii * radii));
    return k0 * (k0 - 1.0) / k1;
}

float sdCapsule(float3 p, float3 a, float3 b, float r)
{
    float3 ab = b - a;
    float t = clamp(dot(p - a, ab) / dot(ab, ab), 0.0, 1.0);
    return length(p - a - ab * t) - r;
}

float sdRoundBox(float3 p, float3 b, float r)
{
    float3 q = abs(p) - b;
    return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0) - r;
}

// ── Smooth boolean operators ────────────────────────────────────

float smin(float a, float b, float k)
{
    float h = max(k - abs(a - b), 0.0) / k;
    return min(a, b) - h * h * k * 0.25;
}

float smax(float a, float b, float k)
{
    return -smin(-a, -b, k);
}

// Helper: mirrored primitive along x (for left+right feature pairs).
float mirroredEllipsoid(float3 p, float3 center, float3 radii)
{
    float3 q = float3(abs(p.x), p.yz) - center;
    return sdEllipsoid(q, radii);
}

float mirroredSphere(float3 p, float3 center, float radius)
{
    float3 q = float3(abs(p.x), p.yz) - center;
    return sdSphere(q, radius);
}

// ── Head SDF ────────────────────────────────────────────────────

float sdHead(float3 p)
{
    // 1. Main cranium — egg-shaped, tapered at chin end.
    float d = sdEllipsoid(p - float3(0, 0.25, -0.05), float3(0.72, 0.78, 0.82));

    // 2. Forehead shelf — pushes the front of the skull forward.
    float forehead = sdEllipsoid(p - float3(0, 0.45, 0.40), float3(0.58, 0.30, 0.30));
    d = smin(d, forehead, 0.20);

    // 3. Face plate — the front half of the head.
    float face = sdEllipsoid(p - float3(0, -0.05, 0.30), float3(0.58, 0.58, 0.58));
    d = smin(d, face, 0.30);

    // 4. Temples — thinner sides above the cheekbones.
    float temple = mirroredEllipsoid(p, float3(0.60, 0.28, 0.20), float3(0.08, 0.22, 0.30));
    d = smin(d, temple, 0.20);

    // 5. Cheekbones (zygomatic) — wide midface.
    float cheek = mirroredEllipsoid(p, float3(0.36, -0.02, 0.42), float3(0.19, 0.16, 0.22));
    d = smin(d, cheek, 0.14);

    // 6. Infraorbital pad — under-eye swell.
    float infra = mirroredEllipsoid(p, float3(0.22, 0.00, 0.60), float3(0.12, 0.08, 0.10));
    d = smin(d, infra, 0.08);

    // 7. Brow ridge — prominent bone above the eyes.
    float brow = mirroredEllipsoid(p, float3(0.20, 0.22, 0.65), float3(0.22, 0.06, 0.09));
    d = smin(d, brow, 0.05);

    // 8. Jaw — mandible with visible angle.
    float jaw = mirroredEllipsoid(p, float3(0.30, -0.52, 0.18), float3(0.20, 0.24, 0.30));
    d = smin(d, jaw, 0.14);

    // 9. Chin.
    float chin = sdEllipsoid(p - float3(0, -0.78, 0.32), float3(0.17, 0.15, 0.17));
    d = smin(d, chin, 0.10);

    // 10. Neck — tapered down into the shoulders.
    float neck = sdCapsule(
        p,
        float3(0, -0.85, -0.08),
        float3(0, -1.80, -0.15),
        0.28
    );
    d = smin(d, neck, 0.22);

    // 11. Eye sockets — carved recess.
    float socket = mirroredSphere(p, float3(0.22, 0.08, 0.72), 0.15);
    d = smax(d, -socket, 0.08);

    // 12. Eyeballs — spherical, sit inside the sockets.
    float eye = mirroredSphere(p, float3(0.22, 0.10, 0.75), 0.11);
    d = min(d, eye);

    // 13. Nose bridge — tapered capsule from brow to tip.
    float bridge = sdCapsule(
        p,
        float3(0, 0.16, 0.75),
        float3(0, -0.15, 0.92),
        0.055
    );
    d = smin(d, bridge, 0.07);

    // 14. Nose tip (ball of nose).
    float noseTip = sdEllipsoid(p - float3(0, -0.22, 0.97), float3(0.085, 0.080, 0.090));
    d = smin(d, noseTip, 0.04);

    // 15. Nostril wings.
    float wing = mirroredEllipsoid(p, float3(0.09, -0.24, 0.88), float3(0.055, 0.055, 0.065));
    d = smin(d, wing, 0.03);

    // 16. Nostril holes (carved).
    float nostril = mirroredSphere(p, float3(0.065, -0.27, 0.93), 0.023);
    d = smax(d, -nostril, 0.005);

    // 17. Philtrum — vertical groove between nose and lip.
    float philtrum = sdRoundBox(p - float3(0, -0.37, 0.84), float3(0.020, 0.060, 0.035), 0.005);
    d = smax(d, -philtrum, 0.02);

    // 18. Upper lip.
    float upperLip = sdEllipsoid(p - float3(0, -0.45, 0.82), float3(0.165, 0.035, 0.060));
    d = smin(d, upperLip, 0.02);

    // 19. Cupid's bow — small notch in upper lip.
    float cupid = sdRoundBox(p - float3(0, -0.435, 0.86), float3(0.028, 0.016, 0.018), 0.004);
    d = smax(d, -cupid, 0.007);

    // 20. Lower lip — fuller than upper.
    float lowerLip = sdEllipsoid(p - float3(0, -0.53, 0.80), float3(0.145, 0.045, 0.060));
    d = smin(d, lowerLip, 0.022);

    // 21. Mouth line — thin separator.
    float mouthLine = sdRoundBox(p - float3(0, -0.495, 0.84), float3(0.155, 0.004, 0.022), 0.002);
    d = smax(d, -mouthLine, 0.008);

    // 22. Mentolabial sulcus — chin crease below lower lip.
    float chinCrease = sdRoundBox(p - float3(0, -0.64, 0.72), float3(0.075, 0.016, 0.028), 0.004);
    d = smax(d, -chinCrease, 0.018);

    // 23. Ears — thin shell.
    float ear = mirroredEllipsoid(p, float3(0.80, 0.05, -0.05), float3(0.050, 0.20, 0.13));
    d = smin(d, ear, 0.04);

    // 24. Ear concha — carved bowl.
    float concha = mirroredSphere(p, float3(0.77, 0.02, -0.05), 0.065);
    d = smax(d, -concha, 0.025);

    // 25. Ear lobes.
    float earLobe = mirroredSphere(p, float3(0.80, -0.14, -0.02), 0.050);
    d = smin(d, earLobe, 0.025);

    return d;
}

// ── Normal via central differences ──────────────────────────────

float3 calcHeadNormal(float3 p)
{
    float2 e = float2(0.0005, 0.0);
    return normalize(float3(
        sdHead(p + e.xyy) - sdHead(p - e.xyy),
        sdHead(p + e.yxy) - sdHead(p - e.yxy),
        sdHead(p + e.yyx) - sdHead(p - e.yyx)
    ));
}

// ── Ray marching ────────────────────────────────────────────────

static const int   MAX_MARCH_STEPS = 128;
static const float MARCH_EPSILON   = 0.0005;
static const float MARCH_MAX_DIST  = 20.0;

HitInfo marchHead(Ray ray)
{
    HitInfo info;
    info.hit        = false;
    info.t          = 0.0;
    info.backT      = -1.0;
    info.tangent    = float3(1, 0, 0);
    info.hasTangent = false;

    float t = 0.01;
    for (int i = 0; i < MAX_MARCH_STEPS; i++)
    {
        float3 p = ray.origin + ray.direction * t;
        float d = sdHead(p);

        if (d < MARCH_EPSILON)
        {
            info.hit      = true;
            info.t        = t;
            info.position = p;
            info.normal   = calcHeadNormal(p);
            float3 lp     = normalize(p - float3(0, 0.05, 0));
            info.uv       = float2(
                0.5 + atan2(lp.z, lp.x) / (2.0 * PI),
                0.5 - asin(clamp(lp.y, -1.0, 1.0)) / PI
            );

            // Continue marching to find the back-face exit.
            float tBack = t + 0.01;
            for (int j = 0; j < MAX_MARCH_STEPS; j++)
            {
                float3 pb = ray.origin + ray.direction * tBack;
                float db = sdHead(pb);
                if (db > MARCH_EPSILON)
                {
                    info.backT = tBack;
                    break;
                }
                tBack += max(abs(db), 0.001);
                if (tBack > MARCH_MAX_DIST)
                    break;
            }

            return info;
        }

        t += d;
        if (t > MARCH_MAX_DIST)
            break;
    }

    return info;
}
