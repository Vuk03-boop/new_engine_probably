// Emitter next-event estimation (ADR-0005 Amendment 3), shared by `reference.slang` (4A) and
// `shade.slang` (4B) so both run one definition (host mirror: `light::reference::emitter_sample`).
// Included after emitters_common.slang, once the includer has declared
// `StructuredBuffer<Emitter> emitters` and `bool occluded_segment(float3 origin, float3 seg)`.
//
// One emitter by the alias table (a uniform index, then the coin), a point uniform in its solid angle
// (or on its area below SOLID_ANGLE_MIN or with `area`), one visibility segment to the point offset
// 1/256 voxel off the emitter. Returns L_e cos_x S / (P(i) pi), without beta and the albedo. Draws in
// order: the index (one or more next_u32), the coin (one), then u and v, whatever the outcome.

float3 emitter_sample(float3 p, float3 n, uint count, bool area, bool fault_no_solid_angle, bool fault_no_emitter_cos, inout uint rng)
{
    uint i = uniform_index(count, rng);
    Emitter e = emitters[i];
    if ((next_u32(rng) >> 8u) >= e.meta.x)
        e = emitters[e.meta.y];
    float u = rnd(rng);
    float v = rnd(rng);
    float3 ne = face_normal(e.meta.z);
    if (dot(ne, p - e.p0_area.xyz) <= 0.0)
        return float3(0.0, 0.0, 0.0);
    bool solid = false;
    SphRect r;
    if (!area)
    {
        r = sph_rect(e.p0_area.xyz, e.eu_pdf.xyz, e.ev.xyz, p);
        solid = r.solid_angle >= SOLID_ANGLE_MIN;
    }
    float3 y = solid ? sph_rect_sample(r, u, v) : e.p0_area.xyz + u * e.eu_pdf.xyz + v * e.ev.xyz;
    float3 d = y - p;
    float d2 = dot(d, d);
    float3 w = d * rsqrt(d2);
    float cx = dot(n, w);
    float cy = -dot(ne, w);
    if (cx <= 0.0 || cy <= 0.0)
        return float3(0.0, 0.0, 0.0);
    if (occluded_segment(p, y + ne * RAY_OFFSET - p))
        return float3(0.0, 0.0, 0.0);
    float k;
    if (fault_no_solid_angle)
        k = e.p0_area.w;
    else if (solid)
        k = r.solid_angle;
    else if (fault_no_emitter_cos)
        k = e.p0_area.w / d2;
    else
        k = e.p0_area.w * cy / d2;
    return e.radiance.xyz * (cx * k / (e.eu_pdf.w * PI));
}
