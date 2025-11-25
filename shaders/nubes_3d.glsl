// Vuelo entre nubes volumétrico
// Usa scale para tamaño de nubes y speed para velocidad de vuelo

float hash(vec3 p) {
    p = fract(p * 0.3183099 + 0.1);
    p *= 17.0;
    return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
}

float noise(in vec3 x) {
    vec3 i = floor(x);
    vec3 f = fract(x);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(mix(hash(i + vec3(0, 0, 0)), hash(i + vec3(1, 0, 0)), f.x),
                   mix(hash(i + vec3(0, 1, 0)), hash(i + vec3(1, 1, 0)), f.x), f.y),
               mix(mix(hash(i + vec3(0, 0, 1)), hash(i + vec3(1, 0, 1)), f.x),
                   mix(hash(i + vec3(0, 1, 1)), hash(i + vec3(1, 1, 1)), f.x), f.y), f.z);
}

float map(in vec3 p) {
    // Aquí speed mueve la cámara en Z
    vec3 q = p - vec3(0.0, 0.0, 1.0) * iTime * speed;
    float f;
    // Scale afecta al tamaño del ruido
    f = 0.50000 * noise(q * scale); q = q * 2.02;
    f += 0.25000 * noise(q * scale); q = q * 2.03;
    f += 0.12500 * noise(q * scale);
    return clamp(1.5 - p.y - 2.0 + 1.75 * f, 0.0, 1.0);
}

void mainImage( out vec4 fragColor, in vec2 fragCoord ) {
    vec2 p = (fragCoord.xy - 0.5 * iResolution.xy) / iResolution.y;
    vec3 ro = vec3(0.0, 0.0, 0.0);
    vec3 rd = normalize(vec3(p, -1.0));

    vec4 sum = vec4(0.0);
    float t = 0.0;
    for (int i = 0; i < 64; i++) {
        vec3 pos = ro + t * rd;
        float den = map(pos);
        if (den > 0.01) {
            float dif = clamp((den - map(pos + 0.3)) / 0.6, 0.0, 1.0);
            vec3 col = vec3(0.9, 0.9, 0.95) + vec3(0.1, 0.1, 0.2) * dif;
            col *= den;
            sum = sum + vec4(col, 1.0) * (1.0 - sum.a) * den;
        }
        t += max(0.1, 0.05 * t);
        if (sum.a > 0.99) break;
    }
    vec3 sky = vec3(0.6, 0.71, 0.75) - p.y * 0.2;
    fragColor = vec4(mix(sky, sum.xyz, sum.a), 1.0);
}