/**
"Raining Blood, From a lacerated sky. Bleeding its horror, creating my
structure now I shall reign in blood." - Slayer

This is inspired by bigwings' heartfelt (https://www.shadertoy.com/view/ltffzl)
*/

#define PI 3.141592653589793

float hash21(vec2 p)
{
	uvec2 q = uvec2(ivec2(p)) * uvec2(1597334673U, 3812015801U);
	uint n = (q.x ^ q.y) * 1597334673U;
	return float(n) / float(0xffffffffU);
}

vec3 hash13(float p) {
   vec3 p3 = fract(vec3(p) * vec3(.1031,.11369,.13787));
   p3 += dot(p3, p3.yzx + 19.19);
   return fract(vec3((p3.x + p3.y)*p3.z, (p3.x+p3.z)*p3.y, (p3.y+p3.z)*p3.x));
}

float rainDrops(vec2 st, float time, float size)
{
    vec2 uv = st * size;
    uv.x *= iResolution.x / iResolution.y;
    vec2 gridUv = fract(uv) - .5; // grid
   	vec2 id = floor(uv);
    vec3 h = (hash13(id.x * 467.983 + id.y * 1294.387) - .5) * .8;
    vec2 dropUv = gridUv - h.xy;
    vec4 noise = textureLod(iChannel1, id * .05, 0.);
    float drop = smoothstep(.25, 0., length(dropUv)) *
        max(0., 1. - fract(time * (noise.b + .1) * .2 + noise.g) * 2.);
    return drop;
}

vec2 wigglyDrops(vec2 st, float time, float size)
{
    vec2 wigglyDropAspect = vec2(2., 1.);
    vec2 uv = st * size * wigglyDropAspect;
    uv.x *= iResolution.x / iResolution.y;
    uv.y += time * .23;

    vec2 gridUv = fract(uv) - .5; // rectangular grid
    vec2 id = floor(uv);
    
    float h = hash21(id);
    time += h * 2. * PI;
    float w = st.y * 10.;
    float dx = (h - .5) * .8;
    dx += (.3 - abs(dx)) * pow(sin(w), 2.) * sin(2. * w) *
        pow(cos(w), 3.) * 1.05; // wiggle
    float dy = -sin(time + sin(time + sin(time) * .5)) * .45; // slow down drop before continuing falling
    dy -= (gridUv.x - dx) * (gridUv.x - dx);
    
    vec2 dropUv = (gridUv - vec2(dx, dy)) / wigglyDropAspect;
    float drop = smoothstep(.06, .0, length(dropUv));
    
    vec2 trailUv = (gridUv - vec2(dx, time * .23)) / wigglyDropAspect;
    trailUv.y = (fract((trailUv.y) * 8.) - .5) / 8.;
    float trailDrop = smoothstep(.03, .0, length(trailUv));
    trailDrop *= smoothstep(-.05, .05, dropUv.y) * smoothstep(.4, dy, gridUv.y) *
        	(1.-step(.4, gridUv.y));
    
    float fogTrail = smoothstep(-.05, .05, dropUv.y) * smoothstep(.4, dy, gridUv.y) *
			smoothstep(.05, .01, abs(dropUv.x)) * (1.-step(.4, gridUv.y));
    
    return vec2(drop + trailDrop, fogTrail);
}

vec2 getDrops(vec2 st, float time)
{
    vec2 largeDrops = wigglyDrops(st, time * 2., 1.6);
    vec2 mediumDrops = wigglyDrops(st + 2.65, (time + 1296.675) * 1.4, 2.5);
    vec2 smallDrops = wigglyDrops(st - 1.67, time - 896.431, 3.6);
    float rain = rainDrops(st, time, 20.);
    
    vec2 drops;
    drops.y = max(largeDrops.y, max(mediumDrops.y, smallDrops.y));
    drops.x = smoothstep(.4, 2., (1. - drops.y) * rain + largeDrops.x +
                          mediumDrops.x + smallDrops.x); // drops kinda blend together

    return drops;
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    vec2 st = fragCoord / iResolution.xy;
    float time = mod(iTime + 100., 7200.);
    
    vec2 drops = getDrops(st, time);
    vec2 offset = drops.xy;
    float lod = (1. - drops.y) * 4.8;
    
    // This is kinda expensive, would love to use a cheaper method here.
    vec2 dropsX = getDrops(st + vec2(.001, 0.), time);
    vec2 dropsY = getDrops(st + vec2(0., .001), time);
    vec3 normal = vec3(dropsX.x - drops.x, dropsY.x - drops.x, 0.);
    normal.z = sqrt(1. - normal.x * normal.x - normal.y * normal.y);
    normal = normalize(normal);
    
    float lightning = sin(time * sin(time * 30.)); // screen flicker
    float lightningTime = mod(time, 10.) / 9.9;
   	lightning *= 1. - smoothstep(.0, .1, lightningTime)
        + smoothstep(.9, 1., lightningTime); // lightning flash mask
    
	vec3 col = textureLod(iChannel0, st+normal.xy * 3., lod).rgb;
    col *= (1. + lightning);
    
    col *= vec3(1., .8, .7); // slight red-ish tint
    col += (drops.y > 0. ? vec3(.5, -.1, -.15)*drops.y : vec3(0.)); // bloody trails
    col *= (drops.x > 0. ? vec3(.8, .2, .1) * (1.-drops.x) : vec3(1.)); // blood colored drops
    
    col = mix(col, col*smoothstep(.8, .35, length(st - .5)), .6); // vignette
    
    fragColor = vec4(col, 1.0);
}