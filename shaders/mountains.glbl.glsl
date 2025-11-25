#define SC (250.0)

mat2 rotate2D(float r) {
    return mat2(cos(r), sin(r), -sin(r), cos(r));
}


vec2 fluid(vec2 uv){
 //add some turbulence
 //uv /=2.;
 //return vec2(sin(uv.x),cos(uv.y))/2.;
 
 
 //decrease this number to increase the turbulence of erosion
 float turbulence = 5.;
 //float t1 = turbulence*turbulence;
 
 //uv *= turbulence;
 //vec2 uv1 = uv/t1;
 
 vec2 result = uv;
 
 //more height variation
 //uv *= (1.+(sin(uv.x/128.)+cos(uv.y/128.))/128.);

 
 for (float i = 1.; i < 3.; i += 1.)
  {
    float s2 = (sin(uv.x*i))/turbulence;
    uv.y += s2;
    float s1 = cos(uv.y*i)/turbulence;
    uv.x += s1;
    uv=uv.yx;
  }
  return (uv-result).yx;
}


#define OCTAVES 8
vec3 fbm0(in vec2 uv,int octaves)
{
    //this function generates the terrain height
    //uv *= .75;
    float value = 0.;
    float amplitude = .75,n2=0.;
    //amplitude /= 16.;
    vec2 n1 = vec2(0.);
    
    //rotate 45 degrees
    //mat2 r = rotate2D(0.785398);
    
    float terrain_scale = 1.;
    uv *= terrain_scale;
    vec2 uv1 = uv;
    uv += uv;
    for (int i = 0; i < octaves; i++)
    {
        vec2 f1 = fluid(uv).yx;
        //domain warping
        uv += f1;
        
        n1 =
            //vec2(sin(uv.x+cos(uv.y*.37)/.37),cos(uv.y+sin(uv.x*.37)/.37))
            //vec2(sin(uv.x-n1.y*value),cos(uv.y-n1.x*value))
            //vec2(sin(uv.x),cos(uv.y))
            //vec2(sin(uv.x),cos(uv.y))*r //badlands
            //(vec2(sin(uv.x),cos(uv.y))+n1)/2.
            //(n1+abs(n1-vec2(sin(uv.x),cos(uv.y))))/2.
            f1
        ;
        uv1 = uv;
        n2 =
            n1.x+n1.y
            //((n1.x+n1.y)-n2)
            //n1.x+n1.y+n2*.37
        ;
        value -=
            abs(n2)*amplitude
            //abs(n2-value) * amplitude
        ;
        
        //waves
        //value += sin(uv.x/1000.+iTime)/10.;
        
        //erosion
        value = sqrt(value*value+.0001);
        
        amplitude *= .37;
        
        //This makes it somewhat more realistic
        //amplitude *= (1.+sin(uv.x/4.)/4.);

        //uv *= uv*2.05*r+vec2(0.,n2);
        uv += uv;
        
        //r = rotate2D(12.+(n2)*value/8.);
    }
    
    return vec3(value/terrain_scale,uv);
}

float fbm(vec2 uv,int octaves){
    return fbm0(uv,octaves).x;
}

/*
float fbm(in vec2 uv,int octaves){
uv /= 32.;
  vec3 col = vec3(0);
  for(int k = 0; k < 12; k++){
        //another interesting variant:
        //if(uv.y>uv.x) uv = uv.yx;
        vec2 offset =
            vec2(uv.x,-uv.y)*1.5
            //vec2(mod(uv.x*2.-.5,2.),mod(uv.y,2.)*2.)
            //vec2(mod(uv.x*2.,2.),sign(uv.x-uv.y)*mod(uv.y,2.)*2.)
        ;
        uv = abs(fract(uv.yx-offset)-.5);
        if(uv.y < uv.x) uv /= 1.5;
        //else uv /= 2.;
    }
    return uv.x/8.;
}
*/


//a more realistic terrain
float fbm1(in vec2 uv,int octaves){
    uv *= .8;
    return fbm(uv*2.,octaves)/2.+fbm(uv,octaves);
}


float fbm(in vec2 uv){
    return fbm(uv,OCTAVES);
}

float f(in vec3 p,int iters)
{   
    float h = fbm(p.xz,iters);
    return h;
}

float f(in vec3 p)
{   
    float h = fbm(p.xz,12);
    return h;
}

vec3 getNormal(vec3 p, float t)
{
    vec3 eps=vec3(.001 * t, .0, .0);
    vec3 n=vec3(f(p - eps.xyy) - f(p + eps.xyy),
                2. * eps.x,
                f(p - eps.yyx) - f(p + eps.yyx));
  
    return normalize(n);
}

float rayMarching(in vec3 ro, in vec3 rd, float tMin, float tMax)
{

    //TODO: use the local minimum in the previous frame to accelerate raymarching in the next frame
    //vec2 prev = vec2(0.);
    
    float t = tMin;
	for( int i = 0; i < 300; i++ )
	{
        vec3 pos = ro + t * rd;
		float h = pos.y - f(pos,OCTAVES);
        //if(prev.y<h && prev.y<prev.x) return t; //this is the local minimum
		if( abs(h) < (0.0015 * t) || t > tMax) 
            break;
		//prev = vec2(prev.y,h);
        t += 0.4 * h;
	}
	return t;
}

/*
//raymarching with LOD
float rayMarching(in vec3 ro, in vec3 rd, float tMin, float tMax)
{
    float t = tMin;
    int oct = 2;
	for( int i = 0; i < 300; i++ )
	{
        vec3 pos = ro + t * rd;
		float h = pos.y - f(pos,OCTAVES);
		if( abs(h) < (0.0015 * t) || t > tMax ) 
            if(oct < 8) oct += 2;
            else break;
		t += 0.4 * h;
	}

	return t;
}
*/

vec3 lighting(vec3 p, vec3 normal, vec3 L, vec3 V)
{
    vec3 sunColor = vec3(1., .956, .839);
    vec3 albedo = vec3(1.);
   	vec3 diff = max(dot(normal, L) * albedo, 0.);
    
    vec3 refl = normalize(reflect(L, normal));
    float spec = max(dot(refl, -normalize(V)), 0.);
    spec = pow(spec, 18.);
    spec = clamp(spec, 0., 1.);
    float sky = max(0.0, dot(vec3(0.,1.,0.), normal));
    
    //float amb = 0.5 * smoothstep(0.0, 2.0, p.y);
    
    vec3 col = diff * sunColor;
    col += spec * sunColor;
    col += sky * vec3(0., .6, 1.) * .1;
    //col += amb * .2;
    
   	return col;
}

mat3 lookAt(vec3 origin, vec3 target, float roll)
{
    vec3 rr = vec3(sin(roll), cos(roll), 0.0);
    vec3 ww = normalize(target - origin);
    vec3 uu = normalize(cross(ww, rr));
    vec3 vv = normalize(cross(uu, ww));

    return mat3(uu, vv, ww);
}

vec3 camerapath(float t)
{
    vec3 p=vec3(-13.0+3.5*cos(t),3.3,-1.1+2.4*cos(2.4*t+2.0));
	return p;
}

#define rot(spin) mat2(cos(spin),sin(spin),-sin(spin),cos(spin))

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    vec2 uv = (fragCoord - iResolution.xy * .5) / iResolution.y;
	
    vec3 lightDir = normalize(vec3(-.8, .15, -.3));
    
    vec3 camStep = vec3(lightDir.x, 0., lightDir.z) * (iTime/2.+12.)/4.;
    vec3 camPos = vec3(8., 2., 5.) + camStep;
    vec3 camTarget = vec3(1., 1., 4.) + camStep;

    
    mat3 mat = lookAt(camPos, camTarget, 0.0);
    
    vec3 ro = camPos;
    ro.y += fbm(ro.xz,OCTAVES)-1.8;

    vec3 rd = normalize(mat * vec3(uv.xy, 1.0));
    
    //rd.yx *= rot(0.785398/4.);
    //rd.xz *= rot(1.5*0.785398);
    
    if (length(iMouse.xy) > 40.0) {
        rd.yx *= rot(3.14*0.5-iMouse.y/iResolution.y*3.14);
        rd.xz *= rot(3.14-iMouse.x/iResolution.x*3.14*2.0-iTime/4.);
    }

    
    float tMin = .1;
    float tMax = 20.;
    float t = rayMarching(ro, rd, tMin, tMax);
    
    vec3 col = vec3(0.);
    
    if (t > tMax)
    {
        // from iq's shader, https://www.shadertoy.com/view/MdX3Rr
        float sundot = clamp(dot(rd, lightDir), 0.0, 1.0);
        col = vec3(0.3,0.5,0.85) - rd.y*rd.y*0.5;
        col = mix( col, 0.85*vec3(0.7,0.75,0.85), pow( 1.0-max(rd.y,0.0), 4.0 ) );
        // sun
		col += 0.25*vec3(1.0,0.7,0.4)*pow( sundot,5.0 );
		col += 0.25*vec3(1.0,0.8,0.6)*pow( sundot,64.0 );
		col += 0.2*vec3(1.0,0.8,0.6)*pow( sundot,512.0 );
        // clouds
		vec2 sc = ro.xz + rd.xz*(SC*1000.0-ro.y)/rd.y;
		col = mix( col, vec3(1.0,0.95,1.0), 0.5*smoothstep(0.5,0.8,fbm(0.0005*sc/SC)) );
        // horizon
        col = mix( col, 0.68*vec3(0.4,0.65,1.0), pow( 1.0-max(rd.y,0.0), 16.0 ) );
    }
    else
    {
        vec3 p = ro + rd * t;
        vec3 normal = getNormal(p, t);
        vec3 viewDir = normalize(ro - p);
        
        // lighting terrian
        col = lighting(p, normal, lightDir, viewDir);
        
        // fog
        float fo = 1.0-exp(-pow(30. * t/SC,1.5) );
        vec3 fco = 0.65*vec3(0.4,0.65,1.0);
        col = mix( col, fco, fo);
    }
    
    // Gama correction
    col = pow(clamp(col, 0., 1.), vec3(.45)); 
    
    fragColor = vec4(vec3(col), 1.0);
}