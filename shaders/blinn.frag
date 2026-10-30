#version 330 core

precision mediump float;

// Blinn-Phong: specular uses H = normalize(L + V). Pair with any phong.vert
// that writes normal_interp, vertPos, colorInterp.

in vec3 normal_interp;
in vec3 vertPos;
in vec3 colorInterp;

uniform mat3 K_materials;
uniform mat3 I_light;
uniform float shininess;
uniform vec3 light_pos;

out vec4 fragColor;

void main() {
    vec3 N = normalize(normal_interp);
    vec3 L = normalize(light_pos - vertPos);
    vec3 V = normalize(-vertPos);
    vec3 H = normalize(L + V);

    float diff = max(dot(N, L), 0.0);
    float spec = 0.0;
    if (diff > 0.0) {
        spec = pow(max(dot(N, H), 0.0), shininess);
    }

    vec3 g = vec3(diff, spec, 1.0);
    vec3 rgb = (matrixCompMult(K_materials, I_light) * g) * colorInterp;
    fragColor = vec4(rgb, 1.0);
}
