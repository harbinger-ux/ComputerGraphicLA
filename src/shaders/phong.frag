#version 330 core
in vec3 vertPos;
in vec3 normal_interp;
in vec3 colorInterp;
uniform vec3 light_pos;
out vec4 fragColor;
void main() {
    vec3 N = normalize(normal_interp);
    vec3 L = normalize(light_pos - vertPos);
    fragColor = vec4(colorInterp * max(dot(N, L), 0.0), 1.0);
}