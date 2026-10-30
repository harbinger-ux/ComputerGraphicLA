#version 330 core
layout(location = 0) in vec3 vertex;
layout(location = 1) in vec3 color;
layout(location = 2) in vec3 normal;
uniform mat4 projection, modelview;
uniform vec3 light_pos;
out vec3 color_interp;
void main() {
    vec4 vertPos4 = modelview * vec4(vertex, 1.0);
    vec3 vertPos = vec3(vertPos4) / vertPos4.w;
    vec3 N = normalize(transpose(inverse(mat3(modelview))) * normal);
    vec3 L = normalize(light_pos - vertPos);
    color_interp = color * max(dot(N, L), 0.0);
    gl_Position = projection * vertPos4;
}