#version 330 core
layout(location = 0) in vec3 vertex;
layout(location = 1) in vec3 color;
layout(location = 2) in vec3 normal;
uniform mat4 projection, modelview;
out vec3 vertPos;
out vec3 normal_interp;
out vec3 colorInterp;
void main() {
    vec4 vertPos4 = modelview * vec4(vertex, 1.0);
    vertPos = vec3(vertPos4) / vertPos4.w;
    normal_interp = transpose(inverse(mat3(modelview))) * normal;
    colorInterp = color;
    gl_Position = projection * vertPos4;
}