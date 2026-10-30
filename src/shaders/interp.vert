#version 330 core
layout(location = 0) in vec3 vertex;
layout(location = 1) in vec3 color;
uniform mat4 projection, modelview;
out vec3 color_interp;
void main() { color_interp = color; gl_Position = projection * modelview * vec4(vertex, 1.0); }