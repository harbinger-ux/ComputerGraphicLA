#version 330 core
in vec3 color_interp;
out vec4 fragColor;
void main() { fragColor = vec4(color_interp, 1.0); }