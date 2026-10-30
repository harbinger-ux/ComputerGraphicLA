#version 330 core
out vec4 fragColor;
// Outputs bright Orange so Flat and Edge modes contrast heavily against the background
void main() { fragColor = vec4(1.0, 0.5, 0.0, 1.0); }