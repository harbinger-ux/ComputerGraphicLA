import os

os.makedirs('src/shaders', exist_ok=True)

shaders = {
    'flat.vert': """#version 330 core
layout(location = 0) in vec3 vertex;
uniform mat4 projection, modelview;
void main() { gl_Position = projection * modelview * vec4(vertex, 1.0); }""",

    'flat.frag': """#version 330 core
out vec4 fragColor;
// Outputs bright Orange so Flat and Edge modes contrast heavily against the background
void main() { fragColor = vec4(1.0, 0.5, 0.0, 1.0); }""",

    'interp.vert': """#version 330 core
layout(location = 0) in vec3 vertex;
layout(location = 1) in vec3 color;
uniform mat4 projection, modelview;
out vec3 color_interp;
void main() { color_interp = color; gl_Position = projection * modelview * vec4(vertex, 1.0); }""",

    'interp.frag': """#version 330 core
in vec3 color_interp;
out vec4 fragColor;
void main() { fragColor = vec4(color_interp, 1.0); }""",

    'gouraud.vert': """#version 330 core
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
}""",

    'gouraud.frag': """#version 330 core
in vec3 color_interp;
out vec4 fragColor;
void main() { fragColor = vec4(color_interp, 1.0); }""",

    'phong.vert': """#version 330 core
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
}""",

    'phong.frag': """#version 330 core
in vec3 vertPos;
in vec3 normal_interp;
in vec3 colorInterp;
uniform vec3 light_pos;
out vec4 fragColor;
void main() {
    vec3 N = normalize(normal_interp);
    vec3 L = normalize(light_pos - vertPos);
    fragColor = vec4(colorInterp * max(dot(N, L), 0.0), 1.0);
}""",

    'blinn.frag': """#version 330 core
precision mediump float;
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
    if (diff > 0.0) { spec = pow(max(dot(N, H), 0.0), shininess); }
    vec3 g = vec3(diff, spec, 1.0);
    vec3 rgb = (matrixCompMult(K_materials, I_light) * g) * colorInterp;
    fragColor = vec4(rgb, 1.0);
}""",

    'phong_texture.vert': """#version 330 core
layout(location = 0) in vec3 vertex;
layout(location = 1) in vec3 color;
layout(location = 2) in vec3 normal;
layout(location = 3) in vec2 texcoord;
uniform mat4 projection, modelview;
out vec3 normal_interp;
out vec3 color_interp;
out vec3 vert_pos;
out vec2 texcoord_interp;
void main(){
  color_interp = color;
  vec4 vert_pos4 = modelview * vec4(vertex, 1.0);
  vert_pos = vec3(vert_pos4) / vert_pos4.w;
  mat3 normal_matrix = transpose(inverse(mat3(modelview)));
  normal_interp = normal_matrix * normal;
  texcoord_interp = texcoord;
  gl_Position = projection * vert_pos4;
}""",

    'phong_texture.frag': """#version 330 core
precision mediump float;
in vec3 normal_interp;
in vec3 vert_pos;
in vec3 color_interp;
in vec2 texcoord_interp;
uniform mat3 K_materials;
uniform mat3 I_light;
uniform float phong_factor;
uniform float shininess;
uniform vec3 light_pos;
uniform sampler2D texture1;
out vec4 fragColor;
void main() {
  vec3 N = normalize(normal_interp);
  vec3 L = normalize(light_pos - vert_pos);
  vec3 R = reflect(-L, N);
  vec3 V = normalize(-vert_pos);
  vec3 lv = light_pos - vert_pos;
  float lvd = 1.0/(dot(lv, lv));
  float specAngle = max(dot(R, V), 0.0);
  float specular = pow(specAngle, shininess);
  vec3 g = vec3(lvd*max(dot(L, N), 0.0), specular, 1.0);
  vec3 rgb = matrixCompMult(K_materials, I_light) * g;
  fragColor = vec4(rgb, 1.0);
  vec4 color_interp4 = vec4(color_interp, 1.0);
  float color_factor = 0.2;
  float texture_factor = 1.0 - (color_factor + phong_factor);
  fragColor = color_factor*color_interp4 + phong_factor*fragColor + texture_factor*texture(texture1, texcoord_interp);
}"""
}

for name, src in shaders.items():
    with open(f'src/shaders/{name}', 'w') as f:
        f.write(src)
print("Shaders generated securely in src/shaders/!")