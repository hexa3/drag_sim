/*
    drag_sim_gl.cpp  –  Real-time drag simulation with OpenGL

    A cube falls through a chosen atmospheric layer.
    Drag force: Fd = 0.5 * rho * v^2 * Cd * A
    Net accel : a  = g - Fd/m

    Inspired by kavan's "Simulating Atoms in C++" – particle physics
    visualised in a minimal, live OpenGL window.

    Dependencies : GLFW3, GLEW, GLM
    Compile:
        g++ -O2 -std=c++17 drag_sim_gl.cpp \
            -lGL -lGLEW -lglfw -o drag_sim_gl
    Run:
        ./drag_sim_gl
*/

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>
#include <sstream>
#include <iomanip>
#include <algorithm>

// ─────────────────────────────────────────────────────────────
//  Constants / layer table
// ─────────────────────────────────────────────────────────────
static const int W = 900, H = 700;
static const float G = 9.81f;

struct Layer {
    const char* name;
    float rho;          // kg/m^3
    float r, g, b;      // tint colour for the sky gradient
};

static const Layer LAYERS[] = {
    { "Troposphere  (0-12 km)",   1.225f,  0.10f, 0.18f, 0.35f },
    { "Stratosphere (12-50 km)",  0.089f,  0.05f, 0.08f, 0.20f },
    { "Mesosphere   (50-85 km)",  0.001f,  0.02f, 0.02f, 0.08f },
    { "Thermosphere (85-600 km)", 1e-7f,   0.00f, 0.00f, 0.04f },
    { "Exosphere    (600+ km)",   1e-12f,  0.00f, 0.00f, 0.00f },
};
static const int NUM_LAYERS = 5;

// ─────────────────────────────────────────────────────────────
//  Simulation state
// ─────────────────────────────────────────────────────────────
struct Sim {
    float mass   = 10.f;
    float side   = 0.30f;
    float Cd     = 1.05f;
    float rho    = 1.225f;

    float v      = 0.f;     // velocity  (m/s, downward +)
    float x      = 0.f;     // position  (m,   downward +)
    float t      = 0.f;     // elapsed time

    float Fd     = 0.f;
    float accel  = 0.f;

    bool  running = false;
    bool  paused  = false;

    // history for graph
    static const int HIST = 512;
    float vHist[HIST] = {};
    float tHist[HIST] = {};
    int   hHead = 0, hCount = 0;

    float terminalV() const {
        float A = side * side;
        return std::sqrt((2.f * mass * G) / (rho * Cd * A));
    }

    void step(float dt) {
        float A  = side * side;
        Fd    = 0.5f * rho * v * std::fabs(v) * Cd * A;
        accel = G - Fd / mass;
        v    += accel * dt;
        x    += v     * dt;
        t    += dt;

        vHist[hHead] = v;
        tHist[hHead] = t;
        hHead = (hHead + 1) % HIST;
        if (hCount < HIST) hCount++;
    }

    void reset() {
        v = x = t = Fd = accel = 0.f;
        hHead = hCount = 0;
        std::memset(vHist, 0, sizeof vHist);
        std::memset(tHist, 0, sizeof tHist);
        running = paused = false;
    }
};

// ─────────────────────────────────────────────────────────────
//  Shaders
// ─────────────────────────────────────────────────────────────
static const char* VS_CUBE = R"(
#version 330 core
layout(location=0) in vec3 pos;
layout(location=1) in vec3 nrm;
uniform mat4 MVP;
uniform mat4 M;
out vec3 fragNrm;
out vec3 fragPos;
void main(){
    gl_Position = MVP * vec4(pos,1.0);
    fragNrm = mat3(M) * nrm;
    fragPos = (M * vec4(pos,1.0)).xyz;
}
)";

static const char* FS_CUBE = R"(
#version 330 core
in vec3 fragNrm;
in vec3 fragPos;
uniform vec3 cubeColor;
uniform float glowAmount;
out vec4 outColor;
void main(){
    vec3 L = normalize(vec3(1.0,2.0,1.5));
    float diff = max(dot(normalize(fragNrm), L), 0.0);
    vec3 ambient = cubeColor * 0.25;
    vec3 diffuse = cubeColor * diff * 0.75;
    vec3 col = ambient + diffuse + cubeColor * glowAmount * 0.4;
    outColor = vec4(col, 1.0);
}
)";

// flat shader for lines / quads
static const char* VS_FLAT = R"(
#version 330 core
layout(location=0) in vec2 pos;
uniform mat4 P;
void main(){ gl_Position = P * vec4(pos,0.0,1.0); }
)";

static const char* FS_FLAT = R"(
#version 330 core
uniform vec4 color;
out vec4 outColor;
void main(){ outColor = color; }
)";

GLuint compileShader(GLenum type, const char* src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    return s;
}
GLuint makeProgram(const char* vs, const char* fs) {
    GLuint p = glCreateProgram();
    GLuint v = compileShader(GL_VERTEX_SHADER,   vs);
    GLuint f = compileShader(GL_FRAGMENT_SHADER, fs);
    glAttachShader(p, v); glAttachShader(p, f);
    glLinkProgram(p);
    glDeleteShader(v); glDeleteShader(f);
    return p;
}

// ─────────────────────────────────────────────────────────────
//  Cube geometry
// ─────────────────────────────────────────────────────────────
static const float CUBE_VERTS[] = {
    // pos              normal
    -1,-1,-1,  0, 0,-1,   1,-1,-1,  0, 0,-1,   1, 1,-1,  0, 0,-1,  -1, 1,-1,  0, 0,-1,
    -1,-1, 1,  0, 0, 1,   1,-1, 1,  0, 0, 1,   1, 1, 1,  0, 0, 1,  -1, 1, 1,  0, 0, 1,
    -1,-1,-1, -1, 0, 0,  -1, 1,-1, -1, 0, 0,  -1, 1, 1, -1, 0, 0,  -1,-1, 1, -1, 0, 0,
     1,-1,-1,  1, 0, 0,   1, 1,-1,  1, 0, 0,   1, 1, 1,  1, 0, 0,   1,-1, 1,  1, 0, 0,
    -1,-1,-1,  0,-1, 0,   1,-1,-1,  0,-1, 0,   1,-1, 1,  0,-1, 0,  -1,-1, 1,  0,-1, 0,
    -1, 1,-1,  0, 1, 0,   1, 1,-1,  0, 1, 0,   1, 1, 1,  0, 1, 0,  -1, 1, 1,  0, 1, 0,
};
static const unsigned short CUBE_IDX[] = {
     0, 1, 2,  0, 2, 3,   4, 5, 6,  4, 6, 7,
     8, 9,10,  8,10,11,  12,13,14, 12,14,15,
    16,17,18, 16,18,19,  20,21,22, 20,22,23,
};

// ─────────────────────────────────────────────────────────────
//  Simple bitmap font  (5×7 per char, ASCII 32-126)
//  We draw text purely via GL_POINTS / quads to avoid freetype.
//  Each character is rendered as a 5-wide × 7-tall pixel grid.
// ─────────────────────────────────────────────────────────────
// 5x7 pixel font – a minimal printable-ASCII bitmap font.
// Encoded as 5 bytes per char (each byte = one column, bit0=top).
static const uint8_t FONT5x7[][5] = {
    {0x00,0x00,0x00,0x00,0x00}, // space
    {0x00,0x00,0x5f,0x00,0x00}, // !
    {0x00,0x07,0x00,0x07,0x00}, // "
    {0x14,0x7f,0x14,0x7f,0x14}, // #
    {0x24,0x2a,0x7f,0x2a,0x12}, // $
    {0x23,0x13,0x08,0x64,0x62}, // %
    {0x36,0x49,0x55,0x22,0x50}, // &
    {0x00,0x05,0x03,0x00,0x00}, // '
    {0x00,0x1c,0x22,0x41,0x00}, // (
    {0x00,0x41,0x22,0x1c,0x00}, // )
    {0x08,0x2a,0x1c,0x2a,0x08}, // *
    {0x08,0x08,0x3e,0x08,0x08}, // +
    {0x00,0x50,0x30,0x00,0x00}, // ,
    {0x08,0x08,0x08,0x08,0x08}, // -
    {0x00,0x60,0x60,0x00,0x00}, // .
    {0x20,0x10,0x08,0x04,0x02}, // /
    {0x3e,0x51,0x49,0x45,0x3e}, // 0
    {0x00,0x42,0x7f,0x40,0x00}, // 1
    {0x42,0x61,0x51,0x49,0x46}, // 2
    {0x21,0x41,0x45,0x4b,0x31}, // 3
    {0x18,0x14,0x12,0x7f,0x10}, // 4
    {0x27,0x45,0x45,0x45,0x39}, // 5
    {0x3c,0x4a,0x49,0x49,0x30}, // 6
    {0x01,0x71,0x09,0x05,0x03}, // 7
    {0x36,0x49,0x49,0x49,0x36}, // 8
    {0x06,0x49,0x49,0x29,0x1e}, // 9
    {0x00,0x36,0x36,0x00,0x00}, // :
    {0x00,0x56,0x36,0x00,0x00}, // ;
    {0x00,0x08,0x14,0x22,0x41}, // <
    {0x14,0x14,0x14,0x14,0x14}, // =
    {0x41,0x22,0x14,0x08,0x00}, // >
    {0x02,0x01,0x51,0x09,0x06}, // ?
    {0x32,0x49,0x79,0x41,0x3e}, // @
    {0x7e,0x11,0x11,0x11,0x7e}, // A
    {0x7f,0x49,0x49,0x49,0x36}, // B
    {0x3e,0x41,0x41,0x41,0x22}, // C
    {0x7f,0x41,0x41,0x22,0x1c}, // D
    {0x7f,0x49,0x49,0x49,0x41}, // E
    {0x7f,0x09,0x09,0x01,0x01}, // F
    {0x3e,0x41,0x41,0x49,0x7a}, // G
    {0x7f,0x08,0x08,0x08,0x7f}, // H
    {0x00,0x41,0x7f,0x41,0x00}, // I
    {0x20,0x40,0x41,0x3f,0x01}, // J
    {0x7f,0x08,0x14,0x22,0x41}, // K
    {0x7f,0x40,0x40,0x40,0x40}, // L
    {0x7f,0x02,0x04,0x02,0x7f}, // M
    {0x7f,0x04,0x08,0x10,0x7f}, // N
    {0x3e,0x41,0x41,0x41,0x3e}, // O
    {0x7f,0x09,0x09,0x09,0x06}, // P
    {0x3e,0x41,0x51,0x21,0x5e}, // Q
    {0x7f,0x09,0x19,0x29,0x46}, // R
    {0x46,0x49,0x49,0x49,0x31}, // S
    {0x01,0x01,0x7f,0x01,0x01}, // T
    {0x3f,0x40,0x40,0x40,0x3f}, // U
    {0x1f,0x20,0x40,0x20,0x1f}, // V
    {0x7f,0x20,0x18,0x20,0x7f}, // W
    {0x63,0x14,0x08,0x14,0x63}, // X
    {0x03,0x04,0x78,0x04,0x03}, // Y
    {0x61,0x51,0x49,0x45,0x43}, // Z
    {0x00,0x00,0x7f,0x41,0x00}, // [
    {0x02,0x04,0x08,0x10,0x20}, // backslash
    {0x00,0x41,0x7f,0x00,0x00}, // ]
    {0x04,0x02,0x01,0x02,0x04}, // ^
    {0x40,0x40,0x40,0x40,0x40}, // _
    {0x00,0x01,0x02,0x04,0x00}, // `
    {0x20,0x54,0x54,0x54,0x78}, // a
    {0x7f,0x48,0x44,0x44,0x38}, // b
    {0x38,0x44,0x44,0x44,0x20}, // c
    {0x38,0x44,0x44,0x48,0x7f}, // d
    {0x38,0x54,0x54,0x54,0x18}, // e
    {0x08,0x7e,0x09,0x01,0x02}, // f
    {0x08,0x14,0x54,0x54,0x3c}, // g
    {0x7f,0x08,0x04,0x04,0x78}, // h
    {0x00,0x44,0x7d,0x40,0x00}, // i
    {0x20,0x40,0x44,0x3d,0x00}, // j
    {0x00,0x7f,0x10,0x28,0x44}, // k
    {0x00,0x41,0x7f,0x40,0x00}, // l
    {0x7c,0x04,0x18,0x04,0x78}, // m
    {0x7c,0x08,0x04,0x04,0x78}, // n
    {0x38,0x44,0x44,0x44,0x38}, // o
    {0x7c,0x14,0x14,0x14,0x08}, // p
    {0x08,0x14,0x14,0x18,0x7c}, // q
    {0x7c,0x08,0x04,0x04,0x08}, // r
    {0x48,0x54,0x54,0x54,0x20}, // s
    {0x04,0x3f,0x44,0x40,0x20}, // t
    {0x3c,0x40,0x40,0x40,0x3c}, // u
    {0x1c,0x20,0x40,0x20,0x1c}, // v
    {0x3c,0x40,0x30,0x40,0x3c}, // w
    {0x44,0x28,0x10,0x28,0x44}, // x
    {0x0c,0x50,0x50,0x50,0x3c}, // y
    {0x44,0x64,0x54,0x4c,0x44}, // z
    {0x00,0x08,0x36,0x41,0x00}, // {
    {0x00,0x00,0x7f,0x00,0x00}, // |
    {0x00,0x41,0x36,0x08,0x00}, // }
    {0x08,0x04,0x08,0x10,0x08}, // ~
};

// ─────────────────────────────────────────────────────────────
//  Text rendering via flat quads
// ─────────────────────────────────────────────────────────────
struct TextRenderer {
    GLuint prog, vao, vbo;
    int winW, winH;

    void init(int w, int h) {
        winW = w; winH = h;
        prog = makeProgram(VS_FLAT, FS_FLAT);
        glGenVertexArrays(1, &vao);
        glGenBuffers(1, &vbo);
        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
        glEnableVertexAttribArray(0);
    }

    // Draw a single character at pixel (px,py), scale s pixels per font-pixel
    void drawChar(char c, float px, float py, float s,
                  float r, float g, float b, float a) {
        if (c < 32 || c > 126) return;
        const uint8_t* bm = FONT5x7[c - 32];
        std::vector<float> verts;
        for (int col = 0; col < 5; col++) {
            for (int row = 0; row < 7; row++) {
                if (bm[col] & (1 << row)) {
                    float x0 = px + col * s;
                    float y0 = py + row * s;
                    float x1 = x0 + s, y1 = y0 + s;
                    // convert to NDC
                    auto nx = [&](float v){ return v / winW * 2.f - 1.f; };
                    auto ny = [&](float v){ return 1.f - v / winH * 2.f; };
                    verts.insert(verts.end(), {nx(x0),ny(y0), nx(x1),ny(y0),
                                               nx(x1),ny(y1), nx(x0),ny(y1)});
                }
            }
        }
        if (verts.empty()) return;
        glUseProgram(prog);
        glUniform4f(glGetUniformLocation(prog,"color"), r,g,b,a);
        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, verts.size()*4, verts.data(), GL_DYNAMIC_DRAW);
        for (int i = 0; i < (int)verts.size()/8; i++)
            glDrawArrays(GL_TRIANGLE_FAN, i*4, 4);
    }

    void drawString(const std::string& s, float px, float py, float scale,
                    float r, float g, float b, float a=1.f) {
        float cx = px;
        for (char c : s) {
            drawChar(c, cx, py, scale, r, g, b, a);
            cx += 6.f * scale;
        }
    }
};

// ─────────────────────────────────────────────────────────────
//  Flat line/rect renderer
// ─────────────────────────────────────────────────────────────
struct FlatRenderer {
    GLuint prog, vao, vbo;
    int winW, winH;

    void init(int w, int h) {
        winW = w; winH = h;
        prog = makeProgram(VS_FLAT, FS_FLAT);
        glGenVertexArrays(1, &vao);
        glGenBuffers(1, &vbo);
        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
        glEnableVertexAttribArray(0);
    }

    void drawLines(const std::vector<float>& pts,
                   float r, float g, float b, float a=1.f, float lw=1.f) {
        glUseProgram(prog);
        glUniform4f(glGetUniformLocation(prog,"color"),r,g,b,a);
        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        // pts in pixel coords, convert to NDC
        std::vector<float> nd;
        nd.reserve(pts.size());
        for (int i = 0; i < (int)pts.size(); i+=2) {
            nd.push_back(pts[i]   / winW * 2.f - 1.f);
            nd.push_back(1.f - pts[i+1] / winH * 2.f);
        }
        glBufferData(GL_ARRAY_BUFFER, nd.size()*4, nd.data(), GL_DYNAMIC_DRAW);
        glLineWidth(lw);
        glDrawArrays(GL_LINE_STRIP, 0, (GLsizei)(nd.size()/2));
    }

    void drawRect(float x, float y, float w, float h,
                  float r, float g, float b, float a=1.f) {
        std::vector<float> pts = {
            x,y,  x+w,y,  x+w,y+h,  x,y+h, x,y
        };
        glUseProgram(prog);
        glUniform4f(glGetUniformLocation(prog,"color"),r,g,b,a);
        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        std::vector<float> nd;
        for (int i=0;i<(int)pts.size();i+=2){
            nd.push_back(pts[i]  /winW*2.f-1.f);
            nd.push_back(1.f-pts[i+1]/winH*2.f);
        }
        glBufferData(GL_ARRAY_BUFFER, nd.size()*4, nd.data(), GL_DYNAMIC_DRAW);
        glDrawArrays(GL_LINE_STRIP, 0, 5);
    }

    void fillRect(float x, float y, float w, float h,
                  float r, float g, float b, float a=0.15f) {
        float pts[] = {x,y, x+w,y, x+w,y+h, x,y+h};
        glUseProgram(prog);
        glUniform4f(glGetUniformLocation(prog,"color"),r,g,b,a);
        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        std::vector<float> nd;
        for (int i=0;i<8;i+=2){
            nd.push_back(pts[i]  /winW*2.f-1.f);
            nd.push_back(1.f-pts[i+1]/winH*2.f);
        }
        glBufferData(GL_ARRAY_BUFFER, nd.size()*4, nd.data(), GL_DYNAMIC_DRAW);
        glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
    }
};

// ─────────────────────────────────────────────────────────────
//  Particle trail
// ─────────────────────────────────────────────────────────────
struct Particle { float x,y,vx,vy,life,maxLife,size; };

// ─────────────────────────────────────────────────────────────
//  App
// ─────────────────────────────────────────────────────────────
struct App {
    GLFWwindow* win = nullptr;
    GLuint cubeProg;
    GLuint cubeVAO, cubeVBO, cubeEBO;
    TextRenderer text;
    FlatRenderer flat;

    Sim sim;
    int selectedLayer = 0;
    int screenState = 0; // 0=menu, 1=sim, 2=graph

    // particles
    std::vector<Particle> particles;

    // cube visual position (screen coords, viewport centred)
    float cubeScreenY = 0.f;   // how far cube has moved down in screen
    static const float VP_SIZE; // viewport half-size in world units

    double prevTime = 0.0;
    float  simSpeed = 1.f;

    // for graph at end
    std::vector<float> graphT, graphV, graphFd;

    void initGL() {
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        cubeProg = makeProgram(VS_CUBE, FS_CUBE);

        // build cube buffers
        glGenVertexArrays(1, &cubeVAO);
        glGenBuffers(1, &cubeVBO);
        glGenBuffers(1, &cubeEBO);
        glBindVertexArray(cubeVAO);
        glBindBuffer(GL_ARRAY_BUFFER, cubeVBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(CUBE_VERTS), CUBE_VERTS, GL_STATIC_DRAW);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, cubeEBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(CUBE_IDX), CUBE_IDX, GL_STATIC_DRAW);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6*4, (void*)0);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6*4, (void*)(3*4));
        glEnableVertexAttribArray(1);

        text.init(W, H);
        flat.init(W, H);
    }

    void spawnParticles(float px, float py) {
        static unsigned seed = 12345;
        auto rng = [&](float lo, float hi){
            seed = seed * 1664525u + 1013904223u;
            float f = (seed >> 16) / 65535.f;
            return lo + f*(hi-lo);
        };
        for (int i = 0; i < 3; i++) {
            Particle p;
            p.x = px + rng(-8.f,8.f);
            p.y = py + rng(-4.f,4.f);
            p.vx = rng(-12.f,12.f);
            p.vy = rng(-30.f,-5.f);     // upward drift (screen y up = negative)
            p.maxLife = p.life = rng(0.4f, 1.2f);
            p.size = rng(2.f, 5.f);
            particles.push_back(p);
        }
    }

    void renderMenu() {
        glClearColor(0.04f,0.04f,0.06f,1.f);
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);

        text.drawString("DRAG SIMULATION", 240, 60, 3.f, 0.85f,0.85f,0.95f);
        text.drawString("Select atmospheric layer:", 220, 130, 2.f, 0.5f,0.6f,0.7f);

        for (int i = 0; i < NUM_LAYERS; i++) {
            bool sel = (i == selectedLayer);
            float fy = 180.f + i*50.f;
            if (sel) {
                flat.fillRect(200, fy-6, 500, 36, 0.2f,0.5f,0.9f, 0.18f);
                flat.drawRect(200, fy-6, 500, 36, 0.3f,0.6f,1.f,  0.6f);
            }
            char idx[4]; snprintf(idx,4,"[%d]",i+1);
            text.drawString(idx,            210, fy, 2.f,
                            sel?0.3f:0.35f, sel?0.7f:0.5f, sel?1.f:0.65f);
            text.drawString(LAYERS[i].name, 270, fy, 2.f,
                            sel?0.9f:0.65f, sel?0.9f:0.65f, sel?0.95f:0.7f);
            // density
            char rhoStr[40];
            snprintf(rhoStr,40,"rho=%.2e kg/m3", LAYERS[i].rho);
            text.drawString(rhoStr, 600, fy, 1.5f, 0.4f,0.4f,0.5f);
        }

        text.drawString("ENTER  start    UP/DOWN  select", 210, 450, 1.5f, 0.35f,0.45f,0.55f);
    }

    void renderSim(float dt) {
        const Layer& L = LAYERS[selectedLayer];

        // sky background gradient (simple two-colour via two rects)
        glClearColor(L.r*0.5f, L.g*0.5f, L.b*0.7f, 1.f);
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);

        // ── 3-D cube viewport (left 540 px) ──────────────────
        int vpW = 540, vpH = H;
        glViewport(0, 0, vpW, vpH);
        glEnable(GL_DEPTH_TEST);

        // camera
        float aspect = (float)vpW / vpH;
        glm::mat4 proj = glm::perspective(glm::radians(45.f), aspect, 0.1f, 200.f);
        glm::mat4 view = glm::lookAt(glm::vec3(4.f,1.f,8.f),
                                     glm::vec3(0.f,0.f,0.f),
                                     glm::vec3(0.f,1.f,0.f));

        // cube world position: cube moves downward; camera follows softly
        // map sim.x (metres) → scene units (cap at ±20)
        float wy = -std::min(sim.x * 0.08f, 20.f);

        // cube model matrix
        float angle = (float)glfwGetTime() * 0.4f;
        glm::mat4 M = glm::translate(glm::mat4(1.f), glm::vec3(0.f, wy, 0.f));
        M = glm::rotate(M, angle, glm::vec3(0.3f,1.f,0.2f));
        float s = sim.side;
        M = glm::scale(M, glm::vec3(s,s,s));
        glm::mat4 MVP = proj * view * M;

        glUseProgram(cubeProg);
        glUniformMatrix4fv(glGetUniformLocation(cubeProg,"MVP"),1,GL_FALSE,glm::value_ptr(MVP));
        glUniformMatrix4fv(glGetUniformLocation(cubeProg,"M"),  1,GL_FALSE,glm::value_ptr(M));

        // colour: white-blue, glow proportional to drag
        float normV   = (sim.terminalV() > 0.001f)
                      ? std::min(sim.v / sim.terminalV(), 1.f) : 0.f;
        float glow    = normV * 0.8f;
        glUniform3f(glGetUniformLocation(cubeProg,"cubeColor"), 0.6f+normV*0.3f,
                    0.75f, 0.95f);
        glUniform1f(glGetUniformLocation(cubeProg,"glowAmount"), glow);

        glBindVertexArray(cubeVAO);
        glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_SHORT, nullptr);

        // trail line  (centre of cube screen-space path)
        // (just a subtle vertical dash set)
        glDisable(GL_DEPTH_TEST);

        // ── HUD panel (right 360 px) ──────────────────────────
        glViewport(540, 0, W-540, H);
        glDisable(GL_DEPTH_TEST);

        // re-init flat/text for sub-viewport
        // (we pass absolute pixel coords offset by 540)
        // easier: just draw in full-window coords with scissor
        glViewport(0, 0, W, H);

        // panel background
        flat.fillRect(548, 10, W-558, H-20, 0.04f,0.04f,0.08f, 0.85f);
        flat.drawRect(548, 10, W-558, H-20, 0.15f,0.25f,0.45f, 0.9f);

        // title
        text.drawString("DRAG SIM", 560, 22, 2.f, 0.3f,0.6f,1.f);

        // layer name
        text.drawString(L.name, 560, 50, 1.5f, 0.4f,0.55f,0.65f);

        // stats
        auto label = [&](const char* k, float val, const char* unit,
                         float py, float vr=1.f,float vg=1.f,float vb=1.f){
            text.drawString(k, 560, py, 1.5f, 0.45f,0.45f,0.5f);
            char buf[32]; snprintf(buf,32,"%.3f %s", val, unit);
            text.drawString(buf, 700, py, 1.5f, vr,vg,vb);
        };

        label("time",   sim.t,         "s",   115, 0.7f,0.7f,0.8f);
        label("v",      sim.v,         "m/s", 140, 0.3f,0.9f,1.0f);
        label("x",      sim.x,         "m",   165, 0.7f,0.8f,0.9f);
        label("Fd",     sim.Fd,        "N",   190, 1.0f,0.6f,0.3f);
        label("accel",  sim.accel,     "m/s2",215, 0.8f,0.9f,0.5f);
        label("v_term", sim.terminalV(),"m/s", 240, 0.5f,0.5f,0.6f);

        // progress bar toward terminal velocity
        float pct = (sim.terminalV()>0.001f)
                  ? std::min(sim.v / sim.terminalV(), 1.f) : 0.f;
        flat.fillRect(560, 268, 320*pct, 14, 0.2f,0.7f,1.f, 0.7f);
        flat.drawRect(560, 268, 320,     14, 0.2f,0.4f,0.6f,0.9f);
        text.drawString("terminal", 560, 285, 1.5f, 0.3f,0.4f,0.5f);

        // mini velocity graph in panel
        float gx = 560, gy = 330, gw = 310, gh = 160;
        flat.fillRect(gx, gy, gw, gh, 0.02f,0.04f,0.08f, 0.8f);
        flat.drawRect(gx, gy, gw, gh, 0.15f,0.25f,0.4f);
        text.drawString("velocity / time", gx, gy-16, 1.5f, 0.3f,0.4f,0.5f);

        if (sim.hCount > 1) {
            // find v range
            float vlo = 1e9f, vhi = -1e9f;
            for (int i = 0; i < sim.hCount; i++){
                int idx = (sim.hHead - sim.hCount + i + Sim::HIST) % Sim::HIST;
                vlo = std::min(vlo, sim.vHist[idx]);
                vhi = std::max(vhi, sim.vHist[idx]);
            }
            if (vhi - vlo < 0.1f) vhi = vlo + 1.f;
            std::vector<float> pts;
            for (int i = 0; i < sim.hCount; i++) {
                int idx = (sim.hHead - sim.hCount + i + Sim::HIST) % Sim::HIST;
                float tx = gx + (float)i/(sim.hCount-1)*gw;
                float ty = gy + gh - (sim.vHist[idx]-vlo)/(vhi-vlo)*gh;
                pts.push_back(tx); pts.push_back(ty);
            }
            flat.drawLines(pts, 0.3f,0.8f,1.f,1.f,1.5f);
        }

        // cube props
        text.drawString("CUBE", 560, 510, 1.5f, 0.3f,0.45f,0.6f);
        char buf[64];
        snprintf(buf,64,"mass  %.1f kg", sim.mass);
        text.drawString(buf, 560, 530, 1.5f, 0.55f,0.55f,0.6f);
        snprintf(buf,64,"side  %.2f m", sim.side);
        text.drawString(buf, 560, 550, 1.5f, 0.55f,0.55f,0.6f);
        snprintf(buf,64,"Cd    %.2f",   sim.Cd);
        text.drawString(buf, 560, 570, 1.5f, 0.55f,0.55f,0.6f);
        snprintf(buf,64,"rho   %.2e kg/m3", sim.rho);
        text.drawString(buf, 560, 590, 1.5f, 0.55f,0.55f,0.6f);

        // controls
        text.drawString("SPC pause  R reset  G graph", 560, 630, 1.5f, 0.3f,0.3f,0.35f);
        text.drawString("Q quit", 560, 648, 1.5f, 0.3f,0.3f,0.35f);
        if (sim.paused)
            text.drawString("PAUSED", 650, 82, 2.f, 1.f,0.8f,0.3f);
    }

    void renderGraph() {
        glViewport(0,0,W,H);
        glClearColor(0.03f,0.03f,0.05f,1.f);
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
        glDisable(GL_DEPTH_TEST);

        if (graphT.empty()) {
            text.drawString("No data.", 350,300,2.f,0.5f,0.5f,0.5f);
            return;
        }

        text.drawString("SIMULATION RESULTS", 250, 20, 2.5f, 0.6f,0.7f,0.9f);

        float tMax = graphT.back();
        float vMax = *std::max_element(graphV.begin(), graphV.end());
        float fdMax= *std::max_element(graphFd.begin(),graphFd.end());
        if (vMax  < 0.1f) vMax  = 1.f;
        if (fdMax < 0.1f) fdMax = 1.f;

        // ── Velocity plot ──
        {
            float gx=60,gy=70,gw=380,gh=220;
            flat.fillRect(gx,gy,gw,gh, 0.02f,0.04f,0.08f,0.9f);
            flat.drawRect(gx,gy,gw,gh, 0.15f,0.3f,0.5f);
            text.drawString("Velocity (m/s)", gx, gy-16, 1.5f, 0.4f,0.6f,0.8f);

            std::vector<float> pts;
            for (int i=0;i<(int)graphT.size();i++){
                float tx = gx + graphT[i]/tMax*gw;
                float ty = gy + gh - graphV[i]/vMax*gh;
                pts.push_back(tx); pts.push_back(ty);
            }
            flat.drawLines(pts,0.3f,0.85f,1.f,1.f,1.5f);

            // terminal line
            float vt = sim.terminalV();
            float yline = gy + gh - (vt/vMax)*gh;
            if (yline >= gy && yline <= gy+gh) {
                std::vector<float> tl = {gx,yline, gx+gw,yline};
                flat.drawLines(tl,1.f,0.6f,0.2f,0.6f,1.f);
                text.drawString("terminal", gx+gw+4, yline-4, 1.5f, 1.f,0.6f,0.2f);
            }

            // axis labels
            char buf[32];
            snprintf(buf,32,"%.1f",vMax);
            text.drawString(buf,  gx-42,gy,       1.5f,0.4f,0.5f,0.55f);
            text.drawString("0",  gx-16,gy+gh-6,  1.5f,0.4f,0.5f,0.55f);
            snprintf(buf,32,"%.1fs",tMax);
            text.drawString(buf,  gx+gw-24,gy+gh+6,1.5f,0.4f,0.5f,0.55f);
            text.drawString("0",  gx,gy+gh+6,     1.5f,0.4f,0.5f,0.55f);
        }

        // ── Drag force plot ──
        {
            float gx=460,gy=70,gw=380,gh=220;
            flat.fillRect(gx,gy,gw,gh, 0.02f,0.04f,0.08f,0.9f);
            flat.drawRect(gx,gy,gw,gh, 0.15f,0.3f,0.5f);
            text.drawString("Drag Force (N)", gx, gy-16, 1.5f, 0.9f,0.55f,0.2f);

            std::vector<float> pts;
            for (int i=0;i<(int)graphT.size();i++){
                float tx = gx + graphT[i]/tMax*gw;
                float ty = gy + gh - graphFd[i]/fdMax*gh;
                pts.push_back(tx); pts.push_back(ty);
            }
            flat.drawLines(pts,1.f,0.55f,0.2f,1.f,1.5f);

            char buf[32];
            snprintf(buf,32,"%.1f N",fdMax);
            text.drawString(buf, gx-54,gy,     1.5f,0.6f,0.4f,0.3f);
            text.drawString("0",  gx-16,gy+gh-6,1.5f,0.6f,0.4f,0.3f);
            snprintf(buf,32,"%.1fs",tMax);
            text.drawString(buf, gx+gw-24,gy+gh+6,1.5f,0.4f,0.5f,0.55f);
        }

        // ── Stats summary ──
        {
            float sx=60, sy=340;
            text.drawString("SUMMARY", sx, sy, 2.f, 0.5f,0.6f,0.75f);
            char buf[80];
            const Layer& L = LAYERS[selectedLayer];
            snprintf(buf,80,"Layer       : %s", L.name);
            text.drawString(buf, sx, sy+30, 1.5f,0.6f,0.65f,0.7f);
            snprintf(buf,80,"Terminal v  : %.3f m/s", sim.terminalV());
            text.drawString(buf, sx, sy+52, 1.5f,0.3f,0.85f,1.f);
            snprintf(buf,80,"Final v     : %.3f m/s", graphV.back());
            text.drawString(buf, sx, sy+74, 1.5f,0.6f,0.65f,0.7f);
            snprintf(buf,80,"Distance    : %.2f m",   sim.x);
            text.drawString(buf, sx, sy+96, 1.5f,0.6f,0.65f,0.7f);
            snprintf(buf,80,"Max drag    : %.3f N",   fdMax);
            text.drawString(buf, sx, sy+118,1.5f,0.6f,0.65f,0.7f);
            snprintf(buf,80,"Sim time    : %.2f s",   tMax);
            text.drawString(buf, sx, sy+140,1.5f,0.6f,0.65f,0.7f);
        }

        text.drawString("ESC  back to menu   Q  quit", 260, H-36, 1.5f, 0.3f,0.3f,0.35f);
    }

    void keyCallback(int key, int action) {
        if (action != GLFW_PRESS && action != GLFW_REPEAT) return;

        if (screenState == 0) {
            if (key == GLFW_KEY_UP)   selectedLayer = (selectedLayer+NUM_LAYERS-1)%NUM_LAYERS;
            if (key == GLFW_KEY_DOWN) selectedLayer = (selectedLayer+1)%NUM_LAYERS;
            if (key == GLFW_KEY_ENTER || key == GLFW_KEY_KP_ENTER) {
                sim.reset();
                sim.rho = LAYERS[selectedLayer].rho;
                sim.running = true;
                graphT.clear(); graphV.clear(); graphFd.clear();
                screenState = 1;
            }
        } else if (screenState == 1) {
            if (key == GLFW_KEY_SPACE) sim.paused = !sim.paused;
            if (key == GLFW_KEY_R)    { sim.reset(); sim.rho = LAYERS[selectedLayer].rho; graphT.clear(); graphV.clear(); graphFd.clear(); }
            if (key == GLFW_KEY_G)    screenState = 2;
            if (key == GLFW_KEY_ESCAPE){ sim.reset(); screenState = 0; }
        } else if (screenState == 2) {
            if (key == GLFW_KEY_ESCAPE){ screenState = 0; sim.reset(); sim.rho=LAYERS[selectedLayer].rho; graphT.clear();graphV.clear();graphFd.clear(); }
        }
        if (key == GLFW_KEY_Q) glfwSetWindowShouldClose(win, GLFW_TRUE);
    }

    void run() {
        glfwInit();
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
        win = glfwCreateWindow(W, H, "Drag Simulation", nullptr, nullptr);
        glfwMakeContextCurrent(win);
        glfwSwapInterval(1);
        glfwSetWindowUserPointer(win, this);
        glfwSetKeyCallback(win, [](GLFWwindow* w, int k, int, int a, int){
            ((App*)glfwGetWindowUserPointer(w))->keyCallback(k,a);
        });

        glewExperimental = GL_TRUE;
        glewInit();
        initGL();
        prevTime = glfwGetTime();

        static const float SIM_DT = 1.f/240.f;
        float accum = 0.f;

        while (!glfwWindowShouldClose(win)) {
            double now = glfwGetTime();
            float  dt  = (float)(now - prevTime);
            prevTime   = now;
            if (dt > 0.1f) dt = 0.1f;

            if (screenState == 1 && sim.running && !sim.paused) {
                accum += dt * simSpeed;
                while (accum >= SIM_DT) {
                    sim.step(SIM_DT);
                    accum -= SIM_DT;
                    // record every ~0.1 s
                    if (std::fmod(sim.t, 0.1f) < SIM_DT * 2.f) {
                        graphT.push_back(sim.t);
                        graphV.push_back(sim.v);
                        graphFd.push_back(std::fabs(sim.Fd));
                    }
                }
            }

            glViewport(0,0,W,H);
            if      (screenState == 0) renderMenu();
            else if (screenState == 1) renderSim(dt);
            else if (screenState == 2) renderGraph();

            glfwSwapBuffers(win);
            glfwPollEvents();
        }
        glfwDestroyWindow(win);
        glfwTerminate();
    }
};

const float App::VP_SIZE = 10.f;

// ─────────────────────────────────────────────────────────────
int main() {
    App app;
    app.run();
    return 0;
}

