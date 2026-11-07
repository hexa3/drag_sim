# Drag Simulation (OpenGL)

A compact, real‑time physics visualization of a falling cube with aerodynamic drag. This single‑file C++ demo uses OpenGL to render a rotating cube and a small HUD that shows live simulation statistics and graphs. The simulation models the quadratic drag force

$$
F_d = \tfrac{1}{2}\,\rho\,C_d\,A\,v^2
$$

and the net downward acceleration

$$
a = g - \frac{F_d}{m}.
$$

The program numerically integrates motion using a fixed time step and records data for a final graph view.

---

## Table of contents

- **Overview**
- **Features**
- **Quick start**
- **Controls**
- **Physics & numerical integration**
- **Code structure**
- **Platform build instructions**
- **Customization**
- **Troubleshooting**
- **Contributing & license**

---

## Overview

This project is a simple, visually-oriented simulator that demonstrates how quadratic drag affects a falling object. It is intended as an educational demo and a compact example of combining a physical simulation with minimal OpenGL rendering (shaders, simple HUD, and a pixel font renderer).

- Single source file: `Drag.cpp` (self-contained demo)
- Uses GLFW, GLEW and GLM for windowing, OpenGL extensions, and math
- Fixed-step integration for stable behavior

## Features

- Real‑time 3D cube rendering with a phong‑like shading and glow
- HUD with live stats (time, position, velocity, drag force, acceleration)
- Mini graph of recent velocity history and a full results graph view
- Multiple atmospheric layers (density presets) selectable from the menu
- Pause / reset / graph export (in-memory) and keyboard-driven UI
- Small particle trail and simple 5×7 bitmap font for text rendering

## Quick start

1. Install dependencies (see platform section below).
2. Compile `Drag.cpp` as shown in the platform-specific instructions.
3. Run the produced binary and use the keyboard controls described below.

Example (Linux):

```bash
g++ -O2 -std=c++17 Drag.cpp -lGL -lGLEW -lglfw -o drag_sim
./drag_sim
```

> Note: the project header contains a sample compile command; adapt it for your OS and package manager (see below).

## Controls

- Menu: Up / Down to choose layer, Enter to start
- In simulation:
  - Space: pause / resume
  - R: reset simulation
  - G: switch to results graph view
  - ESC: back to menu
  - Q: quit

## Physics & numerical integration

The simulation uses a simple explicit Euler update with a fixed physics timestep (SIM_DT = 1/240 s) inside a real-time loop. On each physics step the code computes:

1. Cross-sectional area: $A = s^2$ (for a cube of side length $s$)
2. Drag: $F_d = \tfrac{1}{2}\,\rho\,C_d\,A\,v\,|v|$ (sign handled by $v\,|v|$)
3. Acceleration: $a = g - F_d/m$
4. Update velocity and position with Euler steps:

$$
v_{t+\Delta t} = v_t + a\,\Delta t,\qquad x_{t+\Delta t} = x_t + v_{t+\Delta t}\,\Delta t
$$

The code also computes the theoretical terminal velocity by solving $a=0$ for $v$:

$$
v_{t} = \sqrt{\frac{2m g}{\rho C_d A}}.
$$

Because the integrator is explicit and uses a small fixed step, the simulation is stable for the demo parameters. If you change mass/area drastically, consider reducing `SIM_DT` or using a higher-order integrator.

## Code structure (high level)

- `Sim` — holds the physics state and parameters: `mass`, `side`, `Cd`, `rho`, `v`, `x`, `t`, plus history buffers for plotting. Key methods: `step(dt)`, `reset()`, and `terminalV()`.
- `TextRenderer` — draws text using a compact 5×7 bitmap font (FONT5x7). Text is rendered as quads so the demo avoids external font libs.
- `FlatRenderer` — 2D line/rect rendering used for HUD, graphs and panels.
- `App` — main application object that sets up OpenGL, loads the cube geometry, compiles shaders, handles input, runs the main loop and switches between `menu`, `sim`, and `graph` screens.

The file is intentionally compact and organized so you can read how rendering, UI, and simulation interact in one place.

## Dependencies & downloads

Before building you will need the following libraries and tools. In most cases you can install them from your platform's package manager (see the Platform build instructions), but direct downloads and links are provided if you prefer to install manually.

- OpenGL driver: a GPU driver that supports at least OpenGL 3.3 core profile. On Windows install the latest drivers from your GPU vendor (NVIDIA / AMD / Intel). On Linux ensure Mesa or vendor drivers are up to date.
- GLFW (windowing & input): https://www.glfw.org/ — download page: https://www.glfw.org/download.html
- GLEW (OpenGL extension loader): http://glew.sourceforge.net/ — GitHub releases: https://github.com/nigels-com/glew/releases
- GLM (math library, header-only): https://github.com/g-truc/glm — releases: https://github.com/g-truc/glm/releases
- C++ compiler with C++17 support (GCC, Clang, or MSVC)
- Build tools: `cmake` (recommended) or your platform's native toolchain (make, MSBuild)

Optional / helper tools:
- `vcpkg` (Windows / cross-platform) — https://github.com/microsoft/vcpkg
- `brew` (macOS) — https://brew.sh
- MSYS2 / pacman (Windows, MinGW-w64) — https://www.msys2.org

Notes:
- If you use a package manager (apt, pacman, brew, vcpkg), prefer installing the packaged versions — they provide headers and libs in the correct locations and avoid manual linking headaches.
- The quick commands in the Platform build instructions install the development packages for you; use those first if you are unsure.

## Platform build instructions

Below are practical examples for popular platforms. The exact package names vary by distribution or package manager.

### Linux (Debian / Ubuntu)

```bash
sudo apt update
sudo apt install build-essential libgl1-mesa-dev libglu1-mesa-dev libglew-dev libglfw3-dev libglm-dev
g++ -O2 -std=c++17 Drag.cpp -lGL -lGLEW -lglfw -o drag_sim
./drag_sim
```

If you see link errors about `GL` or `glfw`, ensure the `-lGL` and `-lglfw` flags are present and that the dev packages are installed.

### macOS (Homebrew)

```bash
brew install glew glfw glm
g++ -O2 -std=c++17 Drag.cpp -lGLEW -lglfw -framework OpenGL -o drag_sim
./drag_sim
```

Framework linking for OpenGL (`-framework OpenGL`) is required on macOS.

### Windows (MSYS2 / MinGW-w64)

Open the **MSYS2 MinGW 64-bit** shell and install packages:

```bash
# (run these in the MSYS2 MinGW64 shell)
pacman -Syu
pacman -S mingw-w64-x86_64-toolchain mingw-w64-x86_64-glew mingw-w64-x86_64-glfw mingw-w64-x86_64-glm
g++ -O2 -std=c++17 Drag.cpp -lglew32 -lglfw3 -lopengl32 -lgdi32 -o drag_sim.exe
./drag_sim.exe
```

If you use `vcpkg` or Visual Studio, prefer importing libraries with CMake and then build using the IDE.

### CMake (example)

Below is a very small CMake skeleton to integrate into a CMake-based workflow. You will still need to install or make available the GLFW/GLEW/GLM packages.

```cmake
cmake_minimum_required(VERSION 3.10)
project(DragSim)
add_executable(drag_sim Drag.cpp)
# Adjust include/link paths to your platform or use find_package with appropriate modules
target_link_libraries(drag_sim PRIVATE GLEW::GLEW glfw OpenGL::GL)
```

## Customization & ideas for extension

- Change default physical parameters in the `Sim` struct (`mass`, `side`, `Cd`, `rho`) to experiment with different shapes and materials.
- Make density a function of altitude for a realistic atmosphere (exponential decay with altitude).
- Add wind (horizontal velocity) or lift models for non-symmetric objects.
- Replace bitmap text with FreeType for higher-quality fonts and localization.
- Add replay export or CSV logging of the `graphT`, `graphV`, and `graphFd` arrays.

## Troubleshooting

- Black window / no geometry: ensure GLEW is initialized (`glewInit()` is called) and your GPU drivers support at least OpenGL 3.3 core profile.
- Shader compile/link errors: your driver or linking flags might not match the required OpenGL profile.
- Linker errors on Windows: double-check the libraries (`-lglew32 -lglfw3 -lopengl32`) and that you build in the MinGW64 environment when using MinGW packages.
- Running slowly: lower the viewport resolution in code or reduce rendering quality; the physics is cheap relative to rendering.

## Where to look in the code

- Simulation parameters and logic: `Sim` struct near the top of the file.
- Input handling: `App::keyCallback()`.
- Rendering code and shaders: `VS_CUBE`, `FS_CUBE`, `VS_FLAT`, `FS_FLAT` and the `App::renderSim()` / `App::renderGraph()` methods.

## Credits & acknowledgements

- Inspired by kavan's “Simulating Atoms in C++” approach to minimal physics visualizations (see header in `Drag.cpp`).
- Uses the open-source libraries GLFW, GLEW and GLM.

## License

[MIT](LICENSE)

Part of a robotics & simulation series with [torque](https://github.com/hexa3/torque), [helios](https://github.com/hexa3/helios), and [block_craftz](https://github.com/hexa3/block_craftz).

---

<p align="center">Built by <a href="https://sheheryar.xyz">Sheheryar</a> · <a href="https://github.com/hexa3">github.com/hexa3</a></p>

<p align="center"><sub><i>Disclaimer: this README was generated by AI. I'm honestly terrible at describing my own projects (my first draft is usually just "it does the thing, trust me"), so I let a robot do the explaining.</i></sub></p>
