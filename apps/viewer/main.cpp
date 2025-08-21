#include <SDL2/SDL.h>
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <fluid2d/fluid2d.hpp>

constexpr int MAX_PARTICLES = 300;
constexpr int NX            = 30;
constexpr int NY            = 30;

constexpr float G0           = 9.81f;
constexpr float PIXELS_PER_G = 100.f;
constexpr float MAX_G        = 100.f;

using Sim = fluid2d::Flip2D<MAX_PARTICLES, NX, NY>;

static void drawArrow(SDL_Renderer *ren, int x0, int y0, float vx, float vy) {
  int x1 = x0 + int(vx), y1 = y0 + int(vy);
  SDL_RenderDrawLine(ren, x0, y0, x1, y1);
  float len = std::sqrt(vx * vx + vy * vy);
  if (len > 1.f) {
    float ux   = vx / len, uy = vy / len, hx = -uy, hy = ux;
    int   size = 10;
    SDL_RenderDrawLine(ren, x1, y1, x1 - int(ux * 15 + hx * size), y1 - int(uy * 15 + hy * size));
    SDL_RenderDrawLine(ren, x1, y1, x1 - int(ux * 15 - hx * size), y1 - int(uy * 15 - hy * size));
  }
}

int main(int, char **) {
  const int W = 960, H = 720;
  if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER) != 0) {
    std::fprintf(stderr, "SDL init failed: %s\n", SDL_GetError());
    return 1;
  }
  SDL_Window *win = SDL_CreateWindow("Fluid2D (grid + arrow gravity)",
                                     SDL_WINDOWPOS_CENTERED,
                                     SDL_WINDOWPOS_CENTERED,
                                     W,
                                     H,
                                     SDL_WINDOW_SHOWN);
  SDL_Renderer *ren = SDL_CreateRenderer(win, -1, SDL_RENDERER_ACCELERATED);
  if (!ren) {
    std::fprintf(stderr, "Renderer failed\n");
    return 1;
  }

  // ---- Create sim ----
  Sim             sim;
  fluid2d::Config cfg{1.0f, 0.75f, 1000.f, 0.010f, 0.9f, 25, 2, true};
  sim.init(cfg);

  // Seed some particles
  const float sx    = cfg.width / NX, sy = cfg.height / NY;
  int         added = 0;
  for (int i = 0; i < static_cast<int>(sim.get_nx()) && added < MAX_PARTICLES; ++i) {
    for (int j = 0; j < static_cast<int>(sim.get_ny()) && added < MAX_PARTICLES; ++j) {
      if ((i < NX / 2) && (j < NY - 2)) {
        float x = (i + 0.5f) * sx * 0.95f;
        float y = (j + 0.5f) * sy * 0.95f;
        sim.add_particle({x, y}, {0, 0});
        ++added;
      }
    }
  }

  // Gravity via arrow
  float g_mag         = G0,    dir_x         = 0.f,  dir_y  = -1.f;
  bool  dragging      = false, showGridLines = true, paused = false;
  auto  apply_gravity = [&]() {
    sim.set_gravity({g_mag * dir_x, g_mag * dir_y});
  };
  apply_gravity();

  const float sim_dt  = 1.f / 120.f;
  bool        running = true;
  while (running) {
    SDL_Event e;
    while (SDL_PollEvent(&e)) {
      if (e.type == SDL_QUIT)
        running = false;
      else if (e.type == SDL_KEYDOWN) {
        switch (e.key.keysym.sym) {
          case SDLK_ESCAPE:
            running = false;
            break;
          case SDLK_SPACE:
            paused = !paused;
            break;
          case SDLK_g:
            showGridLines = !showGridLines;
            break;
          case SDLK_r:
            g_mag = G0;
            dir_x = 0.f;
            dir_y = -1.f;
            apply_gravity();
            break;
          case SDLK_LEFT:
            dir_x -= 0.05f;
            break;
          case SDLK_RIGHT:
            dir_x += 0.05f;
            break;
          case SDLK_UP:
            dir_y += 0.05f;
            break;
          case SDLK_DOWN:
            dir_y -= 0.05f;
            break;
          case SDLK_LEFTBRACKET:
            g_mag = std::max(0.f, g_mag - 0.5f);
            apply_gravity();
            break;
          case SDLK_RIGHTBRACKET:
            g_mag = std::min(MAX_G, g_mag + 0.5f);
            apply_gravity();
            break;
        }
        float len = std::sqrt(dir_x * dir_x + dir_y * dir_y);
        if (len > 1e-5f) {
          dir_x /= len;
          dir_y /= len;
          apply_gravity();
        }
      } else if (e.type == SDL_MOUSEBUTTONDOWN && e.button.button == SDL_BUTTON_LEFT) {
        dragging = true;
      } else if (e.type == SDL_MOUSEBUTTONUP && e.button.button == SDL_BUTTON_LEFT) {
        dragging = false;
      } else if (e.type == SDL_MOUSEWHEEL) {
        if (!dragging) {
          if (e.wheel.y > 0)
            g_mag = std::min(MAX_G, g_mag + 0.5f);
          if (e.wheel.y < 0)
            g_mag = std::max(0.f, g_mag - 0.5f);
          apply_gravity();
        }
      }
    }

    if (dragging) {
      int mx, my;
      SDL_GetMouseState(&mx, &my);
      float vx  = float(mx - W / 2), vy = float(my - H / 2);
      float len = std::sqrt(vx * vx + vy * vy);
      if (len > 1e-5f) {
        dir_x = vx / len;
        dir_y = -vy / len;
        g_mag = std::min(MAX_G, (len / PIXELS_PER_G) * G0);
        apply_gravity();
      }
    }

    if (!paused)
      sim.step(sim_dt);

    // ---- Render ----
    SDL_SetRenderDrawColor(ren, 255, 255, 255, 255); // white background
    SDL_RenderClear(ren);

    const uint16_t *counts = sim.get_cell_counts();
    const int       NXv    = sim.get_nx(), NYv = sim.get_ny();
    const float     cellW  = float(W) / NXv;
    const float     cellH  = float(H) / NYv;

    // 1) SOLID cells: black fill
    for (int i = 0; i < NXv; ++i) {
      for (int j = 0; j < NYv; ++j) {
        if (sim.is_solid(i, j)) {
          SDL_SetRenderDrawColor(ren, 0, 0, 0, 255);
          SDL_Rect rc{int(i * cellW), int(H - (j + 1) * cellH),
                      int(std::ceil(cellW)), int(std::ceil(cellH))};
          SDL_RenderFillRect(ren, &rc);
        }
      }
    }

    // 2) Fluid occupancy: blue fill where count>0 and not solid
    for (int i = 0; i < NXv; ++i) {
      for (int j = 0; j < NYv; ++j) {
        if (!sim.is_solid(i, j) && counts[i * NYv + j] > 0) {
          SDL_SetRenderDrawColor(ren, 0, 120, 255, 255);
          SDL_Rect rc{int(i * cellW), int(H - (j + 1) * cellH),
                      int(std::ceil(cellW)), int(std::ceil(cellH))};
          SDL_RenderFillRect(ren, &rc);
        }
      }
    }

    // 3) Grid lines: black
    SDL_SetRenderDrawColor(ren, 0, 0, 0, 255);
    for (int i = 0; i <= NXv; ++i)
      SDL_RenderDrawLine(ren, int(i * cellW), 0, int(i * cellW), H);
    for (int j = 0; j <= NYv; ++j)
      SDL_RenderDrawLine(ren, 0, int(j * cellH), W, int(j * cellH));

    // 4) Gravity arrow (black)
    SDL_SetRenderDrawColor(ren, 0, 0, 0, 255);
    float scale = PIXELS_PER_G * (g_mag / G0);
    drawArrow(ren, W / 2, H / 2, dir_x * scale, -dir_y * scale);

    SDL_RenderPresent(ren);
    SDL_Delay(1000 / 60);
  }

  SDL_DestroyRenderer(ren);
  SDL_DestroyWindow(win);
  SDL_Quit();
  return 0;
}