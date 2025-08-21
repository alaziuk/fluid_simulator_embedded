#pragma once
#include <etl/array.h>
#include <cmath>
#include <cstdint>

namespace fluid2d {

  struct Vec2 {
    float x, y;
  };

  struct Config {
    float    width;
    float    height;
    float    density;
    float    particleRadius;
    float    flipRatio;
    uint32_t pressureIterations;
    uint32_t separateIterations;
    bool     compensateDrift;
  };

  struct Particle {
    Vec2 position;
    Vec2 velocity;
  };

  enum CellType { FLUID = 0, AIR = 1, SOLID = 2 };

  template<size_t MAX_PARTICLES, size_t NX, size_t NY>
  class Flip2D {
  public:
    void init(const Config &cfg) {
      this->numCells = NX * NY;
      this->width    = cfg.width;
      this->height   = cfg.height;

      // anisotropic grid spacing
      this->hx    = this->width / float(NX);
      this->hy    = this->height / float(NY);
      this->invhx = 1.f / this->hx;
      this->invhy = 1.f / this->hy;

      this->density            = cfg.density;
      this->flipRatio          = cfg.flipRatio;
      this->pressureIterations = cfg.pressureIterations;
      this->separateIterations = cfg.separateIterations;
      this->compensateDrift    = cfg.compensateDrift;
      this->particleRadius     = cfg.particleRadius;

      this->numParticles = 0;
      this->gravity      = {0.f, -9.81f};

      this->u.fill(0.f);
      this->v.fill(0.f);
      this->uPrev.fill(0.f);
      this->vPrev.fill(0.f);
      this->du.fill(0.f);
      this->dv.fill(0.f);
      this->s.fill(1.0f);
      this->p.fill(0.f);
      this->cellType.fill(AIR);
      this->cellCount.fill(0);

      // borders = SOLID
      for (uint32_t i = 0; i < NX; ++i) {
        this->s[idx(i, 0)]      = 0.0f;
        this->s[idx(i, NY - 1)] = 0.0f;
      }
      for (uint32_t j = 0; j < NY; ++j) {
        this->s[idx(0, j)]      = 0.0f;
        this->s[idx(NX - 1, j)] = 0.0f;
      }

      // ---- Circle mask (in cell units) ----
      // Center at (NX/2, NY/2); radius leaves a 1-cell margin.
      cx_cell     = NX * 0.5f;
      cy_cell     = NY * 0.5f;
      radius_cell = std::min(NX, NY) * 0.5f - 1.0f;

      for (uint32_t i = 0; i < NX; ++i) {
        for (uint32_t j = 0; j < NY; ++j) {
          // distance from cell center to circle center (in cell units)
          float dx   = (float(i) + 0.5f) - cx_cell;
          float dy   = (float(j) + 0.5f) - cy_cell;
          float dist = std::sqrt(dx * dx + dy * dy);
          if (dist > radius_cell) {
            this->s[idx(i, j)] = 0.0f; // SOLID outside the circle
          }
        }
      }
    }

    inline void set_gravity(Vec2 g) {
      this->gravity = g;
    }

    inline void add_particle(Vec2 p, Vec2 v) {
      if (this->numParticles >= MAX_PARTICLES)
        return;
      // If requested spawn is SOLID, push inside the circle
      if (is_solid_at(p.x, p.y)) {
        Particle tmp{p, v};
        push_inside_circle(tmp);
        p = tmp.position;
        v = tmp.velocity;
      }
      this->particles[this->numParticles++] = {p, v};
    }

    void step(float dt) {
      this->integrate_particles(dt);
      if (this->separateIterations)
        this->separate_particles(this->separateIterations);
      this->collide_walls();
      this->particles_to_grid();
      this->pressure_solve(dt);
      this->grid_to_particles();
      this->update_cell_counts();
    }

    // Accessors
    inline const Particle *get_particles() const {
      return this->particles.data();
    }

    inline uint32_t get_num_particles() const {
      return this->numParticles;
    }

    inline uint32_t get_nx() const {
      return NX;
    }

    inline uint32_t get_ny() const {
      return NY;
    }

    inline float get_cell_hx() const {
      return this->hx;
    }

    inline float get_cell_hy() const {
      return this->hy;
    }

    inline float get_width() const {
      return this->width;
    }

    inline float get_height() const {
      return this->height;
    }

    inline bool is_solid(uint32_t i, uint32_t j) const {
      return this->s[idx(i, j)] == 0.f;
    }

    // Counts for LED / grid
    inline const uint16_t *get_cell_counts() const {
      return this->cellCount.data();
    }

    inline uint16_t get_count(uint32_t i, uint32_t j) const {
      return this->cellCount[idx(i, j)];
    }

  private:
    inline uint32_t idx(uint32_t i, uint32_t j) const {
      return i * NY + j;
    }

    // sample SOLID using world coords
    inline bool is_solid_at(float x, float y) const {
      int i = (int) std::floor(x * this->invhx);
      int j = (int) std::floor(y * this->invhy);
      if (i < 0)
        i = 0;
      if (i >= (int) NX)
        i = (int) NX - 1;
      if (j < 0)
        j = 0;
      if (j >= (int) NY)
        j = (int) NY - 1;
      return this->s[idx((uint32_t) i, (uint32_t) j)] == 0.f;
    }

    // project particle from SOLID to just inside the circle and zero velocity
    inline void push_inside_circle(Particle &p) {
      // position in cell units
      float x_c = p.position.x * this->invhx;
      float y_c = p.position.y * this->invhy;

      float gx = x_c - cx_cell;
      float gy = y_c - cy_cell;
      float d  = std::sqrt(gx * gx + gy * gy);
      if (d < 1e-6f) {
        gx = 1.f;
        gy = 0.f;
        d  = 1.f;
      }

      // half-cell inset to avoid jitter with mask boundary
      const float r_in = radius_cell - 0.5f;

      gx /= d;
      gy /= d;
      x_c = cx_cell + gx * r_in;
      y_c = cy_cell + gy * r_in;

      p.position.x = x_c * this->hx;
      p.position.y = y_c * this->hy;
      p.velocity.x = 0.f;
      p.velocity.y = 0.f;
    }

    void integrate_particles(float dt) {
      for (uint32_t i = 0; i < this->numParticles; ++i) {
        auto &P = this->particles[i];
        P.velocity.x += this->gravity.x * dt;
        P.velocity.y += this->gravity.y * dt;
        P.position.x += P.velocity.x * dt;
        P.position.y += P.velocity.y * dt;

        // domain AABB clamp (safety)
        const float minX = this->hx + this->particleRadius;
        const float maxX = this->width - this->hx - this->particleRadius;
        const float minY = this->hy + this->particleRadius;
        const float maxY = this->height - this->hy - this->particleRadius;
        if (P.position.x < minX) {
          P.position.x = minX;
          P.velocity.x = 0.f;
        }
        if (P.position.x > maxX) {
          P.position.x = maxX;
          P.velocity.x = 0.f;
        }
        if (P.position.y < minY) {
          P.position.y = minY;
          P.velocity.y = 0.f;
        }
        if (P.position.y > maxY) {
          P.position.y = maxY;
          P.velocity.y = 0.f;
        }

        // if ended up in SOLID (outside circle), push back in
        if (is_solid_at(P.position.x, P.position.y)) {
          push_inside_circle(P);
        }
      }
    }

    void collide_walls() {
      // (kept as extra safety against AABB bounds)
      const float minX = this->hx + this->particleRadius;
      const float maxX = this->width - this->hx - this->particleRadius;
      const float minY = this->hy + this->particleRadius;
      const float maxY = this->height - this->hy - this->particleRadius;
      for (uint32_t i = 0; i < this->numParticles; ++i) {
        auto &p = this->particles[i];
        if (p.position.x < minX) {
          p.position.x = minX;
          p.velocity.x = 0.f;
        }
        if (p.position.x > maxX) {
          p.position.x = maxX;
          p.velocity.x = 0.f;
        }
        if (p.position.y < minY) {
          p.position.y = minY;
          p.velocity.y = 0.f;
        }
        if (p.position.y > maxY) {
          p.position.y = maxY;
          p.velocity.y = 0.f;
        }
      }
    }

    void separate_particles(uint32_t iters) {
      const float minDist  = 2.f * this->particleRadius;
      const float minDist2 = minDist * minDist;
      for (uint32_t it = 0; it < iters; ++it) {
        for (uint32_t i = 0; i < this->numParticles; ++i) {
          for (uint32_t j = i + 1; j < this->numParticles; ++j) {
            float dx = this->particles[j].position.x - this->particles[i].position.x;
            float dy = this->particles[j].position.y - this->particles[i].position.y;
            float d2 = dx * dx + dy * dy;
            if (d2 > 0.f && d2 < minDist2) {
              float d = std::sqrt(d2);
              float s = 0.5f * (minDist - d) / d;
              dx *= s;
              dy *= s;
              this->particles[i].position.x -= dx;
              this->particles[i].position.y -= dy;
              this->particles[j].position.x += dx;
              this->particles[j].position.y += dy;
            }
          }
        }
      }
    }

    void particles_to_grid() {
      this->uPrev = this->u;
      this->vPrev = this->v;
      this->u.fill(0.f);
      this->v.fill(0.f);
      this->du.fill(0.f);
      this->dv.fill(0.f);

      for (uint32_t i = 0; i < NX; ++i)
        for (uint32_t j             = 0; j < NY; ++j)
          this->cellType[idx(i, j)] = (this->s[idx(i, j)] == 0.f) ? SOLID : AIR;

      // mark fluid cells but never override SOLID
      for (uint32_t k = 0; k < this->numParticles; ++k) {
        uint32_t i  = std::min<uint32_t>(std::floor(this->particles[k].position.x * this->invhx), NX - 1);
        uint32_t j  = std::min<uint32_t>(std::floor(this->particles[k].position.y * this->invhy), NY - 1);
        auto     id = idx(i, j);
        if (this->cellType[id] != SOLID)
          this->cellType[id] = FLUID;
      }

      // accumulate velocities (PIC)
      for (uint32_t k = 0; k < this->numParticles; ++k) {
        uint32_t i = std::min<uint32_t>(std::floor(this->particles[k].position.x * this->invhx), NX - 1);
        uint32_t j = std::min<uint32_t>(std::floor(this->particles[k].position.y * this->invhy), NY - 1);
        auto     n = idx(i, j);
        this->u[n] += this->particles[k].velocity.x;
        this->du[n] += 1.f;
        this->v[n] += this->particles[k].velocity.y;
        this->dv[n] += 1.f;
      }
      for (uint32_t n = 0; n < this->numCells; ++n) {
        if (this->du[n] > 0.f)
          this->u[n] /= this->du[n];
        if (this->dv[n] > 0.f)
          this->v[n] /= this->dv[n];
      }
    }

    void pressure_solve(float dt) {
      this->p.fill(0.f);
      const float h_eff = (this->hx < this->hy) ? this->hx : this->hy;
      const float cp    = this->density * h_eff / dt;

      for (uint32_t iter = 0; iter < this->pressureIterations; ++iter) {
        for (uint32_t i = 1; i < NX - 1; ++i) {
          for (uint32_t j = 1; j < NY - 1; ++j) {
            if (this->cellType[idx(i, j)] != FLUID)
              continue;

            uint32_t c = idx(i, j);
            uint32_t L = idx(i - 1, j), R = idx(i + 1, j);
            uint32_t B = idx(i, j - 1), T = idx(i, j + 1);

            float sx0  = this->s[L], sx1 = this->s[R];
            float sy0  = this->s[B], sy1 = this->s[T];
            float sumS = sx0 + sx1 + sy0 + sy1;
            if (sumS == 0.f)
              continue;

            float div  = (this->u[R] - this->u[c]) + (this->v[T] - this->v[c]);
            float pval = -div / sumS * 1.9f;
            this->p[c] += cp * pval;

            this->u[c] -= sx0 * pval;
            this->u[R] += sx1 * pval;
            this->v[c] -= sy0 * pval;
            this->v[T] += sy1 * pval;
          }
        }
      }
    }

    void grid_to_particles() {
      for (uint32_t k = 0; k < this->numParticles; ++k) {
        uint32_t i = std::min<uint32_t>(std::floor(this->particles[k].position.x * this->invhx), NX - 1);
        uint32_t j = std::min<uint32_t>(std::floor(this->particles[k].position.y * this->invhy), NY - 1);
        auto     n = idx(i, j);
        this->particles[k].velocity.x = this->u[n];
        this->particles[k].velocity.y = this->v[n];
      }
    }

    void update_cell_counts() {
      this->cellCount.fill(0);
      for (uint32_t k = 0; k < this->numParticles; ++k) {
        uint32_t i = std::min<uint32_t>(std::floor(this->particles[k].position.x * this->invhx), NX - 1);
        uint32_t j = std::min<uint32_t>(std::floor(this->particles[k].position.y * this->invhy), NY - 1);
        auto     n = idx(i, j);
        if (this->cellCount[n] != 0xFFFF)
          this->cellCount[n] += 1;
      }
    }

  private:
    uint32_t numCells           = 0;
    float    width              = 0.f,    height             = 0.f;
    float    hx                 = 0.f,    hy                 = 0.f,  invhx          = 0.f, invhy = 0.f;
    float    density            = 1000.f, flipRatio          = 0.9f, particleRadius = 0.01f;
    uint32_t pressureIterations = 40,     separateIterations = 2;
    bool     compensateDrift    = false;
    Vec2     gravity{0.f, -9.81f};

    // circle mask parameters (cell space)
    float cx_cell = 0.f, cy_cell = 0.f, radius_cell = 0.f;

    uint32_t                            numParticles = 0;
    etl::array<Particle, MAX_PARTICLES> particles;

    etl::array<float, NX * NY>    u, v, uPrev, vPrev, du, dv, s, p;
    etl::array<CellType, NX * NY> cellType;

    etl::array<uint16_t, NX * NY> cellCount;
  };

} // namespace fluid2d
