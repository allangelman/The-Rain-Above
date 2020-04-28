import taichi as ti
import random
import numpy as np
import math

@ti.func
def clamp(p):
  if p < 0:
      p = 0
  if p > 1:
      p = 1
  return p

def capsule(grid_pos, a, b):
    ab = a - b
    ap = grid_pos - b
    t = ti.dot(ab, ap) / ti.dot(ab, ab)
    t_clamped = clamp(t)
    return b + t_clamped*ab

def length(a):
    x = (a[0] * a[0])
    y = (a[1] * a[1])
    z = (a[2] * a[2])
    return ti.sqrt(x + y + z)

def absolute(n):
    if n < 0:
        n = 0
    return n

def box(p, s):
    x = max(absolute(p[0]) - s[0], 0.0)
    y = max(absolute(p[1]) - s[1], 0.0)
    z = max(absolute(p[2]) - s[2], 0.0)
    return length(ti.Vector([x, y, z]))

@ti.data_oriented
class MPMSolver:
    material_water = 0
    material_elastic = 1
    material_snow = 2

    def __init__(self, res, size=1, max_num_particles=2**20):
        self.dim = len(res)
        assert self.dim in (
            2, 3), "MPM solver supports only 2D and 3D simulations."
        self.res = res
        self.n_particles = 0  #python value ... it would be 0 for taichi kernel
        self.dx = size / res[0]
        self.inv_dx = 1.0 / self.dx
        #asynch mpm to use different dt for different material  -- advaanced topic
        self.default_dt = 3e-4 * self.dx * size
        self.p_vol = self.dx**self.dim
        self.p_rho = 1
        self.p_mass = self.p_vol * self.p_rho
        self.max_num_particles = max_num_particles
        self.gravity = ti.Vector(self.dim, dt=ti.f32, shape=())
        self.source_bound = ti.Vector(self.dim, dt=ti.f32, shape=2)
        # position
        self.x = ti.Vector(self.dim, dt=ti.f32)
        # velocity
        self.v = ti.Vector(self.dim, dt=ti.f32)
        # affine velocity field
        self.C = ti.Matrix(self.dim, self.dim, dt=ti.f32)
        # deformation gradient
        self.F = ti.Matrix(self.dim, self.dim, dt=ti.f32)
        # material id
        self.material = ti.var(dt=ti.i32)
        # plastic deformation
        self.Jp = ti.var(dt=ti.f32)
        # grid node momemtum/velocity
        self.grid_v = ti.Vector(self.dim, dt=ti.f32, shape=self.res)
        # grid node mass
        self.grid_m = ti.var(dt=ti.f32, shape=self.res)

        #smaller young mod (behaves softly) or smaller dt

        # Young's modulus and Poisson's ratio
        self.E, self.nu = 1e3 * size, 0.2
        # Lame parameters
        self.mu_0, self.lambda_0 = self.E / (
            2 * (1 + self.nu)), self.E * self.nu / ((1 + self.nu) *
                                                    (1 - 2 * self.nu))

        #dynamic list which store particels
        ti.root.dynamic(ti.i, max_num_particles,
                        8192).place(self.x, self.v, self.C, self.F,
                                    self.material, self.Jp)

        if self.dim == 2:
            self.set_gravity((0, -9.8))
        else:
            self.set_gravity((0, -9.8, 0))

    def stencil_range(self):
        return ti.ndrange(*((3, ) * self.dim))

    def set_gravity(self, g):
        assert isinstance(g, tuple)
        assert len(g) == self.dim
        self.gravity[None] = g

    @ti.classkernel
    def p2g(self, dt: ti.f32):
        for p in self.x:
            base = (self.x[p] * self.inv_dx - 0.5).cast(int)
            fx = self.x[p] * self.inv_dx - base.cast(float)
            # Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
            w = [
                0.5 * ti.sqr(1.5 - fx), 0.75 - ti.sqr(fx - 1),
                0.5 * ti.sqr(fx - 0.5)
            ]
            # deformation gradient update
            self.F[p] = (ti.Matrix.identity(ti.f32, self.dim) +
                         dt * self.C[p]) @ self.F[p]
            # Hardening coefficient: snow gets harder when compressed
            h = ti.exp(10 * (1.0 - self.Jp[p]))
            if self.material[
                    p] == self.material_elastic:  # jelly, make it softer
                h = 0.3
            mu, la = self.mu_0 * h, self.lambda_0 * h
            if self.material[p] == self.material_water:  # liquid
                mu = 0.0
            U, sig, V = ti.svd(self.F[p])
            J = 1.0
            for d in ti.static(range(self.dim)):
                new_sig = sig[d, d]
                #section 7 of snow paper ... dt being too big
                if self.material[p] == self.material_snow:  # Snow
                    new_sig = min(max(sig[d, d], 1 - 2.5e-2),
                                  1 + 4.5e-3)  # Plasticity
                self.Jp[p] *= sig[d, d] / new_sig
                sig[d, d] = new_sig
                J *= new_sig
            if self.material[p] == self.material_water:
                # Reset deformation gradient to avoid numerical instability
                new_F = ti.Matrix.identity(ti.f32, self.dim)
                new_F[0, 0] = J
                self.F[p] = new_F
            elif self.material[p] == self.material_snow:
                # Reconstruct elastic deformation gradient after plasticity
                self.F[p] = U @ sig @ V.T()

            stress = 2 * mu * (self.F[p] - U @ V.T()) @ self.F[p].T(
            ) + ti.Matrix.identity(ti.f32, self.dim) * la * J * (J - 1)
            stress = (-dt * self.p_vol * 4 * self.inv_dx**2) * stress
            affine = stress + self.p_mass * self.C[p]

            # Loop over 3x3 grid node neighborhood
            for offset in ti.static(ti.grouped(self.stencil_range())):
                dpos = (offset.cast(float) - fx) * self.dx
                weight = 1.0
                for d in ti.static(range(self.dim)):
                    weight *= w[offset[d]][d]
                self.grid_v[base +
                            offset] += weight * (self.p_mass * self.v[p] +
                                                 affine @ dpos)
                self.grid_m[base + offset] += weight * self.p_mass

    @ti.classkernel
    def grid_op(self, dt: ti.f32, t: ti.f32):
        for I in ti.grouped(self.grid_m):
            if self.grid_m[I] > 0:  # No need for epsilon here
                self.grid_v[I] = (1 / self.grid_m[I]
                                  ) * self.grid_v[I]  # Momentum to velocity
                self.grid_v[I] += dt * self.gravity[None]
                for d in ti.static(range(self.dim)):
                    #if each component of position is less than 3 and the velocity is less than 0
                    if I[d] < 3 and self.grid_v[I][d] < 0:
                        self.grid_v[I][d] = 0  # Boundary conditions
                    #if each component of position is greater than the outerbound minus 3 and the velocity is greater than 0
                    if I[d] > self.res[d] - 3 and self.grid_v[I][d] > 0:
                        self.grid_v[I][d] = 0

                    grid_pos = self.dx * I
                    # if ((grid_pos - ti.Vector([0, 1 + ti.cos(t), 6 + ti.cos(t)])).norm() - 1.0**0.5 < 0):
                    #   self.grid_v[I] = [0,-ti.sin(t),-ti.sin(t)]
                    if ((grid_pos - ti.Vector([0, 1, 6])).norm() - 1.0**0.5 < 0):
                        self.grid_v[I] = [0, 0, 0]
                    
                    if ((grid_pos - (capsule(grid_pos, ti.Vector([2,3,6]), ti.Vector([4,4,6])))).norm() - 0.2 < 0):
                        self.grid_v[I] = [0, 0, 0]
                    if ((grid_pos - (capsule(grid_pos, ti.Vector([-1,5,6]), ti.Vector([1,4,6])))).norm() - 0.2 < 0):
                        self.grid_v[I] = [0, 0, 0]
                    
                    # box_position = ti.Vector([1, 4, 6])
                    # box_position_vect = (grid_pos - box_position)
                    # s = ti.Vector([1, 0.25, 0.25])
                    # x = max(abs(box_position_vect[0]) - s[0], 0.0)
                    # y = max(abs(box_position_vect[1]) - s[1], 0.0)
                    # z = max(abs(box_position_vect[2]) - s[2], 0.0)
                    # box = ti.sqrt(x*x + y*y + z*z)
                    # # distToCenter = (grid_pos - box_position).norm()
                    
                    # if ((grid_pos - box_position).norm() - box < 0):
                    #     self.grid_v[I] = [0, 0, 0]
                    
        
                    # if (grid_pos[0] < (box_position[0] + s[0]/2.0) and grid_pos[0] > (box_position[0] - s[0]/2.0)
                    #    and grid_pos[1] < (box_position[1] + s[1]/2.0) and grid_pos[1] > (box_position[1] - s[1]/2.0)
                    #    and grid_pos[2] < (box_position[2] + s[2]/2.0) and grid_pos[2] > (box_position[2] - s[2]/2.0)):
                    #     self.grid_v[I] = [0, 0, 0]
                    
                    # if (grid_posdistToCenter - box< 0):
                    #     self.grid_v[I][0] = 0
                    # if (box[1] < 0):
                    #     self.grid_v[I][1] = 0
                    # if (box[2] < 0):
                    #     self.grid_v[I][2] = 0
                    
                    # box_position2 = grid_pos - ti.Vector([1, 6, 6])
                    # box_position_vec2 = ti.Vector([box_position2[0], box_position2[1], box_position2[2]])
                    # if ((box(box_position_vec2, ti.Vector([0.01, 1, 0.25]))).norm() < 0):
                    #     self.grid_v[I] = [0, 0, 0]
                    
                    # if (grid_pos - ti.Vector([3,3,3]) < 0):
                    #   self.grid_v[I] = [0,0,0]

    @ti.classkernel
    def g2p(self, dt: ti.f32):
        for p in self.x:
            base = (self.x[p] * self.inv_dx - 0.5).cast(int)
            fx = self.x[p] * self.inv_dx - base.cast(float)
            w = [
                0.5 * ti.sqr(1.5 - fx), 0.75 - ti.sqr(fx - 1.0),
                0.5 * ti.sqr(fx - 0.5)
            ]
            new_v = ti.Vector.zero(ti.f32, self.dim)
            new_C = ti.Matrix.zero(ti.f32, self.dim, self.dim)
            # loop over 3x3 grid node neighborhood
            for I in ti.static(ti.grouped(self.stencil_range())):
                dpos = I.cast(float) - fx
                g_v = self.grid_v[base + I]
                weight = 1.0
                for d in ti.static(range(self.dim)):
                    weight *= w[I[d]][d]
                new_v += weight * g_v
                new_C += 4 * self.inv_dx * weight * ti.outer_product(g_v, dpos)
            self.v[p], self.C[p] = new_v, new_C
            self.x[p] += dt * self.v[p]  # advection

    def step(self, frame_dt, t):
        substeps = int(frame_dt / self.default_dt) + 1
        for i in range(substeps):
            dt = frame_dt / substeps
            self.grid_v.fill(0)
            self.grid_m.fill(0)
            self.p2g(dt)
            self.grid_op(dt, t + (i * dt))
            self.g2p(dt)

    @ti.classkernel
    def seed(self, num_original_particles: ti.i32, new_particles: ti.i32,
             new_material: ti.i32):
        for i in range(num_original_particles,
                       num_original_particles + new_particles):
            self.material[i] = new_material
            for k in ti.static(range(self.dim)):
                self.x[i][k] = self.source_bound[0][k] + ti.random(
                ) * self.source_bound[1][k]
            self.v[i] = ti.Vector.zero(ti.f32, self.dim)
            self.F[i] = ti.Matrix.identity(ti.f32, self.dim)
            self.Jp[i] = 1

    def add_cube(self, lower_corner, cube_size, material, sample_density=None):
        if sample_density is None:
            sample_density = 2**self.dim
        vol = 1
        for i in range(self.dim):
            vol = vol * cube_size[i]
        num_new_particles = int(sample_density * vol / self.dx**self.dim + 1)
        assert self.n_particles + num_new_particles <= self.max_num_particles

        for i in range(self.dim):
            self.source_bound[0][i] = lower_corner[i]
            self.source_bound[1][i] = cube_size[i]

        self.seed(self.n_particles, num_new_particles, material)
        self.n_particles += num_new_particles

    @ti.classkernel
    def copy_dynamic_nd(self, np_x: ti.ext_arr(), input_x: ti.template()):
        for i in self.x:
            for j in ti.static(range(self.dim)):
                np_x[i, j] = input_x[i][j]

    @ti.classkernel
    def copy_dynamic(self, np_x: ti.ext_arr(), input_x: ti.template()):
        for i in self.x:
            np_x[i] = input_x[i]

    def particle_info(self):
        np_x = np.ndarray((self.n_particles, self.dim), dtype=np.float32)
        self.copy_dynamic_nd(np_x, self.x)
        np_v = np.ndarray((self.n_particles, self.dim), dtype=np.float32)
        self.copy_dynamic_nd(np_v, self.v)
        np_material = np.ndarray((self.n_particles, ), dtype=np.int32)
        self.copy_dynamic(np_material, self.material)
        return np_x, np_v, np_material
