# fractal.py

import taichi as ti
import numpy as np
import math
from mpm_solver import MPMSolver
from renderer_utils import out_dir, ray_aabb_intersection, inf, eps, \
  intersect_sphere, sphere_aabb_intersect_motion, inside_taichi

ti.require_version(0, 5, 10)
ti.init(arch=ti.x64)

n = 320
m = 20
hit_sphere = 0
pixels = ti.Vector(4, dt=ti.f32, shape=(n * 2, n))
support = 2
shutter_time = 0.5e-3
sphere_radius = 0.03
MAX_STEPS = 100
MAX_DIST = 100.0
SURF_DIST = 0.01
max_num_particles_per_cell = 8192 * 1024
voxel_has_particle = ti.var(dt=ti.i32)
sphere_color = ti.Vector([0.9, 0.8, 0.3])
plane_color = ti.Vector([0.9, 0.4, 0.3])
particle_color = ti.Vector([0.1, 0.4, 0.8])
backgound_color = ti.Vector([0.9, 0.4, 0.6])

pid = ti.var(ti.i32)
num_particles = ti.var(ti.i32, shape=())
bbox = ti.Vector(3, dt=ti.f32, shape=2)
particle_grid_res = 8
inv_dx = 8.0
dx = 1.0 / inv_dx
particle_x = ti.Vector(3, dt=ti.f32)
particle_v = ti.Vector(3, dt=ti.f32)
# grid_density = ti.var(dt=ti.i32)
fin = 0.0
grid_visualization_block_size = 2
max_num_particles = 1024 * 1024 * 4
grid_resolution = 8 // grid_visualization_block_size
# hit_sphere = 0


@ti.layout
def buffers():
    ti.root.dense(ti.ijk, 2).dense(ti.ijk, particle_grid_res // 8).dense(
        ti.ijk, 8).place(voxel_has_particle)
    ti.root.dense(ti.ijk, 4).pointer(ti.ijk, particle_grid_res // 8).dense(
        ti.ijk, 8).dynamic(ti.l, max_num_particles_per_cell, 512).place(pid)

    ti.root.dense(ti.l, max_num_particles).place(particle_x, particle_v)
    # ti.root.dense(ti.ijk, grid_resolution // 8).dense(ti.ijk,
    #                                                   8).place(grid_density)


mpm = MPMSolver(res=(64, 64, 64), size=10)
mpm.add_cube(lower_corner=[0, 3, 6],
             cube_size=[3, 1, 0.5],
             material=MPMSolver.material_water)
mpm.set_gravity((0, -50, 0))
np_x, np_v, np_material = mpm.particle_info()
s_x = np.size(np_x, 0)
s_y = np.size(np_x, 1)
num_part = mpm.n_particles
num_particles[None] = num_part
print('num_input_particles =', num_part)


@ti.func
def complex_sqr(z):
    return ti.Vector([z[0] * z[0] - z[1] * z[1], z[1] * z[0] * 2])


@ti.func
def length(a):
    x = (a[0] * a[0])
    y = (a[1] * a[1])
    z = (a[2] * a[2])
    return ti.sqrt(x + y + z)


@ti.func
def DistLine(ro, rd, p):
    c = ti.cross(p - ro, rd)
    l_c = length(c)
    l_rd = length(rd)
    return l_c / l_rd


@ti.func
def xyz(a):
    return ti.Vector([a[0], a[1], a[2]])

@ti.func
def sdf_Capsule(p, a, b, r):
  ab = b - a
  ap = p - a

  t = ti.dot(ab, ap) / ti.dot(ab, ab)
  t_clamped = clamp(t)
  c = a + t_clamped*ab
  return length(p-c) - r


@ti.func
def GetDist(p, t):
    intersection_object = 0
    s = ti.Vector([0, 1.0, 6.0, 1.0**0.5])
    dist = p - xyz(s)
    sphereDist = length(dist) - s[3]
    planeDist = p[1]
    capsuleDist = sdf_Capsule(p, ti.Vector([2,1,6]), ti.Vector([4,2,6]), 0.2)
    d = min(planeDist, sphereDist, capsuleDist)
    if d == planeDist:
      intersection_object = 8
    else:
      intersection_object = 7
    return d, intersection_object


@ti.func
def inside_particle_grid(ipos):
    pos = ipos * dx
    return bbox[0][0] <= pos[0] and pos[0] < bbox[1][0] and bbox[0][1] <= pos[
        1] and pos[1] < bbox[1][1] and bbox[0][2] <= pos[2] and pos[2] < bbox[
            1][2]


@ti.kernel
def initialize_particle_grid():
    for p in range(num_particles):
        # print(p)
        x = mpm.x[p]
        v = mpm.v[p]
        ipos = ti.Matrix.floor(x * particle_grid_res).cast(ti.i32)
        for i in range(-support, support + 1):
            for j in range(-support, support + 1):
                for k in range(-support, support + 1):
                    offset = ti.Vector([i, j, k])
                    box_ipos = ipos + offset
                    # print(box_ipos[0])
                    # print(box_ipos[1])
                    # print(box_ipos[2])
                    if inside_particle_grid(box_ipos):
                        box_min = box_ipos * (1 / particle_grid_res)
                        box_max = (box_ipos + ti.Vector([1, 1, 1])) * (
                            1 / particle_grid_res)
                        # print(box_min[0])
                        # print(box_min[1])
                        # print(box_min[2])
                        # print(box_max[0])
                        # print(box_max[1])
                        # print(box_max[2])
                        if sphere_aabb_intersect_motion(
                                box_min, box_max, x - 0.5 * shutter_time * v,
                                x + 0.5 * shutter_time * v, sphere_radius):
                            # print(voxel_has_particle[box_ipos])
                            ti.append(pid.parent(), box_ipos, p)
                            voxel_has_particle[box_ipos] = 1


@ti.func
def dda_particle(eye_pos, d, t, step):

    grid_res = particle_grid_res  #8

    # bounding box
    bbox_min = bbox[0]
    bbox_max = bbox[1]

    hit_pos = ti.Vector([0.0, 0.0, 0.0])
    normal = ti.Vector([0.0, 0.0, 0.0])
    c = ti.Vector([0.0, 0.0, 0.0])
    for i in ti.static(range(3)):
        if abs(
                d[i]
        ) < 1e-6:  #iterating over three components of direction vector from rayCast func
            d[i] = 1e-6  #assigning a lower bound to direction vec components... not sure why?

    inter, near, far = ray_aabb_intersection(bbox_min, bbox_max, eye_pos,
                                             d)  #findimg
    near = max(0, near)

    closest_intersection = inf
    ######## rendering 10 particles with no DDA########
    # for k in range(mpm.n_particles):
    #   pos = mpm.x[k]
    #   x = ti.Vector([ pos[0], pos[1], pos[2]])
    #   # p = pid[ipos[0], ipos[1], ipos[2], k]
    #   # v = particle_v[p]
    #   # x = particle_x[p] + t * v
    #   # color = particle_color[p]
    #   # ray-sphere intersection
    #   dist, poss = intersect_sphere(eye_pos, d, x, 0.1)
    #   hit_pos = poss
    #   if dist < closest_intersection and dist > 0:
    #     hit_pos = eye_pos + dist * d
    #     closest_intersection = dist
    #     normal = ti.Matrix.normalized(hit_pos - x)
    # return closest_intersection, normal
    ####################################################

    # print(inter)
    if inter:
        pos = eye_pos + d * (near + eps)

        rinv = 1.0 / d
        rsign = ti.Vector([0, 0, 0])
        for i in ti.static(range(3)):
            if d[i] > 0:
                rsign[i] = 1
            else:
                rsign[i] = -1

        o = grid_res * pos
        ipos = ti.Matrix.floor(o).cast(int)
        dis = (ipos - o + 0.5 + rsign * 0.5) * rinv
        running = 1
        # DDA for voxels with at least one particle
        while running:
            inside = inside_particle_grid(ipos)
            # print(inside)
            if inside:
                # once we actually intersect with a voxel that contains at least one particle, loop over the particle list
                num_particles = voxel_has_particle[ipos]
                if num_particles != 0:
                    num_particles = ti.length(pid.parent(), ipos)
                    # print (num_particles)
                for k in range(num_particles):
                    p = pid[ipos[0], ipos[1], ipos[2], k]
                    # pos = mpm.x[p]
                    v = mpm.v[p]
                    # x = mpm.x[p]
                    # if step == 0:
                    #     x = mpm.x[p]
                    # else:
                    x = mpm.x[p] + ((step) * v)
                    # x = mpm.x[p]
                    # x = ti.Vector([ pos[0], pos[1], pos[2]])
                    # p = pid[ipos[0], ipos[1], ipos[2], k]
                    # v = particle_v[p] #motion blur-- whatever moves should have this
                    # x = particle_x[p] + t * v
                    # color = particle_color[p]
                    # ray-sphere intersection
                    dist, poss = intersect_sphere(eye_pos, d, x, sphere_radius)
                    hit_pos = poss
                    if dist < closest_intersection and dist > 0:
                        hit_pos = eye_pos + dist * d
                        closest_intersection = dist
                        normal = ti.Matrix.normalized(hit_pos - x)
                        # c = color
            else:
                running = 0
                normal = [0, 0, 0]

            if closest_intersection < inf:
                running = 0
            else:
                # hits nothing. Continue ray marching
                mm = ti.Vector([0, 0, 0])
                if dis[0] <= dis[1] and dis[0] <= dis[2]:
                    mm[0] = 1
                elif dis[1] <= dis[0] and dis[1] <= dis[2]:
                    mm[1] = 1
                else:
                    mm[2] = 1
                dis += mm * rsign * rinv
                ipos += mm * rsign

    return closest_intersection, normal


@ti.func
def RayMarch(ro, rd, t):
    intersection_object = 0
    dO = 0.0
    i = 0
    while i < MAX_STEPS:
        p = ro + rd * dO
        dS, intersection_object = GetDist(p, t)
        dO += dS
        if dO > MAX_DIST or dS < SURF_DIST:
            break
        i = i + 1
    return dO, intersection_object


@ti.func
def rayCast(eye_pos, d, t, step):
    sdf_dis, intersection_object = RayMarch(eye_pos, d, t)
    particle_dis, normal = dda_particle(eye_pos, d, t, step)
    if min(sdf_dis, particle_dis) == particle_dis:
        intersection_object = 5
    return min(sdf_dis, particle_dis), normal, intersection_object, sdf_dis
    # return particle_dis


@ti.func
def normalize(p):
    return ti.Vector([p[0] / length(p), p[1] / length(p), p[2] / length(p)])


@ti.func
def GetNormal(p, t):
    d, intersection_object = GetDist(p, t)
    e1 = ti.Vector([0.01, 0.0, 0.0])
    e2 = ti.Vector([0.0, 0.01, 0.0])
    e3 = ti.Vector([0.0, 0.0, 0.01])
    x, intersection_object = GetDist(p - e1, t)
    y, intersection_object = GetDist(p - e2, t)
    z, intersection_object = GetDist(p - e3, t)
    n = ti.Vector([d - x, d - y, d - z])
    return normalize(n)


@ti.func
def clamp(p):

    if p < 0:
        p = 0
    if p > 1:
        p = 1
    return p


@ti.func
def GetLight(p, t, hit, nor, step):
    lightPos = ti.Vector([0.0 + ti.sin(t), 5.0, 6.0 + ti.cos(t)])

    l = normalize(lightPos - p)
    n = GetNormal(p, t)
    if hit == 5: #particles
        n = nor
    else: #sphere or plane
        n = GetNormal(p, t)

    diff = clamp(ti.dot(n, l))
    d, n_, intersection_object, sdf = rayCast(p + n * SURF_DIST * 2.0, l, t, step)
    if (d < length(lightPos - p)):
        diff = diff * 0.1
    diff = (diff + 1.0)/2.0
    return diff


@ti.kernel
def clear_pid():
    for i, j, k in voxel_has_particle:
        voxel_has_particle[i, j, k] = 0
        # ti.deactivate(pid.parent(), [i, j, k])
    for i, j, k in ti.ndrange(particle_grid_res * 4, particle_grid_res * 4,
                              particle_grid_res * 4):
        ti.deactivate(pid.parent(), [i, j, k])


@ti.kernel
def paint(t: ti.f32):
    # Parallized over all pixels
    # fin = ti.Vector([0.0, 0.0, 0.0])
    # intensity = 0.0

    # for i,j in pixels: 
    #     uv = ti.Vector([((i / 640) - 0.5) * (2), (j / 320) - 0.5])
    #     ro = ti.Vector([0.0, 1.0, 1.0])
    #     rd = ti.normalized(ti.Vector([uv[0], uv[1], 1.0]))

    #     d, no, intersection_object = rayCast(ro, rd, t+(0.03*0), 0.03*0)
    #     p = ro + rd * d
    #     light = GetLight(p, t+(0.03*0), intersection_object, no, 0.03*0)

    #     if intersection_object == 8: #if it hit the plane
    #         fin = light * plane_color
    #     elif intersection_object == 7: #if it hit the sphere
    #         fin = light * sphere_color

    #     elif intersection_object == 5: #if it hit the particle
    #         fin = light * particle_color

    #     pixels[i, j] = ti.Vector([fin[0], fin[1], fin[2], 1.0]) #color

##############  MOTION BLUR ATTEMPT 1 ################
    original_t = t
    for x in range(3):
        step_t = original_t + 0.3*x
        # step(t+0.03)
        for i in range(n*2): 
            for j in range(n): 
                uv = ti.Vector([((i / 640) - 0.5) * (2), (j / 320) - 0.5])
                ro = ti.Vector([0.0, 1.0, 1.0])
                rd = ti.normalized(ti.Vector([uv[0], uv[1], 1.0]))

                d, no, intersection_object, sdf = rayCast(ro, rd, step_t, 0.3*x)
                p = ro + rd * d
                light = GetLight(p, step_t, intersection_object, no, 0.3*x)
              
                # rendering the backgound initially, do this at the beginning of each 3 frame loop
                if x == 0:
                  sdf_p = ro + rd * sdf
                  # putting in 6 so that it renders the background instead of the particles
                  sdf_light = GetLight(sdf_p, original_t, 6, no, 0.3*x)
                  pixels[i, j] = ti.Vector([sdf_light, sdf_light, sdf_light, 1.0]) #color
              
                # rendering the particles at the third time step most opaque, the ones at the second and first less opaque to create a trail effect
                if intersection_object == 5:
                  if x == 0:
                      # doing pixel = pixels + ... to add the particle color value on top of the background
                      pixels[i, j] += ti.Vector([light * 0.1, light * 0.1, light * 0.1, 1.0])
                  if x == 1:
                      pixels[i, j] += ti.Vector([light * 0.3, light * 0.3, light * 0.3, 1.0])
                  if x == 2:
                      pixels[i, j] += ti.Vector([light * 0.6, light * 0.6, light * 0.6, 1.0])
##############  MOTION BLUR ATTEMPT 1 ################            
      

##############  MOTION BLUR ATTEMPT 2 ################
    # fin = ti.Vector([0.0, 0.0, 0.0])
    # intensity = 0.0

    # for i,j in pixels: 
    #     pixels[i, j] = ti.Vector([0, 0, 0, 1.0]) #color
    # for x in range(3):
    #     for i in range(n*2): 
    #         for j in range(n): 
    #             uv = ti.Vector([((i / 640) - 0.5) * (2), (j / 320) - 0.5])
    #             ro = ti.Vector([0.0, 1.0, 1.0])
    #             rd = ti.normalized(ti.Vector([uv[0], uv[1], 1.0]))
    #             # for x in range(3):
    #             d, no, intersection_object = rayCast(ro, rd, t+(0.03*x), 0.03*x)
    #             p = ro + rd * d
    #             light = GetLight(p, t+(0.03*x), intersection_object, no, 0.03*x)
    #             #     if x == 0:
    #             #         intensity = intensity + (light * 0.1)
    #             #     if x == 1:
    #             #         intensity = intensity + (light * 0.3)
    #             #     if x == 2:
    #             #         intensity = intensity + (light * 0.6)
    #             if x == 0:
    #                 pixels[i, j] = pixels[i, j] + ti.Vector([intensity + (light * 0.1), intensity + (light * 0.1), intensity + (light * 0.1), 1.0])
    #             if x == 1:
    #                 pixels[i, j] = pixels[i, j] + ti.Vector([intensity + (light * 0.3), intensity + (light * 0.3), intensity + (light * 0.3), 1.0])
    #             if x == 2:
    #                 pixels[i, j] = pixels[i, j] + ti.Vector([intensity + (light * 0.6), intensity + (light * 0.6), intensity + (light * 0.6), 1.0])
    ##############  MOTION BLUR ATTEMPT 2 ################


    ##############  MOTION BLUR ATTEMPT 3 ################
    # fin = 0.0
    # for x in range(3):
    #     for i in range(n*2): 
    #         for j in range(n): 
    #             uv = ti.Vector([((i / 640) - 0.5) * (2), (j / 320) - 0.5])
    #             ro = ti.Vector([0.0, 1.0, 1.0])
    #             rd = ti.normalized(ti.Vector([uv[0], uv[1], 1.0]))
              
    #             d, no, intersection_object= rayCast(ro, rd, t+(0.03*x), 0.03*x)
    #             p = ro + rd * d
    #             light = GetLight(p, t+(0.03*x), intersection_object, no, 0.03*x)
    #             if x == 0:
    #                 fin = fin + (light * 0.1)
    #             if x == 1:
    #                 fin = fin + (light * 0.3)
    #             if x == 2:
    #                 fin = fin + (light * 0.6)
          
    #             pixels[i, j] = ti.Vector([fin, fin, fin, 1.0]) #color
    ##############  MOTION BLUR ATTEMPT 3 ################

gui = ti.GUI("Fractl", (n * 2, n))


@ti.kernel
def initialize_particle_x(x: ti.ext_arr(), v: ti.ext_arr()):
    for i in range(num_particles[None]):
        for c in ti.static(range(3)):
            particle_x[i][c] = x[i, c]
            particle_v[i][c] = v[i, c]

@ti.func
def step(t):
    mpm.step(3e-2, t)

def main():
    for frame in range(1000000):
        clear_pid()
        mpm.step(3e-2, frame * 0.03)
        # colors = np.array([0x068587, 0xED553B, 0xEEEEF0], dtype=np.uint32)
        np_x, np_v, np_material = mpm.particle_info()
        # part_x = np_x.item((20,0))
        # part_y = np_x.item((20,1))
        # part_z = np_x.item((20,2))
        # print(frame)
        for i in range(3):

            # bbox values must be multiples of dx
            #   print(np_x[:, i].min())
            #   print(np_x[:, i].max())
            min_val = (math.floor(np_x[:, i].min() * particle_grid_res) -
                       3) / particle_grid_res
            max_val = (math.floor(np_x[:, i].max() * particle_grid_res) +
                       3) / particle_grid_res
            # print(min)
            # print(max)
            # if min_val == math.nan:
            #   # print("hi")
            #   min_val = -n/2
            # if max_val == math.nan:
            #   # print("by")
            #   max_val = n/2

            # bbox[0][i] = min_val
            bbox[1][i] = max_val
            bbox[0][i] = min_val
            print(bbox[0][i])
            print(bbox[1][i])

        #clear particle grid and pid voxel has particle
        # print('num_input_particles =', num_part)

        # for k in ti.static(range(27)):
        #   base_coord = (inv_dx * particle_x[i] - 0.5).cast(ti.i32) + ti.Vector(
        #       [k // 9, k // 3 % 3, k % 3])
        #   grid_density[base_coord // grid_visualization_block_size] = 1

        initialize_particle_x(np_x, np_v)
        initialize_particle_grid()

        #smaller timestep or implicit time integrator for water/snow error

        # part = ti.Vector([np_x.item((0,0)), np_x.item((0,1)), np_x.item((0,2))])
        # np_x = np_x / 10.0

        # # simple camera transform
        # # screen_x = ((np_x[:, 0] + np_x[:, 2]) / 2 ** 0.5) - 0.2
        # screen_x = (np_x[:, 0])
        # screen_y = (np_x[:, 1])

        # screen_pos = np.stack([screen_x, screen_y], axis=1)
        # casted = ti.cast(np_x, ti.ext_arr())
        # mat_int = mat.cast(int)
        # mat_int2 = mat.cast(ti.i32)
        # size = np_x.size
        paint(frame * 0.03)
        gui.set_image(pixels)
        # gui.circle([0 / 10, 1 / 10], radius=20, color=0xFF0000)
        # gui.circles(screen_pos, radius=1.5, color=colors[np_material])
        gui.show()


if __name__ == '__main__':
    main()
