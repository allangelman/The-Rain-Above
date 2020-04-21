# fractal.py

import taichi as ti
import numpy as np
import math
from mpm_solver import MPMSolver
from renderer_utils import out_dir, ray_aabb_intersection, inf, eps, \
  intersect_sphere, sphere_aabb_intersect_motion, inside_taichi

ti.require_version(0, 5, 10)
ti.init(arch=ti.x64, debug=True)

n = 320
m = 20
hit_sphere = 0
pixels = ti.Vector(4, dt=ti.f32, shape=(n * 2, n))
support = 1
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
frameTime = 0.03
SPHERE = 7
PLANE = 8
PARTICLES = 5

pid = ti.var(ti.i32)
num_particles = ti.var(ti.i32, shape=())
bbox = ti.Vector(3, dt=ti.f32, shape=2)
particle_grid_res = 64
particle_x = ti.Vector(3, dt=ti.f32)
particle_v = ti.Vector(3, dt=ti.f32)
fin = 0.0
max_num_particles = 1024 * 1024 * 4

@ti.layout
def buffers():
    ti.root.dense(ti.ijk, 2).dense(ti.ijk, particle_grid_res // 8).dense(
        ti.ijk, 8).place(voxel_has_particle)
    ti.root.dense(ti.ijk, 4).pointer(ti.ijk, particle_grid_res // 8).dense(
        ti.ijk, 8).dynamic(ti.l, max_num_particles_per_cell, 512).place(pid)
    ti.root.dense(ti.l, max_num_particles).place(particle_x, particle_v)


mpm = MPMSolver(res=(64, 64, 64), size=10)
mpm.add_cube(lower_corner=[0, 5, 6],
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
    capsuleDist = sdf_Capsule(p, ti.Vector([2,3,6]), ti.Vector([4,4,6]), 0.2)
    capsuleDist2 = sdf_Capsule(p, ti.Vector([-1,5,6]), ti.Vector([1,4,6]), 0.2)
    d = min(planeDist, sphereDist, capsuleDist, capsuleDist2)
    if d == planeDist:
      intersection_object = PLANE
    else:
      intersection_object = SPHERE
    return d, intersection_object


@ti.func
def inside_particle_grid(ipos):
    pos = 10*(ipos/particle_grid_res)
    return bbox[0][0] <= pos[0] and pos[0] < bbox[1][0] and bbox[0][1] <= pos[
        1] and pos[1] < bbox[1][1] and bbox[0][2] <= pos[2] and pos[2] < bbox[
            1][2]


@ti.kernel
def initialize_particle_grid():
    for p in range(num_particles[None]):
        x = mpm.x[p]
        v = mpm.v[p]
        # ipos = ti.Matrix.floor(x * particle_grid_res).cast(ti.i32)
        ipos = ti.Matrix.floor((x * particle_grid_res)/10).cast(ti.i32) #particle postion to grid coorid
        for i in range(-support, support + 1):
            for j in range(-support, support + 1):
                for k in range(-support, support + 1):
                    offset = ti.Vector([i, j, k])
                    box_ipos = ipos + offset
                    if inside_particle_grid(box_ipos):
                        # box_min = box_ipos * (1/particle_grid_res)
                        box_min = box_ipos * (10/ particle_grid_res) 
                        # box_max = (box_ipos + ti.Vector([1, 1, 1])) * (1 / particle_grid_res)
                        box_max = (box_ipos + ti.Vector([1, 1, 1])) * (10 / particle_grid_res)

                        # if sphere_aabb_intersect_motion(
                        #         box_min, box_max, x - 0.5 * shutter_time * v,
                        #         x + 0.5 * shutter_time * v, sphere_radius):
                            # print(voxel_has_particle[box_ipos])
                        ti.append(pid.parent(), box_ipos, p)
                        voxel_has_particle[box_ipos] = 1


@ti.func
def dda_particle(eye_pos, d, t, step):

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

    if inter:
        pos = eye_pos + d * (near + eps)

        rinv = 1.0 / d
        rsign = ti.Vector([0, 0, 0])
        for i in ti.static(range(3)):
            if d[i] > 0:
                rsign[i] = 1
            else:
                rsign[i] = -1

        o = (particle_grid_res * pos)/ 10.0
        # o = grid_res * pos 
        ipos = ti.Matrix.floor(o).cast(int)
        dis = (ipos - o + 0.5 + rsign * 0.5) * rinv
        running = 1
        # DDA for voxels with at least one particle
        while running:
            inside = inside_particle_grid(ipos)
            if inside:
                # once we actually intersect with a voxel that contains at least one particle, loop over the particle list
                num_particles = voxel_has_particle[ipos]
                if num_particles != 0:
                    num_particles = ti.length(pid.parent(), ipos)
                for k in range(num_particles):
                    p = pid[ipos[0], ipos[1], ipos[2], k]
                    v = mpm.v[p]
                    x = mpm.x[p] + step * mpm.v[p]

                    dist, poss = intersect_sphere(eye_pos, d, x, sphere_radius)
                    hit_pos = poss
                    if dist < closest_intersection and dist > 0:
                        hit_pos = eye_pos + dist * d
                        closest_intersection = dist
                        normal = ti.Matrix.normalized(hit_pos - x)
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
        intersection_object = PARTICLES
    return min(sdf_dis, particle_dis), normal, intersection_object, sdf_dis

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
    lightPos = ti.Vector([0.0 + ti.sin(t), 7.0, 6.0 + ti.cos(t)])

    l = normalize(lightPos - p)
    n = GetNormal(p, t)
    if hit == PARTICLES: #particles
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
    for i, j, k in ti.ndrange(particle_grid_res * 4, particle_grid_res * 4,
                              particle_grid_res * 4):
        ti.deactivate(pid.parent(), [i, j, k])


@ti.kernel
def paint(t: ti.f32):
    fin = ti.Vector([0.0, 0.0, 0.0]) # Parallized over all pixels
    intensity = 0.0

    for i,j in pixels: 
        uv = ti.Vector([((i / 640) - 0.5) * (2), (j / 320) - 0.5])
        
        starting_y = 5.0
        ending_y = 1.0
        motion_y = -t*4
  
        ro = ti.Vector([1.0, starting_y , 1.0])
        lookat = ti.Vector([1.0, starting_y, 6.0])

        if starting_y + motion_y > ending_y:
          ro = ti.Vector([1.0, starting_y + motion_y, 1.0])
          lookat = ti.Vector([1.0, starting_y + motion_y, 6.0]) 
        else:
          ro = ti.Vector([1.0, ending_y, 1.0])
          lookat = ti.Vector([1.0, ending_y, 6.0])

        zoom = 1.0

        forward = ti.normalized(lookat - ro)
        right = ti.cross(ti.Vector([0.0, 1.0, 0.0]), forward)
        up = ti.cross(forward, right)

        center = ro + forward*zoom
        intersection = center + uv[0]*right + uv[1]*up
        rd = ti.normalized(intersection - ro)

        d, no, intersection_object = rayCast(ro, rd, t+(0.03*0), 0.03*0)
        p = ro + rd * d
        light = GetLight(p, t+(0.03*0), intersection_object, no, 0.03*0)

        if intersection_object == PLANE: #if it hit the plane
            fin = light * plane_color
        elif intersection_object == SPHERE: #if it hit the sphere
            fin = light * sphere_color

        elif intersection_object == PARTICLES: #if it hit the particle
            fin = light * particle_color

        pixels[i, j] = ti.Vector([fin[0], fin[1], fin[2], 1.0]) #color

##############  MOTION BLUR ATTEMPT 1 ################
    # original_t = t
    # # step(t+0.03)
    # for i in range(n*2): #this is parallilized
    #     for j in range(n):
    #         for x in range(3):
    #             step_t = original_t + frameTime*x
    #             uv = ti.Vector([((i / 640) - 0.5) * (2), (j / 320) - 0.5])
    #             ro = ti.Vector([0.0, 1.0, 1.0])
    #             rd = ti.normalized(ti.Vector([uv[0], uv[1], 1.0]))

    #             d, no, intersection_object, sdf = rayCast(ro, rd, step_t, frameTime*x)
    #             p = ro + rd * d
    #             light = GetLight(p, step_t, intersection_object, no, frameTime*x)
              
    #             # rendering the backgound initially, do this at the beginning of each 3 frame loop
    #             if x == 0:
    #               sdf_p = ro + rd * sdf
    #               # putting in SPHERE so that it renders the background instead of the particles
    #               sdf_light = GetLight(sdf_p, original_t, SPHERE, no, frameTime*x)
    #               pixels[i, j] = ti.Vector([sdf_light, sdf_light, sdf_light, 1.0]) #color
              
    #             # rendering the particles at the third time step most opaque, the ones at the second and first less opaque to create a trail effect
    #             if intersection_object == PARTICLES:
    #               if x == 0:
    #                   # doing pixel = pixels + ... to add the particle color value on top of the background
    #                   pixels[i, j] += ti.Vector([light * 0.1, light * 0.1, light * 0.1, 1.0])
    #               if x == 1:
    #                   pixels[i, j] += ti.Vector([light * 0.3, light * 0.3, light * 0.3, 1.0])
    #               if x == 2:
    #                   pixels[i, j] += ti.Vector([light * 0.6, light * 0.6, light * 0.6, 1.0])
##############  MOTION BLUR ATTEMPT 1 ################            

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
        mpm.step(3e-2, frame * frameTime)
  
        np_x, np_v, np_material = mpm.particle_info()
        
        for i in range(3):
            # min_val = (math.floor(np_x[:, i].min() * particle_grid_res) - 
            #           3) / particle_grid_res
            # max_val = (math.floor(np_x[:, i].max() * particle_grid_res) + 
            #           3) / particle_grid_res
            
            # min_val = 10*(math.floor(np_x[:, i].min() * particle_grid_res) - 
            #           3) / particle_grid_res
            # max_val = 10*(math.floor(np_x[:, i].max() * particle_grid_res) + 
            #           3) / particle_grid_res

            min_val = (math.floor(np_x[:, i].min()) / 10 * particle_grid_res - 3) / (particle_grid_res / 10)
            max_val = (math.floor(np_x[:, i].max()) / 10 * particle_grid_res + 3) / (particle_grid_res / 10)
            if min_val < 0:
              min_val = 0
            # grid_to_world((world_to_grid(np_x.min) - 3))
            # @ti.func def world_to_grid(x): return x / 10 * partile_grid_res 
            #make two versions of these functions, ti.func and python (call ti.func version in kernel)
            #setting min and max to 0 and 10 

            # min_val = 0
            # max_val = 10

            bbox[1][i] = max_val
            bbox[0][i] = min_val

        #clear particle grid and pid voxel has particle
        initialize_particle_x(np_x, np_v)
        initialize_particle_grid()

        #smaller timestep or implicit time integrator for water/snow error
        paint(frame * frameTime)
        
        gui.set_image(pixels)
        gui.show()


if __name__ == '__main__':
    main()
