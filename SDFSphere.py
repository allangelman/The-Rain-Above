# fractal.py

import taichi as ti
import numpy as np
import math
from mpm_solver import MPMSolver
from renderer_utils import out_dir, ray_aabb_intersection, inf, eps, \
  intersect_sphere, sphere_aabb_intersect_motion, inside_taichi

ti.require_version(0, 5, 10)
ti.init(arch=ti.x64, debug=False, print_ir=False)
# ti.core.toggle_advanced_optimization(False)

n = 8
m = 20
hit_sphere = 0
pixels = ti.Vector(4, dt=ti.f32, shape=(n * 16, n*9))
support = 1
shutter_time = 0.5e-3
sphere_radius = 0.03
MAX_STEPS = 100
MAX_STEPS_reflection = 50
MAX_DIST = 100.0
SURF_DIST = 0.01
SURF_DIST_reflection = 0.02
REFRACT_INDEX = 1.6
DELTA = 0.001
refractionRatio = 1.0 / REFRACT_INDEX
distanceFactor = 1.0
max_num_particles_per_cell = 8192 * 1024
voxel_has_particle = ti.var(dt=ti.i32)
cloud_color = ti.Vector([170/255, 244/255, 255/255])
cloud_color2 = ti.Vector([209/255, 250/255, 200/255])
cloud_color3 = ti.Vector([50/255, 255/255, 235/255])
plane_color = ti.Vector([210/255, 230/255, 249/255])
particle_color = ti.Vector([107/255, 115/255, 194/255])
capsule_color = ti.Vector([234/255, 244/255, 100/255])
capsule_color2 = ti.Vector([100/255, 189/255, 220/255])
wheel_color = ti.Vector([50/255, 250/255, 170/255])
wheel2_color = ti.Vector([70/255, 130/255, 217/255])
asterick_color = ti.Vector([170/255, 50/255, 250/255])
# cloud_intersection = 0
# backgound_color = ti.Vector([0.9, 0.4, 0.6])
frameTime = 0.03
frameTimeBlur = 0.01
CLOUD  = 1
CLOUD2  = 2
CLOUD3  = 3
PLANE = 4
PARTICLES = 5
CAPSULE = 6
CAPSULE2 = 7
WHEEL = 8
ASTERICK = 9
WHEEL2 = 10
debug = True

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


mpm = MPMSolver(res=(64, 64, 64), size=15)
mpm.add_cube(lower_corner=[1, 12.0, 5.8],
             cube_size=[8, 1, 0.5],
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
def sdf_Box(p, s, r):
#     x = max(abs(p[0]) - s[0], 0.0)
#     y = max(abs(p[1]) - s[1], 0.0)
#     z = max(abs(p[2]) - s[2], 0.0)

    x = max(abs(p[0]) - s[0], 0.0)
    y = max(abs(p[1]) - s[1], 0.0)
    z = max(abs(p[2]) - s[2], 0.0)
    q = ti.Vector([x,y,z])

    return length(q) + min(max(q[0],max(q[1],q[2])),0.0) - r
    # return length(ti.Vector([x, y, z]))

@ti.func
def rotate(a):
    s = ti.sin(a)
    c = ti.cos(a)
    return ti.Matrix([[c, -s], [s, c]])

@ti.func
def rotate_axis_x(box_position, rot_mat):
    rotated_y = rot_mat[0,0] * box_position[1] + rot_mat[1,0] * box_position[2] 
    rotated_z = rot_mat[0,1] * box_position[1] + rot_mat[1,1] * box_position[2] 
    box_position_rotated = ti.Vector([box_position[0], rotated_y, rotated_z])

    return box_position_rotated

@ti.func
def rotate_axis_y(box_position, rot_mat):
    rotated_x = rot_mat[0,0] * box_position[0] + rot_mat[1,0] * box_position[2] 
    rotated_z = rot_mat[0,1] * box_position[0] + rot_mat[1,1] * box_position[2] 
    box_position_rotated = ti.Vector([rotated_x, box_position[1], rotated_z])

    return box_position_rotated

@ti.func
def rotate_axis_z(box_position, rot_mat):
    rotated_x = rot_mat[0,0] * box_position[0] + rot_mat[1,0] * box_position[1] 
    rotated_y = rot_mat[0,1] * box_position[0] + rot_mat[1,1] * box_position[1] 
    box_position_rotated = ti.Vector([rotated_x, rotated_y, box_position[2]])
    
    return box_position_rotated

@ti.func
def mix(x, y, a):
    return x * (1.0-a) + y * a

@ti.func
def opSmoothUnion(d1, d2, k):
    h = clamp( 0.5 + 0.5*(d2-d1)/k)
    return mix(d2, d1, h) - k*h*(1.0-h)

@ti.func
def clouds(p, x, y, z, bump0, bump1, bump2, bump3, bump4):
    s0 = ti.Vector([-1.0+x, 1.4+y, 6.0+z, bump0**0.5])
    dist0 = p - xyz(s0)
    sphereDist0 = length(dist0) - s0[3]
    
    s = ti.Vector([0.0+x, 2.0+y, 6.0+z, bump1**0.5])
    dist = p - xyz(s)
    sphereDist = length(dist) - s[3]
    
    sphere0_1 = opSmoothUnion(sphereDist0, sphereDist, 0.1)

    s2 = ti.Vector([1.0+x, 1.8+y, 6.0+z, bump2**0.5])
    dist2 = p - xyz(s2)
    sphereDist2 = length(dist2) - s2[3]

    sphere0_1_2 = opSmoothUnion(sphere0_1, sphereDist2, 0.1)
   
    s3 = ti.Vector([2.0+x, 1.5+y, 6.0+z, bump3**0.5])
    dist3 = p - xyz(s3)
    sphereDist3 = length(dist3) - s3[3]

    sphere0_1_2_3 = opSmoothUnion(sphere0_1_2, sphereDist3, 0.1)

    s4 = ti.Vector([2.6+x, 1.2+y, 6.0+z, bump4**0.5])
    dist4 = p - xyz(s4)
    sphereDist4 = length(dist4) - s4[3]

    sphere0_1_2_3_4 = opSmoothUnion(sphere0_1_2_3, sphereDist4, 0.1)
    
    box_position3 = p - ti.Vector([1.0+x, 3.3+y, 6.0+z])
    boxDist3 = sdf_Box(box_position3, ti.Vector([4.0, 2.0, 0.1]), 0.1)

    cloud = max(boxDist3, sphere0_1_2_3_4)
    
    return cloud
@ti.func
def GetDistCloud(p, t):
    cloud = 0.0
    cloud = clouds(p, 9.0, 2.8 - ti.sin(t*0.8)*0.1, -0.4, 0.7, 1.0, 1.25, 0.7, 0.4)
    return cloud
@ti.func
def GetDistCloud2(p, t):
    cloud = 0.0
    cloud = clouds(p, 5.0, 3.0 + ti.sin(t)*0.1, -0.8, 0.7, 1.25, 0.9, 0.4, 0.2)
    return cloud
@ti.func
def GetDistCloud3(p, t):
    cloud = 0.0
    cloud = clouds(p, 1.0, 2.6 + ti.cos(t*0.9)*0.1, -0.5, 0.6, 0.55, 1.35, 0.9, 0.6)
    return cloud

@ti.func
def planeSDF(p, n):
    # n_norm = ti.normalized(n)
    nxyz = ti.Vector([n[0], n[1], n[2]])
    return ti.dot(p,nxyz) + n[3]

@ti.func
def GetDist(p, t):
    intersection_object = 0
    # planeDist = p[1]
    planeDist = planeSDF(p, ti.Vector([0, 0, -1.0/ti.sqrt(101.0), 10.0/ti.sqrt(101.0)]))
    # d = 0.0
    # capsuleDist =0.0
    # capsuleDist2 = 0.0
    # if ti.static(debug): 
       
    capsuleDist = sdf_Capsule(p, ti.Vector([8.5,8,6]), ti.Vector([10.5,9,6]), 0.2)
    capsuleDist2 = sdf_Capsule(p, ti.Vector([3.5,10,6]), ti.Vector([5.5,9,6]), 0.2)
    
    rot_mat = rotate(t*0.3)
    rot_mat_a = rotate(t*0.5)
    rot_mat_static = rotate(0.7)
    # capsule_pos_rotated = rotate_axis_y(p, rot_mat)

    box_position_a = p - ti.Vector([2.5,8.5,6.0])
    box_position_rotated_a = rotate_axis_y(box_position_a, rot_mat_a)
    boxDist_a1 = sdf_Box(box_position_rotated_a, ti.Vector([0.5, 0.1, 0.1]), 0.1)
    boxDist_a2 = sdf_Box(box_position_rotated_a, ti.Vector([0.1, 0.5, 0.1]), 0.1)
    boxDist_a3 = sdf_Box(box_position_rotated_a, ti.Vector([0.1, 0.1, 0.5]), 0.1)

    # capsuleDist3 = sdf_Capsule(capsule_pos_rotated, ti.Vector([1.5,7.5,6]), ti.Vector([2.5,7.5,6]), 0.1)
    # capsuleDist4 = sdf_Capsule(capsule_pos_rotated, ti.Vector([2.0,7.0,6]), ti.Vector([2.0,8.0,6]), 0.1)
    # capsuleDist5 = sdf_Capsule(capsule_pos_rotated, ti.Vector([2.0,7.5,5.5]), ti.Vector([2.0,7.5,6.5]), 0.1)

    box_position = p - ti.Vector([7.0, 11.0, 6.0])
    box_position_rotated = rotate_axis_z(box_position, rot_mat)
    boxDist = sdf_Box(box_position_rotated, ti.Vector([1, 0.1, 1]), 0.1)
    boxDist2 = sdf_Box(box_position_rotated, ti.Vector([0.1, 1, 1]), 0.1)

    box_position2 = p - ti.Vector([2.0,12.0,6.0])
    box_position_rotated2 = rotate_axis_x(box_position2, rot_mat)
    box_position_rotated2_static = rotate_axis_y(box_position_rotated2, rot_mat_static)
    boxDist3 = sdf_Box(box_position_rotated2_static, ti.Vector([0.3, 0.1, 1]), 0.1)
    boxDist4 = sdf_Box(box_position_rotated2_static, ti.Vector([0.3, 1, 0.1]), 0.1)

    d = min(planeDist, capsuleDist, capsuleDist2, boxDist, boxDist2, boxDist_a1, boxDist_a2, boxDist_a3, boxDist3, boxDist4)

    # else:
    #     d = min(planeDist, capsuleDist, capsuleDist2, boxDist, boxDist2)

    # box_position3 = p - ti.Vector([0.7, 0.1, 6])
    # boxDist3 = sdf_Box(box_position3, ti.Vector([2.2, 0.25, 0.25]))

    if d == planeDist:
        intersection_object = PLANE    
    elif d == capsuleDist:
        intersection_object = CAPSULE
    elif d == capsuleDist2:
        intersection_object = CAPSULE2
    elif d == boxDist or d == boxDist2:
        intersection_object = WHEEL
    elif d == boxDist3 or d == boxDist4:
        intersection_object = WHEEL2
    else:
        intersection_object = ASTERICK
    
    # print(cloud_intersection)
    
    return d, intersection_object


@ti.func
def inside_particle_grid(ipos):
    # pos = 10*(ipos/particle_grid_res)
    pos = grid_to_world(ipos)
    return bbox[0][0] <= pos[0] and pos[0] < bbox[1][0] and bbox[0][1] <= pos[
        1] and pos[1] < bbox[1][1] and bbox[0][2] <= pos[2] and pos[2] < bbox[
            1][2]

@ti.func
def grid_to_world_forkernel(x):
  return (x*10) / particle_grid_res 

def grid_to_world(x):
  return (x*10) / particle_grid_res 

@ti.func
def world_to_grid_forkernel(x):
  return (x * particle_grid_res) / 10

def world_to_grid(x):
  return (x * particle_grid_res) / 10 


@ti.kernel
def initialize_particle_grid():
    for p in range(num_particles[None]):
        x = mpm.x[p]
        v = mpm.v[p]
        # ipos = ti.Matrix.floor(x * particle_grid_res).cast(ti.i32)
        # ipos = ti.Matrix.floor((x * particle_grid_res)/10).cast(ti.i32) #particle postion to grid coorid
        ipos = ti.Matrix.floor(world_to_grid_forkernel(x)).cast(ti.i32) #particle postion to grid coorid
        for i in range(-support, support + 1):
            for j in range(-support, support + 1):
                for k in range(-support, support + 1):
                    offset = ti.Vector([i, j, k])
                    box_ipos = ipos + offset
                    # print(box_ipos[0])
                    if inside_particle_grid(box_ipos):
                        # print(box_ipos[0])
                        # print(box_ipos[1])
                        # print(box_ipos[2])

                        # box_min = box_ipos * (1/particle_grid_res)
                        # box_min = box_ipos * (10/ particle_grid_res) 
                        box_min = grid_to_world_forkernel(box_ipos)
                        # box_max = (box_ipos + ti.Vector([1, 1, 1])) * (1 / particle_grid_res)
                        # box_max = (box_ipos + ti.Vector([1, 1, 1])) * (10 / particle_grid_res)
                        box_max = grid_to_world_forkernel((box_ipos + ti.Vector([1, 1, 1])))

                        # if sphere_aabb_intersect_motion(
                        #         box_min, box_max, x - 0.5 * shutter_time * v,
                        #         x + 0.5 * shutter_time * v, sphere_radius):
                            # print(voxel_has_particle[box_ipos])
                        ti.append(pid.parent(), box_ipos, p)
                        voxel_has_particle[box_ipos] = 1
@ti.func
def dda_particle2(eye_pos, d, t, step):
    # bbox_min = bbox[0]
    # bbox_max = bbox[1]

    hit_pos = ti.Vector([0.0, 0.0, 0.0])
    normal = ti.Vector([0.0, 0.0, 0.0])
    # c = ti.Vector([0.0, 0.0, 0.0])
    # for i in ti.static(range(3)):
    #     if abs(
    #             d[i]
    #     ) < 1e-6:  #iterating over three components of direction vector from rayCast func
    #         d[i] = 1e-6  #assigning a lower bound to direction vec components... not sure why?

    # inter, near, far = ray_aabb_intersection(bbox_min, bbox_max, eye_pos,
    #                                          d)  #findimg
    # near = max(0, near)

    closest_intersection = inf
    for k in range(mpm.n_particles):
      vel = mpm.v[k]
      pos = mpm.x[k]
    #   x = ti.Vector([ pos[0], pos[1], pos[2]])
      x =  pos + step * vel
      # p = pid[ipos[0], ipos[1], ipos[2], k]
      # v = particle_v[p]
      # x = particle_x[p] + t * v
      # color = particle_color[p]
      # ray-sphere intersection
      dist, poss = intersect_sphere(eye_pos, d, x, sphere_radius)
      hit_pos = poss
      if dist < closest_intersection and dist > 0:
        hit_pos = eye_pos + dist * d
        closest_intersection = dist
        normal = ti.Matrix.normalized(hit_pos - x)
    return closest_intersection, normal

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
    # print(closest_intersection)

    if inter:
        pos = eye_pos + d * (near + eps)

        rinv = 1.0 / d
        rsign = ti.Vector([0, 0, 0])
        for i in ti.static(range(3)):
            if d[i] > 0:
                rsign[i] = 1
            else:
                rsign[i] = -1

        # o = (particle_grid_res * pos)/ 10.0
        o = world_to_grid(pos)
        # o = grid_res * pos 
        ipos = ti.Matrix.floor(o).cast(int)
        dis = (ipos - o + 0.5 + rsign * 0.5) * rinv
        running = 1
        num = 999999999
        num2 = 2000000002
        # DDA for voxels with at least one particle
        while running:
            inside = inside_particle_grid(ipos)
            if inside:
                # print(ipos[0])
                # print(ipos[1])
                # print(ipos[2])
                # once we actually intersect with a voxel that contains at least one particle, loop over the particle list
                num_particles = voxel_has_particle[ipos]
                if num_particles != 0:
                    # print(num)
                    # print(num_particles)
                    # world_pos = grid_to_world(ipos)
                    # print(world_pos[0])
                    # print(world_pos[1])
                    # print(world_pos[2])
                    num_particles = ti.length(pid.parent(), ipos)
                for k in range(num_particles):
                    # print(num2)
                    p = pid[ipos[0], ipos[1], ipos[2], k]
                    # v = mpm.v[p]
                    # x = mpm.x[p] + step * mpm.v[p]
                    x = mpm.x[p]     
                    # print(x[0])
                    # print(x[1])
                    # print(x[2])
                    
                    # print(d[0])
                    # print(d[1])
                    # print(d[2])
                    dist, poss = intersect_sphere(eye_pos, d, x, sphere_radius)
                    # print(num)
                    # print(dist)
                    hit_pos = poss
                    if dist < closest_intersection and dist > 0:
                        hit_pos = eye_pos + dist * d
                        closest_intersection = dist
                        normal = ti.Matrix.normalized(hit_pos - x)
            else:
                running = 0
                normal = [0, 0, 0]
            if closest_intersection < inf:
                # print(num)
                running = 0
            else:
                # print(num2)
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
    cloud_intersection = 0
    cloud_int = 0
    dO = 0.0
    clouddO = 0.02
    i = 0
    cloud = 0.0
    while i < MAX_STEPS:
        p = ro + rd * dO
        dS, intersection_object = GetDist(p, t)
        # clouddS = GetDistCloud(p, t)
        # dS = min(dS, GetDistCloud(p, t))
        dO += dS
        
        # p_cloud = ro + rd * clouddO
        # clouddS = GetDistCloud(p, t)
        # if (clouddS < dS):
        #     cloud_int = 1
        # if (clouddO >= SURF_DIST or clouddO < MAX_DIST) and clouddS < dS:
        #     clouddO += clouddS
        #     cloud_int = 1

        # if cloud_intersection == 1:
        #     cloud_int = 1
        if dO > MAX_DIST or dS < SURF_DIST:
            break
        i = i + 1
    return dO, intersection_object

@ti.func
def RayMarchCloud(ro, rd, t):
    dO = 0.0
    i = 0
    while i < MAX_STEPS:
        p = ro + rd * dO
        dS = GetDistCloud(p, t)
        dO += dS
        if dO > MAX_DIST or dS < SURF_DIST:
            break
        i = i + 1
    return dO

@ti.func
def RayMarchCloud2(ro, rd, t):
    dO = 0.0
    i = 0
    while i < MAX_STEPS:
        p = ro + rd * dO
        dS = GetDistCloud2(p, t)
        dO += dS
        if dO > MAX_DIST or dS < SURF_DIST:
            break
        i = i + 1
    return dO

@ti.func
def RayMarchCloud3(ro, rd, t):
    dO = 0.0
    i = 0
    while i < MAX_STEPS:
        p = ro + rd * dO
        dS = GetDistCloud3(p, t)
        dO += dS
        if dO > MAX_DIST or dS < SURF_DIST:
            break
        i = i + 1
    return dO

@ti.func
def RayMarch_reflection(ro, rd, t):
    intersection_object = 0
    cloud_intersection = 0
    dO = 0.0
    i = 0
    cloud = 0.0
    while i < MAX_STEPS_reflection:
        p = ro + rd * dO
        dS, intersection_object = GetDist(p, t)
        dO += dS
        if dO > MAX_DIST or dS < SURF_DIST_reflection:
            break
        i = i + 1
    return dO, intersection_object


@ti.func
def rayCast(eye_pos, d, t, step):
    cloud_intersection = 0
    cloud_intersection2 = 0
    cloud_intersection3 = 0
    sdf_dis, intersection_object= RayMarch(eye_pos, d, t)
    sdf_intersection = intersection_object
    sdf_discloud = RayMarchCloud(eye_pos, d, t)
    sdf_discloud2 = RayMarchCloud2(eye_pos, d, t)
    sdf_discloud3 = RayMarchCloud3(eye_pos, d, t)
    particle_dis, normal = dda_particle2(eye_pos, d, t, step)
    if min(sdf_dis, particle_dis) == particle_dis:
        intersection_object = PARTICLES
    if min(sdf_dis, particle_dis, sdf_discloud) == sdf_discloud:
        cloud_intersection = 1
    if min(sdf_dis, particle_dis, sdf_discloud3) == sdf_discloud3:
        cloud_intersection3 = 1
    if min(sdf_dis, particle_dis, sdf_discloud, sdf_discloud3, sdf_discloud2) == sdf_discloud2:
        cloud_intersection2 = 1
    return min(sdf_dis, particle_dis), normal, intersection_object, sdf_discloud, cloud_intersection, sdf_discloud2, cloud_intersection2, sdf_discloud3, cloud_intersection3, sdf_dis, sdf_intersection

@ti.func
def rayCast_reflection(eye_pos, d, t, step):
    sdf_dis, intersection_object = RayMarch_reflection(eye_pos, d, t)
    particle_dis, normal = dda_particle(eye_pos, d, t, step)
    if min(sdf_dis, particle_dis) == particle_dis:
        intersection_object = PARTICLES
    return min(sdf_dis, particle_dis), normal, intersection_object, sdf_dis

@ti.func
def normalize(p):
    return ti.Vector([p[0] / length(p), p[1] / length(p), p[2] / length(p)])

@ti.func
def GetNormal(p, t, intersection_object):
    d = 0.0
    x = 0.0
    y = 0.0
    z = 0.0
    # intersection_object = 0
    e1 = ti.Vector([0.01, 0.0, 0.0])
    e2 = ti.Vector([0.0, 0.01, 0.0])
    e3 = ti.Vector([0.0, 0.0, 0.01])
    if (intersection_object == CLOUD):
        d = GetDistCloud(p, t)
        x = GetDistCloud(p - e1, t)
        y = GetDistCloud(p - e2, t)
        z = GetDistCloud(p - e3, t)
    elif (intersection_object == CLOUD2):
        d = GetDistCloud2(p, t)
        x = GetDistCloud2(p - e1, t)
        y = GetDistCloud2(p - e2, t)
        z = GetDistCloud2(p - e3, t)
    elif (intersection_object == CLOUD3):
        d = GetDistCloud3(p, t)
        x = GetDistCloud3(p - e1, t)
        y = GetDistCloud3(p - e2, t)
        z = GetDistCloud3(p - e3, t)
    else:
        d, inter = GetDist(p, t)
        x, inter = GetDist(p - e1, t)
        y, inter = GetDist(p - e2, t)
        z, inter = GetDist(p - e3, t)
    n = ti.Vector([d - x, d - y, d - z])
    return normalize(n)



@ti.func
def clamp(p):

    if p < 0.0:
        p = 0.0
    if p > 1.0:
        p = 1.0
    return p

@ti.func
def reflect(I, N):
    return I - 2.0 * ti.dot(N, I) * N

@ti.func
def refract(I, N, eta):
    R = ti.Vector([0.0,0.0,0.0])
    k = 1.0 - eta * eta * (1.0 - ti.dot(N, I) * ti.dot(N, I))
    if (k < 0.0):
        R = ti.Vector([0.0,0.0,0.0])
    else:
        R = eta * I - (eta * ti.dot(N, I) + ti.sqrt(k)) * N
    return R

@ti.func
def getColor(int_ob):
    fin = ti.Vector([0.0,0.0,0.0])
    if int_ob == PLANE: #if it hit the plane
        fin = plane_color
    elif int_ob == CLOUD: #if it hit the cloud
        fin = cloud_color
    elif int_ob == CLOUD2: #if it hit the cloud
        fin = cloud_color2
    elif int_ob == CLOUD3: #if it hit the cloud
        fin = cloud_color3
    elif int_ob == PARTICLES: #if it hit the particle
        fin = particle_color
    elif int_ob == CAPSULE: #if it hit the capsule 1
        fin = capsule_color
    elif int_ob == CAPSULE2: #if it hit the capsule 2
        fin = capsule_color2
    elif int_ob == WHEEL: #if it hit the capsule 2
        fin = wheel_color
    elif int_ob == WHEEL2:
        fin = wheel2_color
    else:
        fin = asterick_color
    return fin

@ti.func
def GetLight(p, t, hit, nor, step, rd):
    # lightPos = ti.Vector([0.0 + ti.sin(t), 7.0, 6.0 + ti.cos(t)])
    lightPos = ti.Vector([0, 35, 1.0])

    l = normalize(lightPos - p)
    n = ti.Vector([0.0, 0.0, 0.0])
    if hit == PARTICLES: #particles
        n = nor
    else: #sphere or plane
        n = GetNormal(p, t, hit)
    # attenuating the light
    atten = 1.0 / (1.0 + l*0.2 + l*l*0.1)
    spec = pow(max(ti.dot( reflect(-l, n), -rd ), 0.0), 8.0)
    diff = clamp(ti.dot(n, l))    
    # if hit == CLOUD:
    #     print(n[0])
    #     print(n[1])
    #     print(n[2])
    # d, n_, intersection_object, sdf = rayCast(p + n * SURF_DIST * 2.0, l, t, step)
    # if (d < length(lightPos - p)):
    #     diff = diff * 0.1
    diff = (diff + 1.0)/2.0


    sceneCol = (getColor(hit)*(diff + 0.15) + ti.Vector([0.8, 0.8, 0.2])*spec*0.5) * atten
    # if (hit == CLOUD2):
    #     print(getColor(hit)[0])
    #     print(getColor(hit)[1])
    #     print(getColor(hit)[2])
    return sceneCol , n


@ti.kernel
def clear_pid():
    for i, j, k in voxel_has_particle:
        voxel_has_particle[i, j, k] = 0
    for i, j, k in ti.ndrange(particle_grid_res * 4, particle_grid_res * 4,
                              particle_grid_res * 4):
        ti.deactivate(pid.parent(), [i, j, k])



# grid_to_world((world_to_grid(np_x.min) - 3))
# @ti.func def world_to_grid(x): return x / 10 * partile_grid_res 
#make two versions of these functions, ti.func and python (call ti.func version in kernel)
#setting min and max to 0 and 10 

@ti.kernel
def paint(t: ti.f32):
    fin = ti.Vector([0.0, 0.0, 0.0]) # Parallized over all pixels
    intensity = 0.0

    for i in range(n*16): #this is parallilized
        for j in range(n*9):
            for x in range(3):
                uv = ti.Vector([((i / (16*n)) - 0.5) * (2), (j / (9*n)) - 0.5])
                
                starting_y = 12.0
                ending_y = 5.0
                motion_y = -t
                lookat_starting_y = 12.0
                lookat_ending_y = 5.0
                # motion_y = 0

                ro = ti.Vector([5.0, starting_y , 1.0])
                lookat = ti.Vector([5.0, lookat_starting_y, 6.0])

                if starting_y + motion_y > ending_y:
                    ro = ti.Vector([5.0, starting_y + motion_y, 1.0])
                    lookat = ti.Vector([5.0, lookat_starting_y + motion_y, 6.0]) 
                else:
                    ro = ti.Vector([5.0, ending_y, 1.0])
                    lookat = ti.Vector([5.0, lookat_ending_y, 6.0])

                zoom = 1.0

                forward = ti.normalized(lookat - ro)
                right = ti.cross(ti.Vector([0.0, 1.0, 0.0]), forward)
                up = ti.cross(forward, right)

                center = ro + forward*zoom
                intersection = center + uv[0]*right + uv[1]*up
                rd = ti.normalized(intersection - ro)

                d, no, intersection_object, clouddO, cloud_intersection, clouddO2, cloud_intersection2, clouddO3, cloud_intersection3, sdf, sdf_inter= rayCast(ro, rd, t, frameTimeBlur*x)
                p = ro + rd * d
                light, normal = GetLight(p, t, intersection_object, no, frameTimeBlur*x, rd)
                
                if x == 0:
                    sdf_p = ro + rd * sdf
                    # putting in CAPSULE  so that it renders the background instead of the particles
                    sdf_light, normal_sdf = GetLight(sdf_p, t, sdf_inter, no, frameTimeBlur*x, rd)
                    # if (intersection_object == CAPSULE):
                    #     rd2 = reflect(rd, normal)
                    #     d2, no2, intersection_object2 = rayCast_reflection(ro +  normal*.003, rd2, t+(0.03*0), 0.03*0)
                
                    #     p += rd2*d2
                
                    #     light2, normal2 = GetLight(p, t, intersection_object2, no2, frameTimeBlur*x, rd2)
                    #     sdf_light += light2*0.30

                    pixels[i, j] = ti.Vector([sdf_light[0], sdf_light[1], sdf_light[2], 1.0]) #color
                alpha = 0.8
                alpha1 = 0.4
                alpha2 = 0.2
                if intersection_object == PARTICLES:
                    if x == 0:
                      # doing pixel = pixels + ... to add the particle color value on top of the background

                        pixels[i, j] = pixels[i, j]*alpha + ti.Vector([light[0], light[1], light[2], 1.0/(1-alpha)])*(1-alpha)
                    if x == 1:
                        pixels[i, j] = pixels[i, j]*alpha1 + ti.Vector([light[0]*0.6, light[1]*0.6, light[2]*0.6, 1.0/(1-alpha1)])*(1-alpha1)
                    if x == 2:
                        pixels[i, j] = pixels[i, j]*alpha2 + ti.Vector([light[0]*0.9, light[1]*0.9, light[2]*0.9, 1.0/(1-alpha2)])*(1-alpha2)
                
                # alpha4 = 0.5


                # if x == 2:
                #     if cloud_intersection == 1:
                #         p_cloud = ro + rd * clouddO
                #         light_cloud, normal_cloud = GetLight(p_cloud, t, CLOUD, no, 0, rd)
                #         # light += light_cloud*0.30
                #         pixels[i, j] = pixels[i, j] + ti.Vector([light_cloud[0]*0.30, light_cloud[1]*0.30, light_cloud[2]*0.30, 1.0])
                #     if cloud_intersection3 == 1:
                #         p_cloud3 = ro + rd * clouddO3
                #         light_cloud3, normal_cloud3 = GetLight(p_cloud3, t, CLOUD3, no, 0, rd)
                #         # light += light_cloud3*0.30
                #         pixels[i, j] = pixels[i, j] + ti.Vector([light_cloud3[0]*0.30, light_cloud3[1]*0.30, light_cloud3[2]*0.30, 1.0])
                #     if cloud_intersection2 == 1:
                #         p_cloud2 = ro + rd * clouddO2
                #         light_cloud2, normal_cloud2 = GetLight(p_cloud2, t, CLOUD2, no, 0, rd)
                #         # light += light_cloud2*0.30
                #         # pixels[i, j] = pixels[i, j]*alpha4 + ti.Vector([light_cloud2[0], light_cloud2[1], light_cloud2[2], 1.0/(1-alpha4)])*(1-alpha4)
                #         pixels[i, j] = pixels[i, j] + ti.Vector([light_cloud2[0]*0.30, light_cloud2[1]*0.30, light_cloud2[2]*0.30, 1.0])
                
                
                
                # rd2 = reflect(rd, normal)
                # if (intersection_object != PARTICLES and intersection_object != PLANE and intersection_object != CLOUD):
                #     d2, no2, intersection_object2 = rayCast_reflection(ro +  normal*.003, rd2, t+(0.03*0), 0.03*0)
                    
                #     p += rd2*d2
                    
                #     light2, normal2 = GetLight(p, t+(0.03*0), intersection_object2, no2, 0.03*0, rd2)
                #     light += light2*0.20
                
                # pixels[i, j] = ti.Vector([light[0], light[1], light[2], 1.0]) #color

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

gui = ti.GUI("Fractl", (n * 16, n* 9))


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
        
        # for i in range(3):
        
        #     min_val = grid_to_world( world_to_grid( math.floor(np_x[:, i].min())) - 3 ) 
        
        #     max_val = grid_to_world( world_to_grid( math.floor(np_x[:, i].max())) + 3 ) 

        #     bbox[1][i] = max_val
        #     bbox[0][i] = min_val

        #clear particle grid and pid voxel has particle
        # initialize_particle_x(np_x, np_v)
        # initialize_particle_grid()

        #smaller timestep or implicit time integrator for water/snow error
        paint(frame * frameTime)
        
        gui.set_image(pixels)
        gui.show(f'{frame:04d}.png')
        # gui.show()


if __name__ == '__main__':
    main()
