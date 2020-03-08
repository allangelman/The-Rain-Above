# fractal.py

import taichi as ti
import numpy as np
from mpm_solver import MPMSolver

ti.cfg.arch = ti.cuda # Run on GPU by default

n = 320
m = 20
# pixels = ti.var(dt=ti.f32, shape=(n * 2, n))
pixels = ti.Vector(4, dt=ti.f32, shape=(n, n))

MAX_STEPS = 100
MAX_DIST = 100.0
SURF_DIST = 0.01

mpm = MPMSolver(res=(64, 64, 64), size=10)

mpm.add_cube(lower_corner=[0, 7, 6], cube_size=[3, 1, 0.5], material=MPMSolver.material_water)

mpm.set_gravity((0, -50, 0))

np_x, np_v, np_material = mpm.particle_info()
s_x = np.size(np_x, 0)
s_y = np.size(np_x, 1)
particles = ti.Vector(3, dt=ti.f32, shape=(s_x,s_y))
# # particles = ti.Vector(4, dt=ti.f32, shape=(n, n))

@ti.func
def complex_sqr(z):
  return ti.Vector([z[0] * z[0] - z[1] * z[1], z[1] * z[0] * 2])

#matrix cross
@ti.func
def cross(a, b):
  x = a[1]*b[2] - a[2]*b[1]
  y = a[2]*b[0] - a[0]*b[2]
  z = a[0]*b[1] - a[1]*b[0]
  return ti.Vector([x, y, z])

@ti.func
def length(a):
  x = (a[0]*a[0])
  y = (a[1]*a[1])
  z= (a[2]*a[2])
  return ti.sqrt(x + y + z)

@ti.func
def DistLine(ro, rd, p):
  c =  cross(p-ro, rd)
  l_c = length(c)
  l_rd = length(rd)
  return l_c/l_rd

@ti.func
def xyz(a):
  return ti.Vector([ a[0], a[1], a[2] ])

# @ti.func
# def min(a, b):
#   if a<b:
#     return a
#   else:
#     return b

@ti.func
def GetDist(p, t, part_x, part_y, part_z):
  s = ti.Vector([ 0.0, 1.0, 6.0, 5.0**0.5 ])
  dist = p-xyz(s)
  sphereDist = length(dist) - s[3]
  planeDist = p[1]
  
  particle = ti.Vector([ part_x, part_y, part_z, 0.1 ])
  p_dist = p-xyz(particle)
  particleDist = length(p_dist) - particle[3]
  
  d = min(planeDist, sphereDist, particleDist)
  return d

@ti.func   
def RayMarch(ro, rd, t, part_x, part_y, part_z):
  dO = 0.0
  i = 0 
  while i < MAX_STEPS:
    p = ro + rd*dO
    dS = GetDist(p, t, part_x, part_y, part_z)
    dO += dS
    if dO > MAX_DIST or dS < SURF_DIST:
      break
    i = i + 1
  return dO

@ti.func   
def normalize(p):
  return ti.Vector([ p[0]/length(p), p[1]/length(p), p[2]/length(p)])


@ti.func   
def GetNormal(p, t, part_x, part_y, part_z):
  d = GetDist(p, t, part_x, part_y, part_z)
  e1 = ti.Vector([0.01, 0.0, 0.0])
  e2 = ti.Vector([0.0, 0.01, 0.0])
  e3 = ti.Vector([0.0, 0.0, 0.01])
  x = GetDist(p-e1, t, part_x, part_y, part_z)
  y = GetDist(p-e2, t, part_x, part_y, part_z)
  z = GetDist(p-e3, t, part_x, part_y, part_z)
  n = ti.Vector([d-x,d-y,d-z])
  return normalize(n)

@ti.func   
def clamp(p):

  if p<0:
    p = 0
  return p



# @ti.func   
# def dot(a,b):

@ti.func   
def GetLight(p, t, part_x, part_y, part_z):
  lightPos = ti.Vector([ 0.0 + ti.sin(t), 5.0, 6.0 + ti.cos(t) ])

  l = normalize(lightPos - p)
  n = GetNormal(p, t, part_x, part_y, part_z)
  diff = clamp(ti.dot(n,l))
  d = RayMarch(p + n*SURF_DIST*2.0, l, t, part_x, part_y, part_z)
  if (d < length(lightPos - p)):
    diff = diff* 0.1

  return diff


# @ti.func
# def convertToMat(np_x):
#   # for i, j in pixels:
#   #   pixels[i, j] = ti.Vector([np_x[i,0], np_x[i,1], np_x[i,2],1.0])
#   return 0

@ti.kernel
def paint(t: ti.f32, part_x: ti.f32, part_y: ti.f32, part_z: ti.f32, np_x: ti.ext_arr()):
  
  # particles = ti.Vector(3, dt=ti.f32, shape=size)
  # for a, b in particles:
  #   uv = ti.Vector([ ((a/640) )*(2) , (b/320) ])
    # particles[a] = ti.Vector([1.0,1.0,1.0])
    # particles[None] = ti.Vector([np_x[a,0], np_x[a,1], np_x[a,2]])
  # test = convertToMat(np_x)

  

  for i, j in pixels: # Parallized over all pixels
    # c = ti.Vector([-0.8, ti.sin(t) * 0.2])
    # z = ti.Vector([float(i) / n - 1, float(j) / n - 0.5]) * 2
    # iterations = 0 
    # while z.norm() < 20 and iterations < 50:
    #   z = complex_sqr(z) + c
    #   iterations += 1

    #####cirle in 3D space example####

    uv = ti.Vector([ ((i/640) )*(2) , (j/320) ])

    ro = ti.Vector([ 0.0, 1.0, 0.0 ])
    rd = ti.Vector([ uv[0], uv[1], 1.0 ]) 
    
    # particles = ti.Vector(ti.f32)
    # particles.from_numpy(np_x)
    d = RayMarch(ro, rd, t, part_x, part_y, part_z)
    # test = np_x[20,0]
    p = ro +rd*d
    diff = GetLight(p, t, part_x, part_y, part_z)
    # d = d/6.0

    # pixels[i, j] = ti.Vector([diff[0],diff[1],diff[2],1.0]) 
    pixels[i, j] = ti.Vector([diff,diff,diff,1.0]) 



gui = ti.GUI("Fractl", (n , n))

for frame in range(1000000):

  mpm.step(4e-3)
  # colors = np.array([0x068587, 0xED553B, 0xEEEEF0], dtype=np.uint32)
  np_x, np_v, np_material = mpm.particle_info()
  part_x =  np_x.item((20,0))
  part_y = np_x.item((20,1))
  part_z = np_x.item((20,2))
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
  paint(frame * 0.03, part_x, part_y, part_z, np_x)
  gui.set_image(pixels)
  # gui.circle([0 / 10, 1 / 10], radius=20, color=0xFF0000)
  # gui.circles(screen_pos, radius=1.5, color=colors[np_material])
  gui.show()