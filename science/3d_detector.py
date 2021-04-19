import open3d as o3d
import numpy as np
import os
import sys
import math
import random
from time import sleep

class cube:
    def __init__(self, x, y, z, w, l, h, thetaX, thetaY, thetaZ):
        self.x = x
        self.y = y
        self.z = z
        self.w = w
        self.l = l
        self.h = h
        thresh = 0.15
        self.thetaX = thetaX
        self.thetaY = thetaY
        self.thetaZ = thetaZ
         
        self.centerx = x + w/2
        self.centery = y + l/2
        self.centerz = z + h/2
        self.pts_total = 0
        self.inliers_total =0
        ptA = [x, y, z]
        ptB = [x+self.w, y, z]
        ptC = [x, y+self.l, z]
        ptD = [x+self.w, y+self.l, z]
        ptE = [x, y, z+self.h]
        ptF = [x+self.w, y, z+self.h]
        ptG = [x, y+self.l, z+self.h]
        ptH = [x+self.w, y+self.l, z+self.h]
    
        pointsInit = [
            ptA,
            ptB,
            ptC,
            ptD,
            ptE,
            ptF,
            ptG,
            ptH
        ]
        ptA1 = [x-thresh, y-thresh, z-thresh]
        ptB1 = [x+self.w+thresh, y-thresh, z-thresh]
        ptC1 = [x-thresh, y+self.l+thresh, z-thresh]
        ptD1 = [x+self.w+thresh, y+self.l+thresh, z-thresh]
        ptE1 = [x-thresh, y-thresh, z+self.h+thresh]
        ptF1 = [x+self.w+thresh, y-thresh, z+self.h+thresh]
        ptG1 = [x-thresh, y+self.l+thresh, z+self.h+thresh]
        ptH1 = [x+self.w+thresh, y+self.l+thresh, z+self.h+thresh]
    
        pointsInit1 = [
            ptA1,
            ptB1,
            ptC1,
            ptD1,
            ptE1,
            ptF1,
            ptG1,
            ptH1
        ]
        lines = [
            [0, 1],
            [0, 2],
            [1, 3],
            [2, 3],
            [4, 5],
            [4, 6],
            [5, 7],
            [6, 7],
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],
        ]
        lines1 = [
            [0, 1],
            [0, 2],
            [1, 3],
            [2, 3],
            [4, 5],
            [4, 6],
            [5, 7],
            [6, 7],
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],
        ]
        
        colors = [[1, 0, 0] for i in range(len(lines))]
        colors1 = [[0, 1, 0] for i in range(len(lines1))]
        
        self.line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(pointsInit),
            lines=o3d.utility.Vector2iVector(lines),
        )
        self.line_set1 = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(pointsInit1),
            lines=o3d.utility.Vector2iVector(lines1),
        )
        self.line_set.colors = o3d.utility.Vector3dVector(colors)
        self.line_set1.colors = o3d.utility.Vector3dVector(colors1)
        rotation=self.line_set.get_rotation_matrix_from_xyz((math.radians(self.thetaX),math.radians(self.thetaY),math.radians(self.thetaZ)))
        self.line_set.rotate(rotation, (self.centerx,self.centery,self.centerz))
        self.line_set1.rotate(rotation, (self.centerx,self.centery,self.centerz))
        
    def check_pts(self, points):
        pts_tfd = np.asarray(self.line_set.points)
        ptA, ptB, ptC, ptD, ptE, ptF, ptG, ptH = pts_tfd
        pts_tfd1 = np.asarray(self.line_set1.points)
        ptA1, ptB1, ptC1, ptD1, ptE1, ptF1, ptG1, ptH1 = pts_tfd1
        
        dir1 = (ptB-ptA)
        size1 = np.linalg.norm(dir1)
        dir1 = dir1 / size1
        
        dir2 = (ptC-ptA)
        size2 = np.linalg.norm(dir2)
        dir2 = dir2 / size2

        dir3 = (ptE-ptA)
        size3 = np.linalg.norm(dir3)
        dir3 = dir3 / size3
        
        dir11 = (ptB1-ptA1)
        size11 = np.linalg.norm(dir11)
        dir11 = dir11 / size11
        
        dir21 = (ptC1-ptA1)
        size21 = np.linalg.norm(dir21)
        dir21 = dir21 / size21

        dir31 = (ptE1-ptA1)
        size31 = np.linalg.norm(dir31)
        dir31 = dir31 / size31

        cube_center = np.array([self.centerx, self.centery, self.centerz]).reshape(1,3)
        self.pts_total = points.shape[0]
        pts_indices = np.arange(points.shape[0])
        indices = pts_indices.tolist()
        dir_vec =points - cube_center 
        res1 = np.where( (np.absolute(np.dot(dir_vec, dir1)) * 2) >= size1 )[0]
        res2 = np.where( (np.absolute(np.dot(dir_vec, dir2)) * 2) >= size2 )[0]
        res3 = np.where( (np.absolute(np.dot(dir_vec, dir3)) * 2) >= size3 )[0]
        
        res11 = np.where( (np.absolute(np.dot(dir_vec, dir11)) * 2) >= size11 )[0]
        res21 = np.where( (np.absolute(np.dot(dir_vec, dir21)) * 2) >= size21 )[0]
        res31 = np.where( (np.absolute(np.dot(dir_vec, dir31)) * 2) >= size31 )[0]

        fits =   list( set().union(res1, res2, res3) )
        fits1 = set(indices)- set(list( set().union(res11, res21, res31) ))
        self.inliers_total = len(np.intersect1d(fits,list(fits1)))
                
        return self.inliers_total

class GA:
    def __init__(self):
        self.population=[]
        self.fitness =[]
        
    def init_popultation(self, size):
        self.size = size
        for _ in range(self.size):
            rand_x = random.uniform(-3, 3)
            rand_y = random.uniform(-3, 3)
            rand_z = random.uniform(-3, 3)
            rand_w = random.uniform(0.2, 4)
            rand_l = random.uniform(0.2, 4)
            rand_h = random.uniform(0.2, 4)
            rand_thetaX = random.uniform(0, 360)
            rand_thetaY = random.uniform(0, 360)
            rand_thetaZ = random.uniform(0, 360)
            self.population.append(cube(rand_x,rand_y,rand_z,rand_w,rand_l,rand_h,rand_thetaX,rand_thetaY,rand_thetaZ))
            
    def calc_fitness(self):
        self.fitness =[]
        for i in range(self.size):  
            self.fitness.append( 100*(self.population[i].inliers_total/(self.population[i].pts_total-self.population[i].inliers_total+1)))   
        x = zip(self.fitness,self.population)
        xs = sorted(x,reverse = True, key=lambda tup: tup[0])
        self.fitness = [x[0] for x in xs]
        self.population = [x[1] for x in xs]
        
    def selection(self):
        self.population = self.population[:int(len(self.population)/2)]
        
    def crossover(self, r1, r2):
        a = np.random.random(1)
        b = 1 - a
        
        x_new_1 = a * r1.x + b * r2.x
        y_new_1 = a * r1.y + b * r2.y
        z_new_1 = a * r1.y + b * r2.y
        w_new_1 = a * r1.w + b * r2.w
        l_new_1 = a * r1.l + b * r2.l
        h_new_1 = a * r1.h + b * r2.h
        thetaX_new_1 = a * r1.thetaX + b * r2.thetaX
        thetaY_new_1 = a * r1.thetaY + b * r2.thetaY
        thetaZ_new_1 = a * r1.thetaZ + b * r2.thetaZ
        self.cube_new_1 = cube(x_new_1,y_new_1,y_new_1,w_new_1,l_new_1,h_new_1,thetaX_new_1,thetaY_new_1,thetaZ_new_1)
        a = 1 - a
        b = 1 - b
        x_new_2 = a * r1.x + b * r2.x
        y_new_2 = a * r1.y + b * r2.y
        z_new_2 = a * r1.z + b * r2.z
        w_new_2 = a * r1.w + b * r2.w
        l_new_2 = a * r1.l + b * r2.l
        h_new_2 = a * r1.h + b * r2.h
        thetaX_new_2 = a * r1.thetaX + b * r2.thetaX
        thetaY_new_2 = a * r1.thetaY + b * r2.thetaY
        thetaZ_new_2 = a * r1.thetaZ + b * r2.thetaZ
        self.cube_new_2 = cube(x_new_2,y_new_2,y_new_2,w_new_2,l_new_2,h_new_2,thetaX_new_2,thetaY_new_2,thetaZ_new_2)
        
        return self.cube_new_1, self.cube_new_2
        
    def repopulate(self):
        self.offspring = []
        for _ in range(len(self.population)):
            index_1 = np.random.randint(0,len(self.population)-1,1)
            index_2 = np.random.randint(index_1,len(self.population)-1,1)
            child_1, child_2 = self.crossover(self.population[index_1[0]],self.population[index_2[0]]) 
            self.offspring.append(child_1)
        self.population = self.population + self.offspring
        
    def mutate(self, cube):
        cube.x += int(0.9*(1-1.1*np.random.random(1)[0]))
        cube.y += int(0.9*(1-1.1*np.random.random(1)[0]))
        cube.z += int(0.9*(1-1.1*np.random.random(1)[0]))
        cube.w += int(0.9*(1-1.1*np.random.random(1)[0]))
        cube.l += int(0.9*(1-1.1*np.random.random(1)[0]))
        cube.h += int(0.9*(1-1.1*np.random.random(1)[0]))
        cube.thetaX += 0.9*(1-1.1*np.random.random(1)[0])
        cube.thetaY += 0.9*(1-1.1*np.random.random(1)[0])
        cube.thetaZ += 0.9*(1-1.1*np.random.random(1)[0])
        return cube
    
    def mutate_population(self):
        for i in range(len(self.population)):
            self.population[i] = self.mutate(self.population[i])

if __name__ == '__main__':
    vis = o3d.visualization.Visualizer()
    number_steps = 70
    best_fit = 0
    best_model = None
    algorithm = GA()
    algorithm.init_popultation(70)
    vis.create_window(width = 800, height = 600)
    pcd = o3d.io.read_point_cloud("models/cube_test1.ply")
    vis.add_geometry(pcd)
    geometry_list = list()
        
    for _ in range(number_steps):
        #random.shuffle(algorithm.population)
        vis.add_geometry(pcd)
        for i in range(len(algorithm.population)):
            #vis.add_geometry(algorithm.population[i].line_set)
            #vis.add_geometry(algorithm.population[i].line_set1)
            inliers = algorithm.population[i].check_pts(np.asarray(pcd.points))
        algorithm.calc_fitness()
        if algorithm.fitness[0]>best_fit:
            best_fit = algorithm.fitness[0]
            print('Found better fintess: ',best_fit, 'at', _ )
            best_model = algorithm.population[0]
       
        algorithm.selection()
        algorithm.repopulate()
        algorithm.mutate_population()
        #vis.poll_events()
        #vis.update_renderer()
        #vis.clear_geometries()
    
    
    vis.add_geometry(pcd)
    vis.add_geometry(best_model.line_set)
    vis.add_geometry(best_model.line_set1)
    vis.run()
    vis.destroy_window()