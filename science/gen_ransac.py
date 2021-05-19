import open3d as o3d
import numpy as np
from numpy.random import default_rng
from matplotlib import pyplot as plt
import os
import cv2
import sys
import math
import random
from time import sleep
import matplotlib.pyplot as plt


def find_hull(points):
    
    inliers_pcd = o3d.geometry.PointCloud()
    inliers_pcd.points = o3d.utility.Vector3dVector(points)
    #print(type(inliers_pcd))
    hull, _ = inliers_pcd.compute_convex_hull()
    hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
    color = np.random.rand(3).tolist()
    hull_ls.paint_uniform_color(color)
    return hull_ls

def color_pts(points):
    inliers_pcd = o3d.geometry.PointCloud()
    inliers_pcd.points = o3d.utility.Vector3dVector(points)
    color = np.random.rand(3)
    #color = np.array([0,0,0])
    point_colors = np.tile(color, (points.shape[0],1))
    inliers_pcd.colors = o3d.utility.Vector3dVector(point_colors)
    return inliers_pcd

class plane:
    def __init__(self, M0, M1, M2):
        A0 = np.linalg.det(np.array([[M1[1]-M0[1],M2[1]-M0[1]],[M1[2]-M0[2],M2[2]-M0[2]]]))
        A1 = np.linalg.det(np.array([[M1[0]-M0[0],M2[0]-M0[0]],[M1[2]-M0[2],M2[2]-M0[2]]]))
        A2 = np.linalg.det(np.array([[M1[0]-M0[0],M2[0]-M0[0]],[M1[1]-M0[1],M2[1]-M0[1]]]))
        self.A = A0
        self.B = -A1
        self.C = A2
        self.D = -M0[0]*A0+M0[1]*A1-M0[2]*A2
    def calc_dist(self, pcd):
        R = np.absolute(self.A*pcd[:,0]+self.B*pcd[:,1]+self.C*pcd[:,2]+self.D)/(np.sqrt(self.A**2+self.B**2+self.C**2)+ sys.float_info.epsilon)
        #print(type(R))
        return R

def find_pts_plane(pcd):
    indexA = np.random.randint(0,pcd.shape[0])
    A = pcd[indexA]
    distancesA = np.sqrt((pcd[:,0] - A[0])**2+(pcd[:,1] - A[1])**2+ (pcd[:,2] - A[2])**2)
    #Calculating B probabilities
    roundedA = np.around(distancesA,1)
    uniqueB, countsB = np.unique(roundedA, return_counts = True)
    probabilitiesB = countsB/distancesA.shape[0]
    plt.plot(uniqueB, probabilitiesB)
    plt.show()
    choose_thresholded_value_B = np.random.choice(uniqueB, p=probabilitiesB)
    B_candidate_indices = np.argwhere(roundedA==choose_thresholded_value_B)
    B_candidate_indices = B_candidate_indices.reshape(B_candidate_indices.shape[0])
    indexB = np.random.choice(B_candidate_indices)
    B = pcd[indexB]
    pcd = np.delete(pcd, [indexA,indexB],0) 
    #Calculating C probabilities
    mid_AB = np.array([(A[0]+B[0])/2,(A[1]+B[1])/2,(A[2]+B[2])/2])
    distances_mid_AB = np.sqrt((pcd[:,0] - mid_AB[0])**2+(pcd[:,1] - mid_AB[1])**2+ (pcd[:,2] - mid_AB[2])**2)
    roundedAB = np.around(distances_mid_AB,1)
    uniqueC, countsC = np.unique(roundedAB, return_counts = True)
    probabilitiesC = countsC/distances_mid_AB.shape[0]
    
    #plt.plot(uniqueC, probabilitiesC)
    #plt.show()
    choose_thresholded_value_C = np.random.choice(uniqueC, p=probabilitiesC)
    C_candidate_indices = np.argwhere(roundedAB==choose_thresholded_value_C)
    C_candidate_indices = C_candidate_indices.reshape(C_candidate_indices.shape[0])
    indexC = np.random.choice(C_candidate_indices)
    C = pcd[indexC]
    
    
    return A, B, C
   

def find_plane(pcd, iterations, threshold):
    best_fit = 0
    best_inliers_pts = None
    
    inliers = None
    best_model = list()
    for i in range(iterations):
        A, B, C = find_pts_plane(np.asarray(pcd.points))
        p = plane(A,B,C)
        R = p.calc_dist(np.asarray(pcd.points))
        inliers = np.where(R<=threshold)[0]
        inliers_len = inliers.shape[0] +3
        inliers_pts = np.take(np.asarray(pcd.points),inliers,axis=0)
        outliers = np.asarray(pcd.points).shape[0] - inliers_len
        fit = inliers_len/outliers
        if fit>best_fit:
            best_fit = fit
            print('Found better fintess: ',best_fit, 'at', i )
            best_model = [A,B,C]
            best_inliers_pts = inliers_pts
            best_inlier_indices = inliers
    
    points = np.array(best_model)
    return best_model, best_inliers_pts, best_inlier_indices

class sphere:
    def __init__(self, M1, M2, N1, N2):
        M1_2 = M1 + N1
        M2_2 = M2 + N2


pcd = o3d.io.read_point_cloud("models/cube_test1.ply")
#pcd = o3d.io.read_point_cloud("models/fragment.ply") 
#pcd = o3d.io.read_point_cloud("aivisum.ply")

#Read rgb-d
'''
color_raw = o3d.io.read_image("models/apple_1_1_12.png")
depth_raw = o3d.io.read_image("models/apple_1_1_12_depth.png")
rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
    color_raw, depth_raw)
print(rgbd_image)
plt.subplot(1, 2, 1)
plt.title('Redwood grayscale image')
plt.imshow(rgbd_image.color)
plt.subplot(1, 2, 2)
plt.title('Redwood depth image')
plt.imshow(rgbd_image.depth)
plt.show()
pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
    rgbd_image,
    o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
# Flip it, otherwise the pointcloud will be upside down
'''
#pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
vis = o3d.visualization.Visualizer()
vis.create_window(width = 800, height = 600)
vis.add_geometry(pcd)
print(pcd)
models_list = []
inliers_list = []
for i in range(1):
    model, inliers, inliers_indices = find_plane(pcd,1,0.09)
    if inliers.shape[0]>15:
        models_list.append(model)
        inliers_list.append(inliers)
        pts = color_pts(inliers)
        vis.add_geometry(pts)
        pcd.points = o3d.utility.Vector3dVector(np.delete(np.asarray(pcd.points), inliers_indices,0))
        print(pcd)
color = np.array([0,0,0])
point_colors = np.tile(color, (np.asarray(pcd.points).shape[0],1))
colors = o3d.utility.Vector3dVector(point_colors)
pcd.colors = o3d.utility.Vector3dVector(point_colors)
vis.run()
vis.destroy_window()