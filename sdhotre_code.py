#!/usr/bin/env python
# coding: utf-8


import cv2
import numpy as np
import pandas as pd

img = cv2.imread('sample3.jpeg')



# vanishing_points

df = pd.read_csv('coordinates.csv');
data = np.array(df)

P1 = data[0]
P2 = data[1]
P3 = data[2]
P4 = data[3]
P5 = data[4]
P6 = data[5]
P7 = data[6]

endpoints = {
    "x" : [(P1, P2), (P3, P4), (P5, P6)],
    "y" : [(P1, P7), (P3, P5), (P4, P6)],
    "z" : [(P1, P3), (P2, P4), (P7, P5)],
}

reference_point = {
    "x" : list(P2) + [1],
    "y" : list(P7) + [1],
    "z" : list(P3) + [1],
    "o" : list(P1) + [1]
}

vanishing_points = {}
for key in endpoints.keys():
    lines = []
    for ep in endpoints[key]:
        e1, e2 = ep
        # Homogeneous coordinates
        e1 = list(e1) + [1]
        e2 = list(e2) + [1]
        lines.append(np.cross(e1, e2))
        
    if len(lines) == 2:
        
        vanishing = np.cross(lines[0], lines[1])
        vanishing_points[key] = vanishing / vanishing[-1]
        
    else:
        M = np.zeros((3, 3), dtype='float64')
        for j in range(3):
            a, b, c = lines[j]
            M += np.array([[a * a, a * b, a * c], [a * b, b * b, b * c], [a * c, b * c, c * c]])
        # Compute vanishing points
        eig_values, eig_vectors = np.linalg.eig(M)
        vanishing = eig_vectors[:, np.argmin(eig_values)]
        vanishing = vanishing / vanishing[-1]
        vanishing_points[key] = vanishing



#projection matrix, refernce length and homography matrix
reference_length = {}
scaling_factor = {}
projection_matrix = np.zeros((3, 4), dtype='float64')  

for key in vanishing_points.keys():
    reference_length[key] = np.sqrt(np.sum(np.square(np.array(reference_point[key]) - np.array(reference_point["o"]))))
    A = (np.array(vanishing_points[key])-np.array(reference_point[key]))
    B = (np.array(reference_point[key] )-np.array(reference_point['o']))
    a,resid,rank,s = np.linalg.lstsq(A.reshape(-1,1) , B.reshape(-1,1))  
    scaling_factor[key] = a[0][0]/reference_length[key]
    projection_matrix[:, list(vanishing_points).index(key)] = scaling_factor[key]*vanishing_points[key]
    
projection_matrix[:, 3] = reference_point['o']
# linalg.matrix_rank(projection_matrix)    

Hxy = projection_matrix[:, [0,1,3]]
Hyz = projection_matrix[:, [1,2,3]]
Hxz = projection_matrix[:, [0,2,3]]

print(scaling_factor)    
print(projection_matrix) 
    



#texture maps

row, columns = img.shape[:2]
frame_xy = cv2.warpPerspective(img, Hxy, (row, columns), flags=cv2.WARP_INVERSE_MAP)
cv2.imwrite('xy.png', frame_xy)

frame_xz = cv2.warpPerspective(img, Hxz, (row, columns), flags=cv2.WARP_INVERSE_MAP)
cv2.imwrite('xz.png', frame_xz)

frame_yz = cv2.warpPerspective(img, Hyz, (row, columns), flags=cv2.WARP_INVERSE_MAP)
cv2.imwrite('yz.png', frame_yz)

