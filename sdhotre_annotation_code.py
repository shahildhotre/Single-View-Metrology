#!/usr/bin/env python
# coding: utf-8


import cv2
import numpy as np
import pandas as pd

img = cv2.imread('sample3.jpeg')
 

# Annotation

font = cv2.FONT_HERSHEY_SIMPLEX

P1 = (927, 1329)
P2 = (1329, 900)
P3 = (907, 769)
P4 = (1393, 395)
P5 = (157, 404)
P6 = (672, 186)
P7 = (253, 876)


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

###  X line  ###
cv2.line(img,(P1[0],P1[1]),(P2[0],P2[1]),(255,0,0),4)        #blue
cv2.line(img,(P3[0],P3[1]),(P4[0],P4[1]),(255,0,0),4)
cv2.line(img,(P5[0],P5[1]),(P6[0],P6[1]),(255,0,0),4)

###  Y line ###
cv2.line(img,(P1[0],P1[1]),(P7[0],P7[1]),(0,255,0),4)      #green
cv2.line(img,(P3[0],P3[1]),(P5[0],P5[1]),(0,255,0),4)
cv2.line(img,(P4[0],P4[1]),(P6[0],P6[1]),(0,255,0),4)


###  Z line ###
cv2.line(img,(P1[0],P1[1]),(P3[0],P3[1]),(0,0,255),4)     #red
cv2.line(img,(P7[0],P7[1]),(P5[0],P5[1]),(0,0,255),4)
cv2.line(img,(P2[0],P2[1]),(P4[0],P4[1]),(0,0,255),4)

#labelling points
cv2.putText(img, 'P1', P1, font, 1, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(img, 'P2', P2, font, 1, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(img, 'P3', P3, font, 1, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(img, 'P4', P4, font, 1, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(img, 'P5', P5, font, 1, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(img, 'P6', P6, font, 1, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(img, 'P7', P7, font, 1, (255, 255, 255), 2, cv2.LINE_AA)

#labelling axis and reference point
cv2.putText(img, 'O', (P1[0], P1[1] + 40 ), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(img, 'X', (P2[0], P2[1] + 40 ), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(img, 'Y', (P7[0], P7[1] + 40 ), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(img, 'Z', (P3[0], P3[1] + 40 ), font, 1, (255, 255, 255), 2, cv2.LINE_AA)



data = np.zeros((7,2))
data[0,:] = P1
data[1,:] = P2
data[2,:] = P3
data[3,:] = P4
data[4,:] = P5
data[5,:] = P6
data[6,:] = P7

data = np.array(data)

df = pd.DataFrame({"X" : data[:,0], "Y" : data[:,1]})
df.to_csv("sdhotre_coordinates.csv", index=False)

cv2.imshow('Annotation', img)
cv2.imwrite('Annotation_labelling.jpeg', img)
cv2.waitKey(0)


