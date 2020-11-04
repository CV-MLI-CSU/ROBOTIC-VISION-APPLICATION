# Author: Sisi(Scarlett) Liu <scarlett.liu@csu.edu.cn>
# Uni: Central South University, Changsha, China
# Online course: Robotic Vision and Applications
# Created: 09-Sep-2020   Modified: 09-Sep-2020   Revision: 1.0
# Copyright � 2020 SR2V Ltd. All rights reserved

# Lecture 4-4-2 Region Properties

# load library
import cv2
import numpy as np

# load an image
img = cv2.imread('images/rocks.jpg')
img = cv2.resize(img,None,fx=0.3,fy=0.3) # resize the image by 70%
cv2.imshow('img',img)
cv2.waitKey(0)
imgray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(imgray,170,255,0)
cv2.imshow('thresh',thresh)
cv2.waitKey(0)

# morphological operation / close operation
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5)) # Rectangular / Square Kernel
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
cv2.imshow('thresh2',thresh)
cv2.waitKey(0)
contours,hierarchy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
img1 = img.copy()
img_draw = cv2.drawContours(img1,contours,-1,(0,255,0),5)
cv2.imshow('contours2',img_draw)
cv2.waitKey(0)

# filtering out the largest area
areas = []
for i in range(0, len(contours)):
    areas.append(cv2.contourArea(contours[i]))
print(areas)
max_value = max(areas)
max_index = areas.index(max_value)
print(max_value,max_index)

# Contour Features (轮廓特征)
# 1. Moments
cnt = contours[max_index]
M = cv2.moments(cnt)
# From this moments, you can extract useful data like area, centroid etc.
# Centroid is given by the relations, C_x = \frac{M_{10}}{M_{00}} and
# C_y = \frac{M_{01}}{M_{00}}. This can be done as follows:
cx = int(M['m10']/M['m00']) #计算轮廓重心
cy = int(M['m01']/M['m00'])
img2 = img.copy()
# keep thickness as a negative number for filled circle
img_draw2 = cv2.circle(img2, (cx,cy), radius=5, color=(0, 0, 255), thickness=-1)
cv2.imshow('center',img_draw2)
cv2.waitKey(0)

# 2. Contour Area （计算轮廓面积）
# Contour area is given by the function cv2.contourArea() or from moments, M[‘m00’].
area = cv2.contourArea(cnt)

# 3. Contour Perimeter （计算轮廓周长）
perimeter = cv2.arcLength(cnt,True)

# 4. Contour Approximation （轮廓多边形逼近）
# 4.1进行多边形逼近，得到多边形的角点
epsilon = 0.01*cv2.arcLength(cnt,True) # you can change o.1 to 0.01
approx = cv2.approxPolyDP(cnt,epsilon,True)
print(approx)
# 4.2画出多边形
img3 = img.copy()
img_draw3 = cv2.polylines(img3,[approx],True,(0,0,255),2)
cv2.imshow('Contour Approximation',img_draw3)
cv2.waitKey(0)

# 5. Convex Hull (凸包)
hull = cv2.convexHull(cnt)
img4 = img.copy()
img_draw4 = cv2.polylines(img4,[hull],True,(0,0,255),2)
cv2.imshow('hull',img_draw4)
cv2.waitKey(0)

# 6. Checking Convexity. It just return whether True or False
# 检查凸度, 检查曲线是否凸起, 返回值为True或False.
k = cv2.isContourConvex(cnt)
print(k)

# 7. Bounding Rectangle. There are two types of bounding rectangles.
# 7.a Straight Bounding Rectangle 边界矩形
x,y,w,h = cv2.boundingRect(cnt)
img5 = img.copy()
img_draw5 = cv2.rectangle(img5,(x,y),(x+w,y+h),(0,255,0),2)
cv2.imshow('Bounding Rectangle',img_draw5)
cv2.waitKey(0)
# 7.b. Rotated Rectangle 旋转边界矩形
rect = cv2.minAreaRect(cnt)
box = cv2.boxPoints(rect)
box = np.int0(box)
img_draw5 = cv2.drawContours(img5,[box],0,(0,0,255),2)
cv2.imshow('Bounding Rectangle',img_draw5)
cv2.waitKey(0)

# 8. Minimum Enclosing Circle 最小封闭圈
(x,y),radius = cv2.minEnclosingCircle(cnt)
center = (int(x),int(y))
radius = int(radius)
img6 = img.copy()
img_draw6 = cv2.circle(img6,center,radius,(0,255,0),2)
cv2.imshow('img_draw6',img_draw6)
cv2.waitKey(0)

# 9. Fitting an Ellipse 拟合椭圆
ellipse = cv2.fitEllipse(cnt)
img_draw6 = cv2.ellipse(img6,ellipse,(0,255,0),2)
cv2.imshow('img_draw6',img_draw6)
cv2.waitKey(0)

# 10. Fitting a Line 拟合线条
rows,cols = img.shape[:2]
[vx,vy,x,y] = cv2.fitLine(cnt, cv2.DIST_L2,0,0.01,0.01)
lefty = int((-x*vy/vx) + y)
righty = int(((cols-x)*vy/vx)+y)
img7 = img.copy()
img_draw7 = cv2.line(img7,(cols-1,righty),(0,lefty),(0,255,0),2)
cv2.imshow('Fitting a Line',img_draw7)
cv2.waitKey(0)

cv2.destroyAllWindows()

# ContoursProperties（轮廓属性）
# Solidity、Equivalent Diameter, Mask image,Mean Intensity ....
# 1. 纵横比(Aspect Ratio)
x,y,w,h = cv2.boundingRect(cnt)
aspect_ratio = float(w)/h
print(aspect_ratio)

# 2. 范围(Extent)
area = cv2.contourArea(cnt)
x,y,w,h = cv2.boundingRect(cnt)
rect_area = w*h
extent = float(area)/rect_area
print(extent)

# 3. 密实度(Solidity)
area = cv2.contourArea(cnt)
hull = cv2.convexHull(cnt)
hull_area = cv2.contourArea(hull)
solidity = float(area)/hull_area
print(solidity)

# 4. 等效直径(EquivalentDiameter)
area = cv2.contourArea(cnt)
equi_diameter = np.sqrt(4*area/np.pi)
print(equi_diameter)

# 5. 方向(Orientation)
(x,y),(MA,ma),angle = cv2.fitEllipse(cnt)
print((x,y),(MA,ma),angle)

# 6.蒙版和像素点(Maskand Pixel Points)
mask = np.zeros(imgray.shape,np.uint8)
cv2.drawContours(mask,[cnt],0,255,-1)
pixelpoints = np.transpose(np.nonzero(mask)) # Numpy以(行，列)格式给出坐标
#pixelpoints = cv2.findNonZero(mask) # OpenCV以(x,y)格式给出坐标, row=x和column=y

# 7. 最大最小值和它们的位置(maximumvalue,minimum value and their locations)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(imgray,mask = mask)
print(min_val, max_val, min_loc, max_loc)

# 8. 平均颜色或平均强度(meanColor or mean Intensity)
mean_val = cv2.mean(img,mask = mask)
print(mean_val)

# 9. 极值点(ExtremePoints)
leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])
# draw four Extreme Points
img8 = img.copy()
img_draw8 = cv2.circle(img8, leftmost, radius=5, color=(0, 0, 255), thickness=-1)
img_draw8 = cv2.circle(img8, rightmost, radius=5, color=(0, 0, 255), thickness=-1)
img_draw8 = cv2.circle(img8, topmost, radius=5, color=(0, 0, 255), thickness=-1)
img_draw8 = cv2.circle(img8, bottommost, radius=5, color=(0, 0, 255), thickness=-1)
cv2.imshow('Extreme Points',img_draw8)
cv2.waitKey(0)
