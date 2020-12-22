# -*- coding: utf-8 -*-
"""
# Author: Sisi(Scarlett) Liu <scarlett.liu@csu.edu.cn> & Zhangyang Li
# Uni: Central South University, Changsha, China
# Online course: Robotic Vision and Applications
# Created: 09-Sep-2020   Modified: 09-Sep-2020   Revision: 1.0
# Copyright � 2020 SR2V Ltd. All rights reserved

# Lecture 7-1 Vision Guided ABB

"""

import cv2
import numpy as np
import TIS
import socket
import time
import sys
import math
import glob


"""
--------------------camera calbration---------------------------------------
# global camera val
# Intrinsic for camera:  mtx1 mtx2
# distortion cofficients = (k_1,k_2,p_1,p_2,k_3) dist1 dist2
# rotation: rvecs1 rvecs2
# tranlation: tvecs1 tvecs2
"""
# termination criteria1
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
wnum=11
hnum=8
objp = np.zeros((wnum*hnum,3), np.float32)
objp[:,:2] = np.mgrid[0:wnum,0:hnum].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints1 = [] # 3d point in real world space
imgpoints1 = [] # 2d points in image plane.

images1 = glob.glob('*.bmp')

for fname in images1:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners1 = cv2.findChessboardCorners(gray, (wnum,hnum),None,cv2.CALIB_CB_ADAPTIVE_THRESH)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints1.append(objp)

        cv2.cornerSubPix(gray,corners1,(11,11),(-1,-1),criteria)
        imgpoints1.append(corners1)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, (11,8), corners,ret)
        cv2.imshow('img',img)
        cv2.waitKey(500)


#ret1, mtx1, dist1, rvecs1, tvecs1 = cv2.calibrateCamera(objpoints1, imgpoints1, gray.shape[::-1],None,None)

# termination criteria2
criteria2 = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((8*11,3), np.float32)
objp[:,:2] = np.mgrid[0:11,0:8].T.reshape(-1,2)


# Arrays to store object points and image points from all the images.
objpoints2 = [] # 3d point in real world space
imgpoints2 = [] # 2d points in image plane.

images2 = glob.glob('*.png')

for fname in images2:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners2 = cv2.findChessboardCorners(gray, (wnum,hnum),None,cv2.CALIB_CB_ADAPTIVE_THRESH)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints2.append(objp)

        cv2.cornerSubPix(gray,corners2,(11,11),(-1,-1),criteria2)
        imgpoints2.append(corners2)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, (11,8), corners,ret)
        cv2.imshow('img',img)
        cv2.waitKey(500)


ret2, mtx2, dist2, rvecs2, tvecs2 = cv2.calibrateCamera(objpoints2, imgpoints2, gray.shape[::-1],None,None)


img = cv2.imread('1.bmp')
h,  w = img.shape[:2]
newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
# undistort
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

# crop the image
x,y,w,h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite('calibresult.png',dst)
"""
--------------------------------------------------------------------------
"""

#link abb
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_address = ('192.168.125.4', 4005) # current server, and the port number is 10000
print >>sys.stderr, 'starting up on %s port %s' % server_address
sock.bind(server_address)
# Listen for incoming connections
sock.listen(1)
# Wait for a connection
print >> sys.stderr, 'waiting for a connection'
connection, client_address = sock.accept() # '192.168.125.1'
print >> sys.stderr, 'connection from', client_address
"""
--------------------------------------------------------------------------
"""
#isstart=input("if start this program?true or false")
isstart=True
while isstart == True:

    # Open camera, set video format, framerate and determine, whether the sink is color or bw
    # Parameters: Serialnumber, width, height, framerate (numerator only) , color
    # If color is False, then monochrome / bw format is in memory. If color is True, then RGB32
    # colorformat is in memory
    #打开相机，设置视频格式，帧率，并确定接收器是彩色还是黑白
    #参数:序号，宽度，高度，帧率(仅限分子)，颜色
    #如果颜色是假的，那么单色/ bw格式在内存中。如果颜色为真，则RGB32
    Tis1 = TIS.TIS("41814186", 1280, 960,30, True)
    #Tis2 = TIS.TIS("41814186", 1280, 960,30, True)
    # Start the pipeline so the camera streams 启动管道，使摄像机流
    Tis1.Start_pipeline()
    #Tis2.Start_pipeline()
    # Create an OpenCV output window for two cameras 创建一个OpenCV输出窗口
    cv2.namedWindow('Window1')
    #cv2.namedWindow('Window2')
    # Get the image. It is a numpy array  获取图像。它是一个numpy数组
    Tis1.Snap_image(1)  # Snap an image with one second timeout
    image1 = Tis1.Get_image()
     # Display the result
    cv2.imshow('Window1', image1)

    #image1=cv2.imread('textphoto.jpg')
    # 去畸变
    newcameramtx1, roi1=cv2.getOptimalNewCameraMatrix(mtx1,dist1,(w,h),0,(w,h)) # 自由比例参数
    dst1 = cv2.undistort(image1, mtx1, dist1, None, newcameramtx1)
    # 根据前面ROI区域裁剪图片
    x,y,w,h = roi1
    #dst1= dst1[y:y+h, x:x+w]
    cv2.imwrite('camera1.jpg',dst1)
    #image reconized
    path1='camera1.jpg'
    imgc1=cv2.imread(path1)
    # 缩小图像
    height1, width1 = imgc1.shape[:2]
    size = (int(width1*1), int(height1*1))
    imgc1= cv2.resize(imgc1, size, interpolation=cv2.INTER_AREA)
    gray=cv2.cvtColor(imgc1,cv2.COLOR_BGR2GRAY)#转灰度图
    hsv=cv2.cvtColor(imgc1,cv2.COLOR_BGR2HSV)#转hsv色彩空间
    #get black area
    thresh=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C ,cv2.THRESH_BINARY,11,10)
    #ret,thresh=cv2.threshold(gray,150,255,cv2.THRESH_BINARY) #阈值化处理，大于148的变为白色（0为黑色，255为白），输入灰度图
    mask=thresh#构建掩模
    black=cv2.bitwise_and(hsv,hsv,mask=mask)#hsv与掩模进行与运算，提取黑色区域
    #将黑色区域进行二值化处理
    black_gray=cv2.cvtColor(black,cv2.COLOR_HSV2BGR)#hsv转bgr
    black_gray=cv2.cvtColor(black_gray,cv2.COLOR_BGR2GRAY)#bgr转灰度图
    #binaryzation
    _, thresh=cv2.threshold(black_gray,10,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)#阈值化处理，大于10的变为白色
    img_morph=cv2.morphologyEx(thresh,cv2.MORPH_OPEN,(3,3))#开运算，去噪声
    img_morph=cv2.erode(img_morph,(3,3),img_morph,iterations=2)#腐蚀图像
    morph=cv2.dilate(img_morph,(3,3),img_morph,iterations=2)#膨胀图像
    cv2.imwrite('morph.jpg',morph)
    #获取中心区域轮廓及坐标
    img_cp=morph.copy()#复制原来的图像到一张新的图像上
    cnts,_=cv2.findContours(img_cp,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)#轮廓检测，输入二值图，cnts为返回的轮廓
    anum=len(cnts)
    bnum=anum
    cutnum=0
    center=np.zeros([anum,2],dtype=np.int32)
    for i in range(0,anum):
        cnt_second=sorted(cnts,key=cv2.contourArea,reverse=True)[i]#按照轮廓的面积从大到小进行排序，输出第a个轮廓
        box =cv2.minAreaRect(cnt_second)#最小矩形面积
        points=np.int0(cv2.boxPoints(box))#查找旋转矩形的 4 个顶点，points为四个顶点坐标
        mask=np.zeros(gray.shape,np.uint8)#生成灰度图同样大小的零矩阵，uint8是无符号八位整型,表示范围是[0, 255]的整数
        #进行轮廓的颜色填充，第一个参数是一张图片，可以是原图或者其他。第二个参数是轮廓，一个列表。第三个参数是对轮廓（第二个参数）的索引，当需要绘制独立轮廓时很有用，若要全部绘制可设为-1。接下来的参数是轮廓的颜色和厚度。
        mask=cv2.drawContours(mask,[points],-1,255,2)
        p1x,p1y=points[0,0],points[0,1]
        p3x,p3y=points[2,0],points[2,1]
        center_x,center_y=(p1x+p3x)/2,(p1y+p3y)/2#得到中心坐标
        cutsize=150
        if center_y<cutsize or center_y>height1-cutsize:
            cutnum=cutnum+1
            continue
        if center_x<cutsize or center_x>width1-cutsize:
            cutnum=cutnum+1
            continue
        center[i][0]=int(center_x)
        center[i][1]=int(center_y)
    centernew=np.zeros([anum-cutnum,2],dtype=np.int32)
    newnum=0
    for i in range(0,anum):
        if center[i][0]!=0:
            centernew[newnum][0]=center[i][0]
            centernew[newnum][1]=center[i][1]
            newnum=newnum+1
    anum=anum-cutnum
    center=centernew
    img2=cv2.imread(path1)
    for i in range(0,anum):
        b = (255,0,0)
        p=(center[i][0],center[i][1])
        img2=cv2.circle(img2,p,50,b,2)
    cv2.imwrite('trytextphoto.jpg',img2)
    sends=[]
    for i in range(0,anum):
        thingx1=center[i][0]
        thingy1=center[i][1]

        #point trans
        #输入世界坐标原点的像素坐标
        worldx0=659
        worldy0=527
        #图像中心至物体所在平面的距离(单位：mm)
        Zc=1420
        #相机内参
        fx1=mtx1[0][0]
        fy1=mtx1[1][1]
        #在相机坐标系下，目标中心坐标与原点的实际坐标差值
        dtwx=thingx1-worldx0
        dtwy=thingy1-worldy0
        dxc=Zc/fx1*abs(thingx1-worldx0)
        dyc=Zc/fy1*abs(thingy1-worldy0)
        #dxc=0.9*abs(thingx1-worldx0)
        #dyc=0.9*abs(thingy1-worldy0)
        #由相机坐标系转为世界坐标系
        numrvecs1=len(rvecs1)
        transx1=0
        transy1=0
        transz1=0
        for i in range(0,numrvecs1-1):
            transx1=transx1+rvecs1[i][0]
            transy1=transy1+rvecs1[i][1]
            transz1=transz1+rvecs1[i][2]
        transx1=transx1/numrvecs1
        transy1=transy1/numrvecs1
        transz1=transz1/numrvecs1
        transx1=math.radians(transx1)
        transy1=math.radians(transy1)
        transz1=math.radians(transz1)
        Rx1=np.array([[1,0,0],[0,math.cos(transx1),math.sin(transx1)],[0,-math.sin(transx1),math.cos(transx1)]])
        Ry1=np.array([[math.cos(transy1),0,math.sin(transy1)],[0,1,0],[-math.sin(transy1),0,math.cos(transy1)]])
        Rz1=np.array([[math.cos(transz1),math.sin(transz1),0],[-math.sin(transz1),math.cos(transz1),0],[0,0,1]])
        trans1=np.dot(np.dot(Rx1,Ry1),Rz1)
        dpoint1=np.array([dxc,dyc,1])
        dpoint1=dpoint1.T
        dw1=np.dot(trans1,dpoint1)
        dxw1=dw1[0]
        dyw1=dw1[1]
        if thingx1-worldx0<=0:
            dxw1=dxw1
        else:
            dxw1=-dxw1
        if thingy1-worldy0<=0:
            dyw1=dyw1
        else:
            dyw1=-dyw1
        #dxw1,dyw1=dyw1,dxw1
        dxw1=math.floor(dxw1)-18
        dyw1=math.floor(dyw1)-37
        if abs(dxw1)>=300 or abs(dyw1)>=300:
            continue
        hight1=0
        sendwords1=str(dxw1)+','+str(dyw1)+',0,'+str(hight1)+',1,0/0D'
        drophight1 = 300
        drophight2=0
        thinghight=15
        #sendwords2 = str(dxw2) + ',' + str(dyw2) + ',' + str(0) + ',' + str(drophight2) + ',' + str(1) + ',' + str(0) + '/0D'
        pickuppoint = str(dxw1) + ',' + str(dyw1) + ',0,' + str(drophight1) + ',1,0/0D'
        putpointup='200,0,0,'+str(drophight1)+',1,0/0D'
        putdownpoint='200,0,0,'+str(drophight2)+',1,0/0D'
        putdown='200,0,0,'+str(drophight2)+',2,0/0D'
        retools='0,-250,0,200,0,0/0D'
        drophight2=drophight2+thinghight
        bnum=bnum-1
        actionnum=5
        sends.append(sendwords1)

        # Receive the data in small chunks and retransmit it
        data = connection.recv(16)
        print >> sys.stderr, 'received "%s"' % data
        if (data == 'pass' and actionnum!=0):
            print >> sys.stderr, 'sending data back to the client'
            connection.sendall(sendwords1)
            actionnum=actionnum-1
            #print >> 'wait next command'

        else:
            print >> sys.stderr, 'no more data for', client_address
            connection.sendall(retools)
            break

        data = connection.recv(16)
        print >> sys.stderr, 'received "%s"' % data
        if (data == 'pass' and actionnum!=0):
            print >> sys.stderr, 'sending data back to the client'
            connection.sendall(pickuppoint)
            actionnum=actionnum-1
            #print >> 'wait next command'

        else:
            print >> sys.stderr, 'no more data for', client_address
            connection.sendall(retools)
            break

        data = connection.recv(16)
        print >> sys.stderr, 'received "%s"' % data
        if (data == 'pass' and actionnum!=0):
            print >> sys.stderr, 'sending data back to the client'
            connection.sendall(putpointup)
            actionnum=actionnum-1
            #print >> 'wait next command'
        else:
            print >> sys.stderr, 'no more data for', client_address
            connection.sendall(retools)
            break

        data = connection.recv(16)
        print >> sys.stderr, 'received "%s"' % data
        if (data == 'pass' and actionnum!=0):
            print >> sys.stderr, 'sending data back to the client'
            connection.sendall(putdownpoint)
            actionnum=actionnum-1
            #print >> 'wait next command'
        else:
            print >> sys.stderr, 'no more data for', client_address
            connection.sendall(retools)
            print >> 'END'
            break

        data = connection.recv(16)
        print >> sys.stderr, 'received "%s"' % data
        if (data == 'pass' and actionnum!=0):
            print >> sys.stderr, 'sending data back to the client'
            connection.sendall(putdown)
            actionnum=actionnum-1
            #print >> 'wait next command'
        else:
            print >> sys.stderr, 'no more data for', client_address
            connection.sendall(retools)
            #print >> 'END'
            break

        data = connection.recv(16)
        print >> sys.stderr, 'received "%s"' % data
        if (data == 'pass'):
            print >> sys.stderr, 'sending data back to the client'
            connection.sendall(retools)
            ifopencamera2=0
            #print >> 'end'
        else:
            print >> sys.stderr, 'no more data for', client_address
            connection.sendall(retools)
            ifopencamera2=0
            #print >> 'END'
            break

    """
    # Clean up the connection
    connection.close()
    """

    isend=input('is end this program?True or False')
    if isend == True:
        isstart = False

    #isstart=False
# Stop the pipeline and clean up
Tis1.Stop_pipeline()
cv2.destroyAllWindows()
