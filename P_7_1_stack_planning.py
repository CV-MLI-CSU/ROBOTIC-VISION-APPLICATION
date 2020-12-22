# -*- coding: utf-8 -*-
"""
# Author: Sisi(Scarlett) Liu <scarlett.liu@csu.edu.cn> & Jiaye Yang
# Uni: Central South University, Changsha, China
# Online course: Robotic Vision and Applications
# Created: 09-Sep-2020   Modified: 09-Sep-2020   Revision: 1.0
# Copyright � 2020 SR2V Ltd. All rights reserved

# Lecture 7-1 stack_planning

"""
import numpy as np
import cv2 as cv
import glob
import math

# def stack_planning(blocknum,blockcolor):
def stack_planning(thing,st_list,width,obj_point,worldx0=400,worldy0=400,ip1='192.168.125.1'):
    R=abb.Robot(ip=ip1)
    R.set_joints([0,0,0,0,0,0])
    print("Staring moving")
    #设置tool down
    q=Quaternion([0,0,1,0])
    j1=0
    sum_j=0#用来表示现在是第几个
    height=15#代表积木高度
    for j in st_list:
        j1=j1+1#代表层数
        k=0#k代表本层第几
        for i in range(j):
            index=sum_j+i
            # 将角度换为弧度
            angle1=radians(thing[index][3])
            # 换算四元数
            my_quaternion = Quaternion(axis=[0,0,1], angle=angle1)
            quat=(q*my_quaternion).elements
            #手臂到达其上空
            R.set_cartesian([[thing[index][1]+worldx0,thing[index][2]+worldy0,200],quat])
            #抓取
            R.set_cartesian([[thing[index][1]+worldx0,thing[index][2]+worldy0,height],quat])
            print('抓取位置为：',R.get_cartesian())
            #抬起
            R.set_cartesian([[thing[index][1]+worldx0,thing[index][2]+worldy0,200],[quat[0],quat[1],quat[2],quat[3]]])

            quat=Quaternion(axis=[0,0,1],angle=-angle1)
            quat=(q*quat).elements
            #平移
            R.set_cartesian([[obj_point[0]+width*k,obj_point[1],200],[quat[0],quat[1],quat[2],quat[3]]])
            #放置
            R.set_cartesian([[obj_point[0]+width*k,obj_point[1],(j1)*height],[quat[0],quat[1],quat[2],quat[3]]])
            print('放置位置为：',R.get_cartesian())
            #提起来
            R.set_cartesian([[obj_point[0]+width*k,obj_point[1],200],[quat[0],quat[1],quat[2],quat[3]]])
            k=k+1
        sum_j=sum_j+j
    print("grasping end，in total {} blocks are stacked.".format(sum_j))
