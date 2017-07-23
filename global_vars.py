import numpy as np
import cv2
import skeletonize
import extractor 
import scipy
import math


def init_shape(img):
    global shape
    shape=img.shape
    
def init():
    global start
    start=0
    
def init_graph(gr):
    global global_graph
    global_graph=gr
    
def init_figno(a):
    global figno
    figno=a
    
def init_skel(skeleton):
    global global_skel
    global_skel=skeleton
    
def init_stypes():
    global stypes
    stypes=[[0,-1],[-1,-1],[-1,0],[-1,1],[0,1],[1,1],[1,0],[1,-1],[0,0]]
    
def build_img(points,stats=None,borders=True):
    if stats==None:
        left=min(points,key=lambda x:x[1])[1]
        right=max(points,key=lambda x:x[1])[1]
        top=min(points,key=lambda x:x[0])[0]
        bottom=max(points,key=lambda x:x[0])[0]
    else: left,right,top,bottom=stats
    img=np.zeros((bottom-top+1,right-left+1),dtype=np.uint8)
    for point in points:
        img[point[0]-top,point[1]-left]=1
    #if borders==True:   img=add_borders(img)
    return img

def delete_borders(img_copy):
    img=img_copy.copy()
    img=np.delete(img,[img.shape[0]-1,0],axis=0)
    img=np.delete(img,[img.shape[1]-1,0],axis=1)
    return img
    
def add_borders(img_copy):
    img=img_copy.copy()
    img=np.append(img,np.zeros((img.shape[0],1),dtype=np.uint8),axis=1)
    img=np.insert(img,0,np.zeros(img.shape[1],dtype=np.uint8),0)
    img=np.append(img,[np.zeros(img.shape[1],dtype=np.uint8)],0)
    img=np.insert(img,0,0,axis=1)
    return img

def optimal_angle(ang):
    global angle
    angle=math.radians(-ang)
    
def init_skeleton(img_copy):
        img=img_copy.copy()
        img=skeletonize.smooth_contours(img)
        skeleton=skeletonize.skeletonizer(img)
        skeleton=skeletonize.smooth_skeleton(skeleton)
        skeleton=skeletonize.skeletonizer(skeleton)
        return skeleton


def fill_holes(img_copy):
    img=img_copy.copy()
    img=scipy.ndimage.morphology.binary_fill_holes(img)
    return img

def init_type(comp):
    if len(comp.strokes)<4:
        return 0
    elif len(comp.strokes)<5:
        return 1
    elif 5<=len(comp.strokes)<10:   return 2
    else: return 3

class comp:
    
    def __init__(self,stat=None,centroid=None,points=None,line_no=None):
        self.top=stat[cv2.CC_STAT_TOP]
        self.left=stat[cv2.CC_STAT_LEFT]
        self.height=stat[cv2.CC_STAT_HEIGHT]
        self.width=stat[cv2.CC_STAT_WIDTH]
        self.centroid=centroid
        self.points=points
        self.left_point=min(self.points,key=lambda x:x[1])
        self.right_point=max(self.points,key=lambda x:x[1])
        self.line_no=line_no
        self.img=build_img(self.points)
        self.filled_img=fill_holes(self.img)
        self.filled_skel=init_skeleton(self.filled_img)
        self.features2,self.maxima2,self.minima2,self.junctions2,self.end_points2=extractor.extract(self.filled_skel)
        self.features2=extractor.calculate_comp_stats(self.features2)       
        self.type=init_type(self.features2[0])
        self.init_max_points()
        
    def init_convexhull(self):
        hull=scipy.spatial.ConvexHull(self.points)
        self.hull_points=[point for point_no,point in enumerate(hull.points) if point_no in hull.vertices]

    def init_max_points(self):
        self.left_points=[]
        self.right_points=[]
        for x in range(self.top,self.top+self.height):
            y=[point[1] for point in self.points if point[0]==x]
            self.left_points.append([x,min(y)])
            self.right_points.append([x,max(y)])


def init_used_keys():
    global used
    used=[]


def init_comps(output):
    lcount,labels,stats,centroids=output
    points_arr=[[]for i in range(lcount)]
    for loc,label in np.ndenumerate(labels):
        points_arr[label].append(loc)
    temp_dict={}
    for comp_no in range(1,lcount):
        temp_dict[comp_no]=comp(stats[comp_no],centroids[comp_no],points_arr[comp_no])    
    global comp_dict
    comp_dict=temp_dict
    
init_figno(1)