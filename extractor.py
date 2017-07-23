import numpy as np
import matplotlib as mpl
import global_vars
import math
from collections import deque
from astar import AStar
import skeletonize

def contains(small, big):
    for i in range(1 + len(big) - len(small)):
        if small == big[i:i+len(small)]:
            return i, i + len(small) - 1
    return False


def special_points(skeleton_copy):
    img=skeleton_copy.copy()
    img=global_vars.add_borders(img)
    #minima=[]
    #maxima=[]
    junctions=[]
    end_points=[]
    for x,y in np.transpose(np.nonzero(img)):
        nbrs=skeletonize.neighbours(x,y,img)
        if sum(nbrs)<2:
            end_points.append([x-1,y-1])
            """
            if (nbrs[0])[0]<x:
                minima.append([x-1,y-1])
            elif (nbrs[0])[0]>x:
                maxima.append([x-1,y-1])
            """
        elif sum(nbrs)>2:
            nbrs.append(nbrs[0])
            if not(contains([1,1],nbrs)):
                junctions.append([x,y])
    img=global_vars.delete_borders(img)
    return junctions,end_points
    
def neighbors2(node,img=None):
        if img==None:    img=global_vars.global_skel      
        nbrs=[]
        (x,y)=node
        for dx, dy in [(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]:
            temp_x=x+dx
            temp_y=y+dy
            if img[temp_x,temp_y]: nbrs.append((temp_x,temp_y))
        return nbrs

class graph(AStar):
    def __init__(self, img):
        self.lines =  img
        self.width = img.shape[1]
        self.height = img.shape[0]

    def heuristic_cost_estimate(self, n1, n2):
        (x1, y1) = n1
        (x2, y2) = n2
        return math.hypot(x2 - x1, y2 - y1)

    def distance_between(self, n1, n2):
        return 1

    def neighbors(self, node,img=0):
        if not(img):    img=global_vars.global_skel      
        nbrs=[]
        (x,y)=node
        for dx, dy in [(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]:
            temp_x=x+dx
            temp_y=y+dy
            if img[temp_x,temp_y]: nbrs.append((temp_x,temp_y))
        return nbrs


class Component:
    def __init__(self,strokes=None,left=None,right=None,top=None,bottom=None,bfs_img=None,img=None,sp=None,centroid=None,rep=None,r_nbr=None,l_nbr=None):
        self.strokes=strokes
        self.centroid=centroid
        self.left=left
        self.right=right
        self.top=top
        self.bottom=bottom
        self.bfs_img=bfs_img
        self.img=img
        self.sp=sp
        self.rep=rep
        self.r_nbr=r_nbr
        self.l_nbr=l_nbr

class Stroke:
    def __init__(self,stroke_type=None,stroke_points=None,centroid_x=None,centroid_y=None,std_x=None,std_y=None,shape_ratio=None,left=None,right=None,top=None,bottom=None,size=None,pixelspercol=None,img=0):
        self.stroke_type=stroke_type
        self.stroke_points=stroke_points
        self.centroid_x=centroid_x
        self.centroid_y=centroid_y
        self.std_x=std_x
        self.std_y=std_y
        self.shape_ratio=shape_ratio
        self.left=left
        self.right=right
        self.top=top
        self.bottom=bottom
        self.size=size
        self.img=img
        
        
    def display(self):
        print ("type:",self.stroke_type)
        print ("centroid:",self.centroid_x,self.centroid_y)
        print ("Standard deviation:",self.std_x,self.std_y)
        print ("shape:",self.shape_ratio)
        print ("left,right:",self.left,self.right)
        print ("top,bottom:",self.top,self.bottom)
        print ("size:",self.size)
        
    
def neighbors(x,y,img,visit,queue):
    nbrs=[]
    for dx, dy in [(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]:
        temp_x=x+dx
        temp_y=y+dy
        if img[temp_x,temp_y] and not(visit[temp_x,temp_y]) and [temp_x,temp_y] not in queue: nbrs.append([temp_x,temp_y])
    return nbrs

def build_stroke(x,y,img,visit,maxima,minima):
    stroke_direction=[0,0]
    stack=deque()
    previous=[x,y]
    stack.append([x,y])
    stroke_points=[]
    while len(stack):
        [current_x,current_y]=stack.pop()
        direction=[current_x-previous[0],current_y-previous[1]]
        nbrs=neighbors(current_x,current_y,img,visit)
        if direction[0]*stroke_direction[0]==-1:
            if direction[0]==1:
                maxima.append([previous[0]-1,previous[1]-1])
            else: minima.append([previous[0]-1,previous[1]-1])
            nbrs=neighbors(previous[0],previous[1],img,visit)
            break
        if direction[1]*stroke_direction[1]==-1:
            nbrs=neighbors(previous[0],previous[1],img,visit)
            break
        stroke_points.append([current_x,current_y])
        visit[current_x,current_y]=1
        if len(neighbors2((current_x,current_y),img))==1:
            if stroke_direction[0]==-1:
                maxima.append([current_x-1,current_y-1])
            elif stroke_direction[0]==1: minima.append([current_x-1,current_y-1])
        if len(nbrs)>1: 
            break
        if stroke_direction[0]==0:
            stroke_direction[0]=direction[0]
        if stroke_direction[1]==0:
            stroke_direction[1]=direction[1]
        previous=[current_x,current_y]
        stack.extend(nbrs)
    #print (stroke_direction)
    return Stroke(stroke_direction,stroke_points),nbrs,visit,maxima,minima
        
def build_stroke2(x,y,img,visit,maxima,minima,junctions,queue):
    stroke_direction=[0,0]
    stack=deque()
    previous=[x,y]
    stack.append([x,y])
    stroke_points=[]
    while len(stack):
        current_x,current_y=stack.pop()
        if [current_x,current_y] in junctions:
            queue.extend(neighbors(current_x,current_y,img,visit,queue))
            visit[current_x,current_y]=1
            break
           
        direction=[current_x-previous[0],current_y-previous[1]]
        
        
        if direction[0]*stroke_direction[0]==-1:
            if direction[0]==1:
                maxima.append([previous[0]-1,previous[1]-1])
            else: minima.append([previous[0]-1,previous[1]-1])
            for item in neighbors(previous[0],previous[1],img,visit,queue):
                queue.appendleft(item)
            break
        if direction[1]*stroke_direction[1]==-1:
            for item in neighbors(previous[0],previous[1],img,visit,queue):
                queue.appendleft(item)
            break
        stroke_points.append([current_x,current_y])
        visit[current_x,current_y]=1
        
        """
        if len(neighbors2((current_x,current_y),img))==1:
            if stroke_direction[0]==-1:
                maxima.append([current_x-1,current_y-1])
            elif stroke_direction[0]==1: minima.append([current_x-1,current_y-1])
        """
        if stroke_direction[0]==0:
            stroke_direction[0]=direction[0]
        if stroke_direction[1]==0:
            stroke_direction[1]=direction[1]
        nbrs=neighbors(current_x,current_y,img,visit,queue)
        for item in nbrs:
            if item in junctions:
                visit[item[0],item[1]]=1
                nbrs2=neighbors(item[0],item[1],img,visit,queue)
                queue.extend(nbrs2)
                return Stroke(stroke_direction,stroke_points),visit,maxima,minima,queue
        previous=[current_x,current_y]
        stack.extend(nbrs)
    #print (stroke_direction)
    return Stroke(stroke_direction,stroke_points),visit,maxima,minima,queue

            
def extract(img_copy):
    img=img_copy.copy()
    junctions,end_points=special_points(img)
    img=global_vars.add_borders(img)
    visited=np.zeros(img.shape)
    comps=[]
    minima=[]
    maxima=[]
    for y in range(img.shape[1]):
        for x in range(img.shape[0]-1,-1,-1):
            if img[x,y]:
                #print (visited)
                if not(visited[x,y]):
                    queue=deque()
                    strokes=[]
                    queue.append([x,y])
                    while len(queue):
                        [current_x,current_y]=queue.popleft()
                        #stroke,nbr,visited,maxima,minima=build_stroke(current_x,current_y,img,visited,maxima,minima)
                        stroke,visited,maxima,minima,queue=build_stroke2(current_x,current_y,img,visited,maxima,minima,junctions,queue)
                        if len(stroke.stroke_points):
                            strokes.append(stroke)
                    comps.append(Component(strokes=strokes))
    img=global_vars.delete_borders(img)
    return comps,maxima,minima,junctions,end_points                        

def calculate_stroke_stats(stroke):
    stroke.stroke_type=global_vars.stypes.index(stroke.stroke_type)+1
    [x_list,y_list]=list(zip(*stroke.stroke_points))
    stroke.centroid_x=np.mean(x_list)
    stroke.centroid_y=np.mean(y_list)
    stroke.std_x=np.std(x_list)
    stroke.std_y=np.std(y_list)
    stroke.left=min(y_list)
    stroke.right=max(y_list)
    stroke.top=min(x_list)
    stroke.bottom=max(x_list)
    stroke.size=len(stroke.stroke_points)
    stroke.img=global_vars.build_img(stroke.stroke_points)
    stroke.shape_ratio=(stroke.right-stroke.left+1)/(stroke.bottom-stroke.top+1)
    return stroke
    

def calculate_comp_stats(comps):
    global_vars.init_stypes()
    for comp_no,comp in enumerate(comps):
        comps[comp_no].points=[]
        for stroke in comp.strokes:
            comps[comp_no].points.extend(stroke.stroke_points)
        comps[comp_no].img=global_vars.build_img(comp.points,borders=False)
        comps[comp_no].left=min(comp.points,key=lambda x:x[1])[1]
        comps[comp_no].right=max(comp.points,key=lambda x:x[1])[1]
        comps[comp_no].top=min(comp.points,key=lambda x:x[0])[0]
        comps[comp_no].bottom=max(comp.points,key=lambda x:x[0])[0]
        comps[comp_no].centroid=np.mean(comp.points,axis=0)
        for stroke_no,stroke in enumerate(comp.strokes):
            comps[comp_no].strokes[stroke_no]=calculate_stroke_stats(stroke)
    comps=[comp for comp in comps if comp!=None]
    return comps

def build_median_img(comps):
    for comp_no,comp in enumerate(comps):
        #print (comp,comp_no,len(comps))
        if len(comp.strokes)<12:
            if comp.img.shape[0]<8: 
                comps[comp_no].bfs_img=comp.img.copy()
                comps[comp_no].sp=comp.points.copy()
            else:
                windows=[]
                for start_line in range(1,comp.img.shape[0]-6):
                    end_line=start_line+6
                    windows.append(np.transpose(np.nonzero(comp.img[start_line:end_line])))
                #print (comp.img.shape,len(np.transpose(np.nonzero(comp.img))),len(windows))
                dense_window=max(windows,key=lambda x:len(x))
                comps[comp_no].sp=[]
                for point in dense_window:
                    comps[comp_no].sp.append([point[0]+comp.top,point[1]+comp.left])
                comps[comp_no].bfs_img=global_vars.build_img(dense_window,[0,comp.img.shape[1]-1,0,comp.img.shape[0]-1],borders=False)
                
                fig=mpl.pyplot.figure(global_vars.figno)
                fig.add_subplot(1,3,1)
                mpl.pyplot.imshow(comp.img,'gray')
                fig.add_subplot(1,3,2)
                mpl.pyplot.imshow(comps[comp_no].bfs_img,'gray')
                global_vars.figno+=1
                
            
        else:
            comp.points=sorted(comp.points,key=lambda x:x[1])
            comp_points=[tuple(point) for point in comp.points]
            comps[comp_no].sp=list(graph(global_vars.global_skel).astar(comp_points[0],comp_points[len(comp_points)-1]))
            if comp.sp==None: 
                comps[comp_no].bfs_img=comp.img.copy()
            else:
                comps[comp_no].bfs_img=global_vars.build_img(comp.sp,[comp.left,comp.right,comp.top,comp.bottom])
                median_comp=extract(comp.bfs_img)[0]
                for index,stroke in enumerate(median_comp.strokes):
                    median_comp.strokes[index]=calculate_stroke_stats(stroke)
                median_comp.strokes=sorted(median_comp.strokes,key=lambda x:x.left)
                if len(median_comp.strokes)>4:
                    del median_comp.strokes[0]
                    del median_comp.strokes[-1]
                    if len(median_comp.strokes)>4:
                        del median_comp.strokes[0]
                        del median_comp.strokes[-1]
                    comps[comp_no].sp=[]
                    for stroke in median_comp.strokes:
                        comps[comp_no].sp.extend([[point[0]+comp.top-1,point[1]+comp.left-1] for point in stroke.stroke_points])
    points=[]
    for comp in comps:
        points.extend(comp.sp)
    return global_vars.build_img(points,[0,global_vars.global_skel.shape[1]-1,0,global_vars.global_skel.shape[0]-1]),comps
            