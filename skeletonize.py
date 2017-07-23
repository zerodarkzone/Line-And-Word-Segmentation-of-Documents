import numpy
import cv2
import global_vars
import numpy as np

def neighbors_x(node,img):
    nbrs=[]
    (x,y)=node
    for dx, dy in [(-1,-1),(-1,0),(-1,1),(1,1),(1,0),(1,-1)]:
        temp_x=x+dx
        temp_y=y+dy
        if img[temp_x,temp_y]: nbrs.append((temp_x,temp_y))
    return nbrs

def neighbors_y(node,img):
    nbrs=[]
    (x,y)=node
    for dx, dy in [(-1,-1),(-1,1),(0,1),(1,1),(1,-1),(0,-1)]:
        temp_x=x+dx
        temp_y=y+dy
        if img[temp_x,temp_y]: nbrs.append((temp_x,temp_y))
    return nbrs


def smooth_skeleton(skeleton_copy):
    img=skeleton_copy.copy()
    img=global_vars.add_borders(img)
    deletion=[]
    addition=[]
    for x in range(1,img.shape[0]-1):
        y=1
        while y<img.shape[1]-1:
            if img[x,y]:
                y_start=y
                y+=1
                #while img[x,y] :
                while img[x,y]:
                    y+=1
                y-=1
                y_end=y
                start_nbrs=neighbors_x((x,y_start),img)
                end_nbrs=neighbors_x((x,y_end),img)
                if y_start==y_end:
                    if len(start_nbrs)==2:
                        if (start_nbrs[0])[0]==(start_nbrs[1])[0]:
                          for temp_y in range(y_start,y_end+1):
                              deletion.append([x,temp_y])
                              addition.append([(start_nbrs[0])[0],temp_y])
                else:
                    y_arr=list(range(y_start+1,y_end))
                    for item in y_arr:
                        if len(neighbors_x((x,item),img)): 
                            y+=1
                            continue
                    
                    if len(start_nbrs)==1 and len(end_nbrs)==1:
                          if (start_nbrs[0])[0]==(end_nbrs[0])[0]:
                              for temp_y in range(y_start,y_end+1):
                                  deletion.append([x,temp_y])
                                  #img[x,temp_y]=0
                                  addition.append([(start_nbrs[0])[0],temp_y])
                                  #img[(start_nbrs[0])[0],temp_y]=1
                    
                                  #img[x,temp_y]=0
                                  #img[(start_nbrs[0])[0],temp_y]=1 
            y+=1
    for item in deletion:
        img[item[0],item[1]]=0
    for item in addition:
        img[item[0],item[1]]=1
    deletion=[]
    addition=[]
    for y in range(1,img.shape[1]-1):
        x=1
        while x<img.shape[0]-1:
            if img[x,y]:
                x_start=x
                x+=1
                #while img[x,y]: 
                while img[x,y]:    
                    x+=1 
                x-=1
                x_end=x
                start_nbrs=neighbors_y((x_start,y),img)
                end_nbrs=neighbors_y((x_end,y),img)
                
                if x_start==x_end:
                    if len(start_nbrs)==2:
                        if (start_nbrs[0])[1]==(start_nbrs[1])[1]:
                          for temp_x in range(x_start,x_end+1):
                              deletion.append([temp_x,y])
                              addition.append([temp_x,(start_nbrs[0])[1]])
                else:
                    x_arr=list(range(x_start+1,x_end))
                    for item in x_arr:
                        
                        if len(neighbors_y((item,y),img)): 
                            x+=1
                            continue
                    
                    if len(start_nbrs)==1 and len(end_nbrs)==1:
                          if (start_nbrs[0])[1]==(end_nbrs[0])[1]:
                              for temp_x in range(x_start,x_end+1):
                                  deletion.append([temp_x,y])
                                  addition.append([temp_x,(start_nbrs[0])[1]])
                                  #img[temp_x,y]=0
                                  #img[temp_x,(start_nbrs[0])[1]]=1
                    
                                  #img[temp_x,y]=0
                                  #img[temp_x,(start_nbrs[0])[1]]=1
            x+=1
    for item in deletion:
        img[item[0],item[1]]=0
    for item in addition:
        img[item[0],item[1]]=1
    img=global_vars.delete_borders(img)
    return img


def smooth_contours(img_copy):
    img=img_copy.copy()
    
    #print (type(img[0,0]))
    #img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img=global_vars.add_borders(img)
    
    kernel1=np.array([[1,1,0],[1,-1,0],[1,1,0]],dtype=np.int8)
    kernel2=np.array([[1,1,1],[1,-1,1],[0,0,0]],dtype=np.int8)
    kernel3=np.array([[0,1,1],[0,-1,1],[0,1,1]],dtype=np.int8)
    kernel4=np.array([[0,0,0],[1,-1,1],[1,1,1]],dtype=np.int8)
    while True:
        img2=img.copy()
        img=np.array(img*255,dtype=np.uint8)
        hit_miss1=cv2.morphologyEx(img,cv2.MORPH_HITMISS,kernel1)
        hit_miss2=cv2.morphologyEx(img,cv2.MORPH_HITMISS,kernel2)
        hit_miss3=cv2.morphologyEx(img,cv2.MORPH_HITMISS,kernel3)
        hit_miss4=cv2.morphologyEx(img,cv2.MORPH_HITMISS,kernel4)
        img=np.logical_or.reduce((hit_miss1,hit_miss2,hit_miss3,hit_miss4,img))
        img_negative=np.array(255-img,dtype=np.uint8)
        hit_miss1=cv2.morphologyEx(img_negative,cv2.MORPH_HITMISS,kernel1)
        hit_miss2=cv2.morphologyEx(img_negative,cv2.MORPH_HITMISS,kernel2)
        hit_miss3=cv2.morphologyEx(img_negative,cv2.MORPH_HITMISS,kernel3)
        hit_miss4=cv2.morphologyEx(img_negative,cv2.MORPH_HITMISS,kernel4)
        img=np.logical_or.reduce((hit_miss1,hit_miss2,hit_miss3,hit_miss4,img_negative))
        img=255-img_negative
        if np.array_equal(img,img2): break
    img=global_vars.delete_borders(img)
    
    return img

def neighbours(x, y,a):
    p = []
    for dx, dy in [(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]:
        p.append(a[x+dx,y+dy])
    return p


def skeletonizer(image):
    image2=image.copy()
    image2=global_vars.add_borders(image2)
    iterations=0
    while True:
        iterations+=1
        image4=image2.copy()
        #while True:
            #image3=image2.copy()
        for x in range(1,image2.shape[0]-1):
            y=1
            while y<image2.shape[1]-1:
                if image2[x,y] :
                    y_start=y
                    while y<image2.shape[1]-1 and image2[x,y]:    y+=1
                    y_end=y-1
                    if y_end!=y_start:
                        p=neighbours(x,y_end,image2)
                        if not((p[2] and not(p[1])) or (p[4] and not(p[5])) or sum(p)==1): #p[0]+p[1]+p[2]+p[3]+p[4]+p[5]!=0 and p[1]+p[2]+p[3]+p[4]+p[5]+p[6]!=0:
                            image2[x,y_end]=0
                        if image2[x,y_start+1]:
                            p=neighbours(x,y_start,image2)
                            if not((p[0] and not(p[1])) or (p[6] and not(p[5])) or sum(p)==1):
                                image2[x,y_start]=0
                y+=1
            #if np.array_equal(image3,image2): break
        #while True:
            #image3=image2.copy()
        for y in range(1,image2.shape[1]-1):
            x=1
            while x<image2.shape[0]-1:
                if image2[x,y] :
                    x_start=x
                    while x<image2.shape[0]-1 and image2[x,y]: x+=1
                    x_end=x-1
                    if x_end!=x_start:
                        p=neighbours(x_end,y,image2)
                        if not((p[6] and not(p[7])) or (p[4] and not(p[3])) or sum(p)==1) : #p[2]+p[3]+p[4]+p[5]+p[6]+p[7]!=0 and p[3]+p[4]+p[5]+p[6]+p[7]!=0:
                            image2[x_end,y]=0
                        if image2[x_start+1,y]:
                            p=neighbours(x_start,y,image2)
                            if not((p[0] and not(p[7])) or (p[2] and not(p[3])) or sum(p)==1) : #p[6]+p[7]+[0]+[1]+[2]+[3]!=0 and p[7]+[0]+[1]+[2]+[3]+p[4]!=0:
                                image2[x_start,y]=0
                x+=1
            #if np.array_equal(image3,image2): break
        if numpy.array_equal(image4,image2): break
    image2=global_vars.delete_borders(image2)       
    return image2
