import numpy as np
import cv2
import math

def clean_img(img_copy):
    img=img_copy.copy()
    lcount,labels,stats,centroids=cv2.connectedComponentsWithStats(img.astype(np.uint8),8,cv2.CV_32S)
    points=[[] for i in range(lcount)]
    for index,item in np.ndenumerate(labels):
        points[item].append(index)
    stats_zip=list(zip(*stats[1:lcount]))
    height_std=np.median(stats_zip[cv2.CC_STAT_HEIGHT])
    width_std=np.median(stats_zip[cv2.CC_STAT_HEIGHT])
    noise_labels=[label for label,stat in enumerate(stats) if stat[cv2.CC_STAT_HEIGHT]>20*height_std or stat[cv2.CC_STAT_WIDTH]>20*width_std or stat[cv2.CC_STAT_AREA]<4]
    for label in noise_labels:
        for point in points[label]:
            img[point]=0
    return img

#find location of a particular point after rotation
def rotate_point(point,angle,pivot=(0,0),scale=1.0,trans=[0,0]):
    point=list(point)
    point[0]=point[0]+trans[0]
    point[1]=point[1]+trans[1]
    cos=scale*math.cos(angle)
    sin=scale*math.sin(angle)
    x=(point[0]-pivot[0])*cos-(point[1]-pivot[1])*sin
    y=(point[1]-pivot[1])*cos+(point[0]-pivot[0])*sin
    return x+pivot[0],y+pivot[1]


#rotate an image by the angle specified in degrees about a pivot point
def rotate_bound(image, angle, scale=1.0, pivot=(0,0), reshape=True):
    (h, w) = image.shape[:2]
    angle_rad=angle*math.pi/180
    # if reshape==True, borders are extended to retain all data from input image
    if reshape:
        cos=scale*math.cos(angle_rad)
        sin=scale*math.sin(angle_rad)
        
        #find new locations of corner points of the image
        x1,y1=rotate_point([h,0],-angle_rad,pivot,scale)
        x2,y2=rotate_point([0,w],-angle_rad,pivot,scale)
        x3,y3=rotate_point([h,w],-angle_rad,pivot,scale)
        x4,y4=rotate_point([0,0],-angle_rad,pivot,scale)
        x=[int(x1),int(x2),int(x3),int(x4)]
        y=[int(y1),int(y2),int(y3),int(y4)]
        x_min=min(x)
        x_max=max(x)
        y_min=min(y)
        y_max=max(y)
        M = np.array([[cos,-sin,(pivot[1]*(1-cos)+pivot[0]*sin-y_min)],[sin,cos,(pivot[0]*(1-cos)-pivot[1]*sin-x_min)]])
        nW=y_max-y_min
        nH=x_max-x_min
    else: 
        x_min=0
        y_min=0
        nW=w
        nH=h
        M = cv2.getRotationMatrix2D(pivot, -angle, 1.0)
    return cv2.warpAffine(image, M, (nW, nH),flags=cv2.INTER_CUBIC),[x_min,y_min]

def rotate2(img_copy,r=[-45,45]):
    img=img_copy.copy()
    diff_set=[]
    for angle in range(r[0],r[1]):
        rot_img,trans_factor=rotate_bound(img,angle)
        profile=[np.sum(row) for row in rot_img]
        diff_set.append((angle,rot_img,profile))
    optimal_set=max(diff_set,key=lambda x:np.std(x[2]))
    return optimal_set


def binarize(img):
    def calc_thresh_stats(img):
        img=1-img
        out=cv2.connectedComponentsWithStats(img,8,cv2.CV_32S)
        area=[item[cv2.CC_STAT_AREA] for item in out[2]]
        avg=np.mean(area[1:len(area)])
        return [img,avg]
    
    thresh_items=[]
    new_img = cv2.adaptiveThreshold(img,1,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                                    cv2.THRESH_BINARY,15,2)
    
    thresh_items.append(calc_thresh_stats(new_img))
    
    ret,new_img = cv2.threshold(img,0,1,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    thresh_items.append(calc_thresh_stats(new_img))
    ret,new_img = cv2.threshold(img,155,1,cv2.THRESH_BINARY)
    thresh_items.append(calc_thresh_stats(new_img))
    new_img = cv2.adaptiveThreshold(img,1,cv2.ADAPTIVE_THRESH_MEAN_C,\
                                    cv2.THRESH_BINARY,15,2)
    thresh_items.append(calc_thresh_stats(new_img))
    """
    fig=mpl.pyplot.figure(global_vars.figno)
    for index,item in enumerate(thresh_items):
        fig.add_subplot(2,2,index+1)
        mpl.pyplot.imshow(item[0],'gray')
    global_vars.figno+=1
    """
    return (max(thresh_items,key=lambda x:x[1]))[0]