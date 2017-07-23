import cv2
import numpy
import matplotlib
import global_vars

def calc_thresh_stats(img):
    out=cv2.connectedComponentsWithStats(1-img,8,cv2.CV_32S)
    area=[item[cv2.CC_STAT_AREA] for item in out[2]]
    avg=numpy.mean(area[1:len(area)])
    return avg

def binarize(img):
    thresh_items=[]
    new_img = cv2.adaptiveThreshold(img,1,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                                    cv2.THRESH_BINARY,15,2)
    thresh_items.append([new_img,calc_thresh_stats(new_img)])
    ret,new_img = cv2.threshold(img,0,1,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    thresh_items.append([new_img,calc_thresh_stats(new_img)])
    ret,new_img = cv2.threshold(img,155,1,cv2.THRESH_BINARY)
    thresh_items.append([new_img,calc_thresh_stats(new_img)])
    new_img = cv2.adaptiveThreshold(img,1,cv2.ADAPTIVE_THRESH_MEAN_C,\
                                    cv2.THRESH_BINARY,15,2)
    thresh_items.append([new_img,calc_thresh_stats(new_img)])
    fig=matplotlib.pyplot.figure(global_vars.figno)
    for index,item in enumerate(thresh_items):
        fig.add_subplot(2,2,index+1)
        matplotlib.pyplot.imshow(item[0],'gray')
    global_vars.figno+=1
    return (max(thresh_items,key=lambda x:x[1]))[0]