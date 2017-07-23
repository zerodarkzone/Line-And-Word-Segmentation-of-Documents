import numpy as np
import matplotlib as mpl
import cv2
import global_vars
import scipy
import math
import extractor
from sklearn import mixture
import scipy.ndimage


def dist_line_point(point,coeffs):
    a,b,c=coeffs
    return abs(point[0]*a+point[1]*b+c)/math.sqrt(a**2+b**2)

class Lin2:
    def __init__(self,item,clines=None):
        self.heights=[]
        self.puncts=[]
        self.lorw=[]
        self.regressors=[]
        self.median_points=[]
        self.sign_points=[]
        self.farther_centers=[]
        self.mean_ht=0
        self.add_lorw(item,clines=clines)
        if len(self.lorw):
            temp_keys=global_vars.used.copy()
            temp_keys=sorted(temp_keys,key=lambda x:global_vars.comp_dict[x].left+global_vars.comp_dict[x].width,reverse=True)
            for comp_no in temp_keys:
                comp=global_vars.comp_dict[comp_no]
                if (comp.top+comp.height>self.bottom and comp.top>self.bottom) or (comp.top+comp.height<self.top and comp.top<self.top): 
                    continue
                if min(scipy.spatial.distance_matrix(comp.right_points,global_vars.comp_dict[self.lorw[0]].left_points).flatten())<self.max_space:
                    if comp.height<self.mean_ht: continue
                    global lines
                    for line_no,line in enumerate(lines):
                        if comp_no in line.lorw+line.puncts:
                            self.break_component3(comp_no,[self],ret=line_no)
        
    def break_component3(self,comp_no,clines,ret=None):
        comp=global_vars.comp_dict[comp_no]
        for line in clines:
            temp_skel=comp.filled_skel.copy()
            slope,lintercept=np.mean(line.regressors,axis=0)
            upper_intercept=lintercept+(line.center_ht)
            lower_intercept=lintercept-(line.center_ht)
            for x,y in np.transpose(np.nonzero(temp_skel)):
                if (x+comp.top)-slope*(y+comp.left)>upper_intercept or (comp.top+x)-slope*(y+comp.left)<lower_intercept:
                    temp_skel[x,y]=0
            comps,m1,m2,j,ep=extractor.extract(temp_skel)
            for item in comps:
                if len(item.strokes)>4:
                    upper_minima=upper_intercept
                    lower_minima=lower_intercept
                    in_components=np.zeros(global_vars.shape,dtype=np.uint8)
                    out_components=np.zeros(global_vars.shape,dtype=np.uint8)   
                    for x,y in comp.points:
                        if lower_minima<x-y*slope<upper_minima:
                            in_components[x,y]=1
                        else:   out_components[x,y]=1
                    for rep in range(2):    
                        out_comps={}
                        lcount,labels,stats,centroids=cv2.connectedComponentsWithStats(out_components.astype(np.uint8),8,cv2.CV_32S)
                        points_arr=[[]for i in range(lcount)]
                        for loc,label in np.ndenumerate(labels):
                            points_arr[label].append(loc)
                        for cno in range(1,lcount):
                            #global_vars.comp_dict[len(global_vars.comp_dict)+1]=global_vars.comp(stats[cno],centroids[cno],points_arr[cno])
                            out_comps[cno]=global_vars.comp(stats[cno],centroids[cno],points_arr[cno])
                            
                        if not(rep):
                            for cno,c in out_comps.items():
                                if c.type<1:
                                    for x,y in c.points:
                                        in_components[x,y]=1
                                        out_components[x,y]=0
                            
                            if not(np.sum(out_components)):
                                #global_vars.figno+=1
                                return False
                                
                        in_comps={}
                        lcount,labels,stats,centroids=cv2.connectedComponentsWithStats(in_components.astype(np.uint8),8,cv2.CV_32S)
                        points_arr=[[]for i in range(lcount)]
                        for loc,label in np.ndenumerate(labels):
                            points_arr[label].append(loc)
                        for cno in range(1,lcount):
                            in_comps[cno]=global_vars.comp(stats[cno],centroids[cno],points_arr[cno])
                        if not(rep):
                            for cno,c in in_comps.items():
                                if c.type<1: 
                                    
                                    for x,y in c.points:
                                        out_components[x,y]=1
                                        in_components[x,y]=0
                        
                            if not(np.sum(in_components)):
                                return False
                    if ret==None:
                        global global_count
                        global_count+=1
                        for cno,c in in_comps.items():
                            global_vars.comp_dict[len(global_vars.comp_dict)+1]=c
                            
                        for cno,c in out_comps.items():
                            global_vars.comp_dict[len(global_vars.comp_dict)+1]=c
                    else:
                        global lines
                        lines[ret].delete_item(comp_no)
                        for cno,c in in_comps.items():
                            k=len(global_vars.comp_dict)+1
                            global_vars.comp_dict[k]=c
                            self.add_lorw(k)
                            global_vars.used.append(k)
                        for cno,c in out_comps.items():
                            if not(cno):
                                global_vars.comp_dict[comp_no]=c
                                lines[ret].add_lorw(comp_no)
                            else:
                                k=len(global_vars.comp_dict)+1
                                global_vars.comp_dict[k]=c
                                lines[ret].add_lorw(k)
                                global_vars.used.append(k)
                            
                    sort_keys()
                    return True
        return False
        
    
                            
    def calculate_median(self,typ=False):
        if typ==True:
            self.update_img_stats(typ=True)
        comps=[]
        y=0
        while y<self.img.shape[1]:
            if np.sum((np.transpose(self.img))[y]):
                comp=[]
                while y<self.img.shape[1] and np.sum((np.transpose(self.img))[y]):
                    comp.extend([[self.top+index,y+self.left] for index,item in enumerate((np.transpose(self.img))[y]) if item])
                    y+=1
                comps.append(comp)
            y+=1
        self.new_comps=[[comp] for comp in comps]
        for index,item in enumerate(self.new_comps):
            comp=item[0]
            #print (len(comp))
            left_points=[]
            right_points=[]
            for x in range(min(comp,key=lambda x:x[0])[0],max(comp,key=lambda x:x[0])[0]+1):
                y=[point[1] for point in comp if point[0]==x]
                if not(len(y)): continue
                left_points.append([x,min(y)])
                right_points.append([x,max(y)])
            self.new_comps[index].extend([left_points,right_points])
        centroids=[np.mean(comp,axis=0) for comp in comps]
        centroids=[(int(x),int(y)) for x,y in centroids]
        median_line=[]
        if len(centroids)==1:
            for y in range(self.img.shape[1]):
                median_line.append((centroids[0])[0])
        else:
            for index,centroid in enumerate(centroids[0:len(centroids)-1]):
                next_centroid=centroids[index+1]
                slope=(next_centroid[0]-centroid[0])/(next_centroid[1]-centroid[1])
                intercept=centroid[0]-slope*centroid[1]
                first_index=centroid[1]
                last_index=next_centroid[1]
                if not(index):
                    first_index=self.left
                    
                if index==len(centroids)-2:
                    last_index=self.right
                    
                for y in range(first_index,last_index):
                    median_line.append(slope*y+intercept)
        if len(median_line)<self.img.shape[1]:
            diff=self.img.shape[1]-len(median_line)
            for rep in range(diff):
                median_line.append(median_line[-1])
        elif len(median_line)>self.img.shape[1]:
            median_line=median_line[0:self.img.shape[1]]
        self.median_line= median_line
        
    def add_lorw(self,item,clines=None,inspect=True):
        if inspect:
            if clines!=None:
                if self.mean_ht<global_vars.comp_dict[item].height:
                    if self.break_component3(item,clines):
                        return 
        self.lorw.append(item)
        self.lorw=sorted(self.lorw,key=lambda x:global_vars.comp_dict[x].centroid[0])
        self.update_img_stats()
        comp=global_vars.comp_dict[item]
        self.sign_points.extend([(point[0]+comp.top,point[1]+comp.left) for point in comp.minima2+comp.maxima2+comp.junctions2+comp.end_points2])
        for column_no in range(comp.img.shape[1]):
                
                indices=[item_no for item_no,item in enumerate((np.transpose(comp.img))[column_no]) if np.sum(item)]
                
                if len(indices):
                    self.heights.append(max(indices)-min(indices))
        self.update_regressor2()
    
    
    def add_puncts(self,item):
        self.puncts.append(item)
        comp=global_vars.comp_dict[item]
        for column_no in range(comp.img.shape[1]):
            indices=[item_no for item_no,item in enumerate((np.transpose(comp.img))[column_no]) if np.sum(item)]
            if len(indices):
                self.heights.append(max(indices)-min(indices))
        
    def delete_item(self,item):
        if item in self.lorw:
            self.lorw.remove(item)
        elif item in self.puncts:
            self.puncts.remove(item)
    
    def update_img_stats(self,typ=False):
        points=[]
        if typ==False:
            for comp_no in self.lorw:
                points.extend(global_vars.comp_dict[comp_no].points)
        else:   
            for comp_no in self.lorw+self.puncts:
                points.extend(global_vars.comp_dict[comp_no].points)
        points_zip=list(zip(*points))
        self.top=min(points_zip[0])
        self.bottom=max(points_zip[0])
        self.left=min(points_zip[1])
        self.right=max(points_zip[1])
        self.img=global_vars.build_img(points)
            
    def update_regressor2(self):
        if len(self.lorw)==1:    self.max_space=global_vars.comp_dict[self.lorw[0]].width*4
        else:   self.max_space=max([abs(global_vars.comp_dict[self.lorw[comp_no+1]].centroid[0]-global_vars.comp_dict[comp].centroid[0]) for comp_no,comp in enumerate(self.lorw[0:len(self.lorw)-1])])*4
        
        if len(self.regressors)>4:
            del(self.regressors[-1])

        if len(self.sign_points)<25:
            last_comp=global_vars.comp_dict[self.lorw[-1]]
            self.regressors.append([0,(2*last_comp.top+last_comp.height+1)/2])
        else:
            self.sign_points=sorted(self.sign_points,key=lambda x:x[1])
            points=self.sign_points[len(self.sign_points)-25:len(self.sign_points)]
            points_zip=list(zip(*points))
            self.regressors.append(list(np.polyfit(points_zip[1],points_zip[0],1)))
        
        if len(self.heights)<3:
            self.mean_ht=self.center_ht=global_vars.comp_dict[self.lorw[-1]].height
        else:
            self.heights=sorted(self.heights,reverse=True)
            temp_heights=[[ht] for ht in self.heights]
            g=mixture.GMM(n_components=3,covariance_type='spherical')
            g.fit(temp_heights)
            mean_hts=sorted(list(g.means_))
            self.mean_ht=mean_hts[len(mean_hts)-1]
            self.center_ht=mean_hts[min([1,len(mean_hts)-1])]
        
def sort_keys():
    temp_dict=sorted(global_vars.comp_dict.items(),key=lambda x:x[1].left)
    temp_dict={k:v for k,v in temp_dict}
    global keys
    keys=list(temp_dict.keys())

def segment_lines(img_copy):
    global global_count
    global_count=0
    global_vars.init_used_keys()
    global_vars.init_shape(img_copy)
    output=cv2.connectedComponentsWithStats(img_copy.astype(np.uint8),8,cv2.CV_32S)
    global_vars.init_comps(output)
    sort_keys()
    global lines
    puncts=[]
    lines=[]
    while len(global_vars.used)<len(keys):
        for k in keys:
            if k not in global_vars.used: 
                global_vars.used.append(k)
                comp_no=k
                if not(len(lines)): 
                    nline=Lin2(comp_no)
                    if len(nline.lorw):
                        lines.append(nline)
                else:
                    #comp_no=keys[keys_index]
                    comp=global_vars.comp_dict[comp_no]
                    lines=sorted(lines,key=lambda x:scipy.spatial.distance.euclidean(global_vars.comp_dict[x.lorw[-1]].right_point,comp.left_point))
                    candidates=list(range(0,min([len(lines),5])))
                    candidates=[[candidate] for candidate in candidates if global_vars.comp_dict[comp_no].left-lines[candidate].right<=lines[candidate].max_space]
                    if not(len(candidates)):
                        nline=Lin2(comp_no)
                        if len(nline.lorw):
                            lines.append(nline)
                    else:
                        for no,candidate in enumerate(candidates):
                            line=lines[candidate[0]]
                            lines[candidate[0]].regressors=sorted(line.regressors,key=lambda x:np.mean([dist_line_point([point[1],point[0]],[x[0],-1,x[1]]) for point in comp.points]))
                            deviations=[]
                            for regressor in lines[candidate[0]].regressors:
                                deviations.append(np.mean([dist_line_point([point[1],point[0]],[regressor[0],-1,regressor[1]]) for point in comp.points]))
                            candidates[no].append(np.mean(deviations))
                        candidates=sorted(candidates,key=lambda x:x[1])
                        closest_line_index,dev=candidates[0]
                        closest_line=lines[closest_line_index]
                        if comp.type<2 and (comp.height<0.33*lines[closest_line_index].mean_ht or comp.width>2*comp.height):
                                #lines[closest_line_index].add_puncts(comp_no)
                                puncts.append(comp_no)
                                continue
                        
                        
                        upper_intercept=(lines[closest_line_index].regressors[0])[1]+lines[closest_line_index].mean_ht/2
                        lower_intercept=(lines[closest_line_index].regressors[0])[1]-lines[closest_line_index].mean_ht/2
                        window_points=[point for point in comp.points if lower_intercept<=point[0]-point[1]*(lines[closest_line_index].regressors[0])[0]<=upper_intercept]
                        
                        k=0.2
                        
                        if len(window_points)<k*len(comp.points):
                            nline=Lin2(comp_no,[closest_line])
                            if len(nline.lorw):
                                lines.append(nline)
                        else:
                            
                            #detect overlap
                            
                            new_points=[]
                            for stroke in comp.features2[0].strokes:
                                new_points.extend(list(zip(*stroke.stroke_points))[1])
                            new_points=np.unique(new_points)
                            new_points=[point+comp.left for point in new_points]
                            #stroke_nos=[]
                            line_points=[]
                            for column_no,column in enumerate(np.transpose(closest_line.img)):
                                if np.sum(column): line_points.append(column_no+closest_line.left)
                            intersection_set=set(new_points).intersection(line_points)
                            count1=0
                            for index,cno in enumerate(closest_line.lorw):
                                c=global_vars.comp_dict[cno]
                                for stroke_no,stroke in enumerate(c.features2[0].strokes):
                                    pzip=np.unique(list(zip(*stroke.stroke_points))[1])
                                    pzip=[point+c.left for point in pzip]
                                    intersect=set(intersection_set).intersection(pzip)
                                    if len(intersect):
                                        count1+=1
                            count2=0
                            for stroke_no,stroke in enumerate(comp.features2[0].strokes):
                                pzip=np.unique(list(zip(*stroke.stroke_points))[1])
                                pzip=[point+c.left for point in pzip]
                                intersect=set(intersection_set).intersection(pzip)
                                if len(intersect):
                                    count2+=1
                            if count1>4 or count2>4:
                                nline=Lin2(comp_no,clines=[closest_line])
                                if len(nline.lorw):
                                    lines.append(nline)
                            else:
                            
                                if len(candidates)>1:
                                    clines_indices=list(zip(*candidates[1:len(candidates)]))[0]
                                    clines=[line for line_no,line in enumerate(lines) if line_no in clines_indices]
                                    lines[closest_line_index].add_lorw(comp_no,clines=clines)
                                else: lines[closest_line_index].add_lorw(comp_no)
                        
                break
            
    for line_no,line in enumerate(lines):
        if line==None: continue
        count=np.sum(line.img)
        for lno,l in enumerate(lines):
            if l==None: continue
            if count<0.05*np.sum(l.img):
                puncts.extend(line.puncts+line.lorw)
                lines[line_no]=None
                break
    lines=[line for line in lines if line!=None]
    for line in lines:
        line.calculate_median()
    #print_lines2(lines)
    lines=sorted(lines,key=lambda x:np.sum(x.img),reverse=True)
    #lines=sorted(lines,key=lambda x:(np.median(x.regressors,axis=0))[1]) 
    lines=[line for line in lines if line!=None and np.sum(line.img)]
    lines=combine_lines3(lines)
    #print_lines2(lines)
    for comp_no in puncts:
        lno,l=min(enumerate(lines),key=lambda x:np.mean([abs(global_vars.comp_dict[comp_no].centroid[1]-global_vars.comp_dict[cno].centroid[1]) for cno in x[1].lorw]))
        lines[lno].add_puncts(comp_no)
    for line in lines:
        line.calculate_median(typ=True)
    #print_lines2(lines)
    return lines
            
def func(x,a,b,c,d,e):
    return a+b*(c**(d*x+e))

def gauss(x, A,mu,sigma):
    return A*np.exp(-(x-mu)**2/(2*sigma**2))
            
def print_lines2(lines):
    lines_fig=mpl.pyplot.figure(global_vars.figno)
    for line_no,line in enumerate(lines):
        lines_fig.add_subplot(len(lines),1,line_no+1)
        temp_median_line=[x-line.top for x in line.median_line]
        mpl.pyplot.plot(list(range(len(temp_median_line))),temp_median_line)
        #print (line.lorw+line.puncts)
        mpl.pyplot.imshow(line.img,'gray')
    global_vars.figno+=1

def combine_lines3(lines):
    while True:
        prev_lines=lines.copy()
        for line_no,line in enumerate(lines):
            if line!=None:
                line_range=list(range(line.left,line.right))
                line_white_range=[index+line.left for index,item in enumerate(np.transpose(line.img)) if sum(item)]
                candidates=[]
                for lno, l in enumerate(lines):
                    if l!=None and lno!=line_no:
                        new_line_range=list(range(l.left,l.right))
                        overlap=set(new_line_range).intersection(line_range)
                        if len(overlap):
                            dist=[line.median_line[column_no-line.left]-l.median_line[column_no-l.left] for column_no in overlap]
                            ht_gap=abs(np.mean(dist))
                            #print (ht_gap,line.mean_ht,l.mean_ht)
                            if ht_gap<line.mean_ht or ht_gap<l.mean_ht:
                                l_white_range=[index+l.left for index,item in enumerate(np.transpose(l.img)) if sum(item)]
                                intersection_range=set(line_white_range).intersection(l_white_range)
                                if len(intersection_range)<0.33*len(line_white_range) or len(intersection_range)<0.33*len(l_white_range):
                                    candidates.append([lno,ht_gap])
                        elif l.left>line.right:
                            ht_gap=abs(l.median_line[0]-line.median_line[-1])
                            if ht_gap<line.mean_ht or ht_gap<l.mean_ht:
                                    candidates.append([lno,ht_gap])
                if len(candidates):
                    lno,hg=min(candidates,key=lambda x:x[1])   
                    l=lines[lno]
                    
                    for comp_no in l.lorw+l.puncts:
                        if global_vars.comp_dict[comp_no].height<0.33*lines[line_no].mean_ht and global_vars.comp_dict[comp_no].type<2:
                            lines[line_no].add_puncts(comp_no)
                        else:
                            lines[line_no].add_lorw(comp_no,inspect=False)
                    lines[lno]=None
                    lines[line_no].calculate_median()

        lines=[line for line in lines if line!=None]
        if np.array_equal(prev_lines,lines): break
    return lines
                                     
                                