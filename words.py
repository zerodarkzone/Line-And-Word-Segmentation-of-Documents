import scipy
from sklearn import mixture
import numpy as np

def segment_words(lines):
    words=[]

    for line in lines:
        distances=[]
        for comp_no, comp in enumerate(line.new_comps[0:len(line.new_comps)-1]):
            next_comp=line.new_comps[comp_no+1]
            min_distance=min(scipy.spatial.distance_matrix(comp[2],next_comp[1]).flatten())
            distances.append([min_distance])
        distances=np.sort(distances)
        if len(distances)>1:
            if len(distances)>10:
                temp_distances=distances[0:len(distances)-2]
            else: temp_distances=distances.copy()
            mix=mixture.GMM(n_components=2)
            mix.fit(temp_distances)
            lower_label=list(mix.means_).index(min(list(mix.means_)))
            labels=list(mix.predict(distances))
            for label_no,label in enumerate(labels):
                    if label==lower_label:
                        if line.new_comps[label_no]!=None:
                            line.new_comps[label_no+1][0].extend(line.new_comps[label_no][0])
                            line.new_comps[label_no]=None
        words.extend([comp[0] for comp in line.new_comps if comp!=None])
    return words