from sklearn.cluster import KMeans
import numpy as np

def ngram_generator (arr, n) : 
    ngram=[]
    for i in range(len(arr) - n +1 ) :
        ar = arr[i:i+n]
        ngram.append(','.join(ar))
    return ngram    


def ngram_arr_generator(VARIANT) : 
    ngram_arr=[]
    for v in VARIANT :
        ngram = ngram_generator(v.split(','),2)
        ngram = list(set(ngram)) #to remove duplications
        ngram_arr.append(ngram)
    return ngram_arr

def feature_extractor (ngram_arr) :
    features = []
    for g in ngram_arr : 
        features += g
    features=list(set(features))
    return features
    
    
def fecture_vectors_creator (ngram_arr, features)  : 
    feature_vectors = []
    for ngram in ngram_arr : 
        feature = [0] * len(features)
        for n in ngram : 
            feature[features.index(n)] = 1
        feature_vectors.append(feature)
    return feature_vectors



def ngram_kmean (VARIANT) : 

    ngram_arr = ngram_arr_generator(VARIANT)
    features = feature_extractor(ngram_arr)
    feature_vectors = fecture_vectors_creator (ngram_arr, features)

    data = np.array(feature_vectors)
    kmeans = KMeans(n_clusters=5, random_state=0).fit(data)

    return kmeans