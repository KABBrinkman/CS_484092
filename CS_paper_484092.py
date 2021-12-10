#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 15:25:45 2021

@author: kikibrinkman
"""
import json
import numpy as np
from random import shuffle
import pandas as pd
from sklearn.utils import resample
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

with open("TVs-all-merged.json") as jsonFile:
    jsonObject = json.load(jsonFile)
    jsonFile.close()
    
#1262 unique products of all 1624 
pd=pd.DataFrame.from_dict(jsonObject, orient='index')

products=[]
for i in range(0,len(pd)):
    products.append(pd.iloc[i,0])
    if pd.iloc[i,1]!=None:
        products.append(pd.iloc[i,1])
        if pd.iloc[i,2]!=None:
            products.append(pd.iloc[i,2])
            if pd.iloc[i,3]!=None:
                products.append(pd.iloc[i,3])
                
#Preprocess lowercase, hertz and inch.
for i in range(0,len(products)):
    products[i]['title']=products[i]['title'].lower()
    if 'hertz' in products[i]['title']:
       products[i]['title']=products[i]['title'].replace('hertz','hz')
    if '-hz' in products[i]['title']:
       products[i]['title']=products[i]['title'].replace('-hz','hz')
    if ' hz' in products[i]['title']:
       products[i]['title']=products[i]['title'].replace(' hz','hz')
    if '-inch' in products[i]['title']:
       products[i]['title']=products[i]['title'].replace('-inch','"')
    if 'in' in products[i]['title']:
       products[i]['title']=products[i]['title'].replace('in','"')
    if 'inches' in products[i]['title']:
       products[i]['title']=products[i]['title'].replace('inches','"') 
    if 'inch' in products[i]['title']:
       products[i]['title']=products[i]['title'].replace('inch','"')
    if ' "' in products[i]['title']:
       products[i]['title']=products[i]['title'].replace(' "','"')
       
pq_bootstrap=[]
pc_bootstrap=[]
fb_bootstrap=[]
frac_comp_bootstrap=[]
#Perform 5 bootstraps
for a in range(5):
    train=resample(range(0, len(products)), replace=True, n_samples=len(products))
    train=set(train)
    trainset=[]
    testset=[]
    for i in range(len(products)):
        if i in train:
            trainset.append(products[i])
        else:
            testset.append(products[i])        
                
    #Shingle
    def build_shingles(text: str, k: int):
        shingle_set = []
        for i in range(len(text) - k+1):
            shingle_set.append(text[i:i+k])
        return set(shingle_set)              
    
    #Build shingles
    k = 3 
    shingles = []
    for p in range(0,len(products)):
        shingles.append(list((build_shingles(products[p]['title'], k))))
    
    #Make vocab 
    def build_vocab(shingel):
        vocab = []
        for i in range(0,len(shingel)):
            vocab=vocab +(shingel[i])
        return set(vocab)
    
    vocab = build_vocab(shingles)
    
    #Make boolean shingles list
    shingles_1hot = []
    for i in range(len(shingles)):
        shingles_1hot.append([1 if x in shingles[i] else 0 for x in vocab])
            
    #Minhashing,convert sparse vectors into dense vectors.
    def create_hash(vocab_size):
        #permutation
        hash = list(range(1, vocab_size+1))
        shuffle(hash)
        return hash
    
    def minhash(vocab_size: int, nbits: int):
        # function for building multiple minhash vectors
        hashes = []
        for n in range(nbits):
            hashes.append(create_hash(vocab_size))
        return hashes
    
     #n_hash=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50] tried options
     #bands=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50] tried options
    fbs=[[15,5], [15,15], [20,5], [20,10], [20,20], [25,5],[25,25], [30,5],[30,10],[30,15], [35,5], [40,5], [40,10], [40,20], [45,5], [45,15],[50,5], [50,10]]
    eval_n=[]
    for fb in fbs:
         #Minhashing
         signatures=[]
             
         #Create 25 minhash vectors
         minhash_func = minhash(len(vocab), fb[0])
     
         #Use minhash functions for creating signatures
         def create_sign(vector):
             signature = []
             for func in minhash_func:
                 for i in range(1, len(vector)+1):
                     idx = func.index(i)
                     signature_val = vector[idx]
                     if signature_val == 1:
                        signature.append(idx)
                        break
             return signature
     
         signatures = []
         for shingle1hot in shingles_1hot:
             signatures.append(create_sign(shingle1hot))
         
         #Splitting signature in b parts
         def split_sign(signature, b):
             r = int(len(signature)/b)
             subvecs = []
             for i in range(0, len(signature),r):
                 subvecs.append(signature[i:i+r])
             return subvecs    
             
         #column pairs are those that hash to the same bucket for ≥ 1 band
         candidate_pairs=[]
         for j in range(len(signatures)):
             for i in range(j+1,len(signatures)):#j+1,
                 if j!=i:
                     for j_rows, i_rows in zip(split_sign(signatures[j],fb[1]), split_sign(signatures[i],fb[1])):
                         if j_rows == i_rows:
                             candidate_pairs.append([j,i])
                             break       
     
         def jaccard (x,y):
             return len(set(x).intersection(set(y)))/len(set(x).union(set(y)))
        
        #Classification
         Duplicates=[]
         for l in candidate_pairs:
             if (jaccard(shingles[l[0]],shingles[l[1]])>=0.6):
                 Duplicates.append(l)   
         
         """   
         #clustering
         from scipy.cluster.hierarchy import dendrogram, linkage
         from scipy.cluster.hierarchy import single, fcluster
         from scipy.spatial.distance import pdist
         from scipy.spatial.distance import pdist, squareform
         similairty=np.full((len(products), len(products)), 10.0)
         np.fill_diagonal(similairty, 0)
         for [i,j] in candidate_pairs:
             similairty[i,j]=(1-jaccard(products[i]['title'],products[j]['title']))
         similairty=np.triu(similairty, k=1)
         z=single(squareform(similairty))
         thresholds=[0.4]
         for th in thresholds:
             flust=fcluster(z, t=th, criterion='distance')
                  
             Duplicates=[]   
             for j in range(0, len(products)):
                 for i in range(j+1,len(products)):
                      if flust[i]==flust[j]:
                         Duplicates.append([j,i])
                     
             dup=[]
             for [j,i] in Duplicates:
                 if products[j]['modelID']==products[i]['modelID']:
                    dup.append([j,i])   
         
            
        #evaluate isolated LSH
        Duplicates=[]
        for [h,m] in candidate_pairs:
            if products[h]['modelID']==products[m]['modelID']:
               Duplicates.append([h,m])          
        """  
        #Retrieve ID's of flagged duplicates 
         lsh_duplicates=list(set([item for sublist in Duplicates for item in sublist]))
        
         lsh_duplicates_ID=[]
         for dup in lsh_duplicates:
             lsh_duplicates_ID.append(products[dup]['modelID'])
        
        
         def eval(products,candidate_pairs,lsh_duplicates_ID):    
            #Number of correctly identified duplicates    
            d_f=len(lsh_duplicates_ID)-len(list(set(lsh_duplicates_ID)))
            #Number of comparisons
            n_c=len(candidate_pairs)
            PQ=d_f/n_c   
            #Pair Completeness
            ids=[]
            for i in range(len(products)):
                ids.append(products[i]['modelID'])
            unique_ids=len(list(set(ids)))
            #Total amount of duplicates
            d_n=len(products)-unique_ids
            PC=d_f/d_n 
            F1=(2*PQ*PC)/(PQ+PC)
            #Fraction of Comparisons
            #Possible combinations
            n_t=(len(products)*(len(products)-1)/2)
            Frac_Comp=n_c/n_t
            return [fb,PQ,PC,F1,Frac_Comp]    
         
         eval_n.append(eval(products,candidate_pairs,lsh_duplicates_ID))
         
    import pandas as pd 
    fb_bootstrap.append(pd.DataFrame(eval_n).iloc[:,0])
    pq_bootstrap.append(pd.DataFrame(eval_n).iloc[:,1])                   
    pc_bootstrap.append(pd.DataFrame(eval_n).iloc[:,2])
    frac_comp_bootstrap.append(pd.DataFrame(eval_n).iloc[:,4])
    
import pandas as pd 
eval_n=pd.DataFrame(eval_n)
fb_bootstrap=pd.DataFrame(fb_bootstrap)
pq_bootstrap=pd.DataFrame(pq_bootstrap)
frac_comp_bootstrap=pd.DataFrame(frac_comp_bootstrap)
pc_bootstrap=pd.DataFrame(pc_bootstrap)

results=pd.DataFrame({'n_hash':pd.DataFrame(list(eval_n.iloc[:,0])).iloc[:,0], 'PC':pc_bootstrap.mean(), 'PQ':pq_bootstrap.mean(),'Frac_Comp':frac_comp_bootstrap.mean()})
results['b']=pd.DataFrame(list(eval_n.iloc[:,0])).iloc[:,1]
results['r']=results['n_hash']/results['b']
results['F1']=(2*results['PQ']*results['PC'])/(results['PQ']+results['PC'])
results['t']=(1/results['b'])**(1/results['r'])
  
results=results.sort_values('Frac_Comp')    
fig = plt.figure()
ax = plt.axes()
ax.plot(results['Frac_Comp'], results['PC']);
ax.set_xlabel('Fraction of comparisons' )
ax.set_ylabel('Pair completeness' )

fig = plt.figure()
ax = plt.axes()
ax.plot(results['Frac_Comp'], results['PQ']);
ax.set_xlabel('Fraction of comparisons' )
ax.set_ylabel('Pair quality' )

fig = plt.figure()
ax = plt.axes()
ax.plot(results['Frac_Comp'], results['F1']);
ax.set_xlabel('Fraction of comparisons' )
ax.set_ylabel('F1' )

