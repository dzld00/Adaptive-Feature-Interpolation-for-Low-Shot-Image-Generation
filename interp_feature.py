### Adaptive Feature Interpolation
# create a set of new features from old features 
# new_feature = near_interp(old_feature, k, augment_prob)
# k, augment_prob can be generated by function "dynamic_prob" or defined by user 

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.manifold import MDS
import random

def near_interp(embeddings, k, augment_prob):
    if k == 1 or augment_prob == 0:
        return embeddings

    k = min(k, embeddings.size()[0])

    pd = pairwise_distances(embeddings, embeddings)
    pd = pd/pd.max()
    pd_s = (1 / (1+pd))
    
    # Select top k near neighbours
    k_smallest = torch.topk(pd, k, largest=False).indices # shape: batch_size x k 

    # Feature interpolation    
    t = 1
    alpha = torch.ones(k, device=embeddings.device)
    inner_embeddings = []  
    for row in k_smallest:
        for i in range(k):                   
            alpha[i] = pd_s[row[0],row[i]]**t
                
        p = torch.distributions.dirichlet.Dirichlet(alpha).sample().to(embeddings.device)
        
        inner_pts = torch.matmul(p.reshape((1,-1)),embeddings.index_select(0,row))
        inner_embeddings.append(F.normalize(inner_pts))
    
    batch_size = embeddings.size()[0]    
    out_embeddings = []
    
 
    # Output interpolated feature with probability p 
    for idx in range(batch_size):
        p = random.random()
        if p < augment_prob:
            out_embeddings.append(inner_embeddings[idx])
        else:
            out_embeddings.append(embeddings[idx,:].unsqueeze(0))
        
    return torch.stack(out_embeddings).reshape((batch_size,-1))


def dynamic_prob(embeddings):
    embeddings = F.normalize(embeddings)
    batch_size = embeddings.size()[0] 
    
    D = pairwise_distances(embeddings, embeddings)   
    D = D.detach().cpu().numpy()  
    D = D / np.amax(D)
    
    #l_sorted = cmdscale(D) 
    l_sorted = eigen_mds(D)               
    
    # Calculate k,p based on number of large eigenvalues
    k = batch_size - next(x[0] for x in enumerate(l_sorted) if x[1] < 0.1 * l_sorted[0])    
    p = (k-1) / batch_size

    #k = 2
    #p = 0.9
    
    return p, k  


def cmdscale(D):
    """                                                                                       
    Classical multidimensional scaling (MDS)                                                  
                                                                                               
    Parameters                                                                                
    ----------                                                                                
    D : (n, n) array                                                                          
        Symmetric distance matrix.                                                            
                                                                                               
    Returns                                                                                   
    -------                                                                                   
    Y : (n, p) array                                                                          
        Configuration matrix. Each column represents a dimension. Only the                    
        p dimensions corresponding to positive eigenvalues of B are returned.                 
        Note that each dimension is only determined up to an overall sign,                    
        corresponding to a reflection.                                                        
                                                                                               
    e : (n,) array                                                                            
        Eigenvalues of B.                                                                     
                                                                                               
    """
    
    # Number of points                                                                        
    n = len(D)
 
    # Centering matrix                                                                        
    H = np.eye(n) - np.ones((n, n))/n
 
    # YY^T                                                                                    
    B = -H.dot(D**2).dot(H)/2
 
    # Diagonalize                                                                             
    evals, evecs = np.linalg.eigh(B)
 
    # Sort by eigenvalue in descending order                                                  
    idx   = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:,idx]
 
    # Compute the coordinates using positive-eigenvalued components only                      
#     w, = np.where(evals > 0)
#     L  = np.diag(np.sqrt(evals[w]))
#     V  = evecs[:,w]
#     Y  = V.dot(L)
 
    return np.sort(evals)[::-1]


def eigen_mds(pd):   
    mds = MDS(n_components=len(pd), dissimilarity='precomputed')
    pts = mds.fit_transform(pd)

    _,l_sorted,_ = np.linalg.svd(pts)
    
    return l_sorted


def pairwise_distances(x, y):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x**2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)
    
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    # Ensure diagonal is zero if x=y
    # if y is None:
    #     dist = dist - torch.diag(dist.diag)
    return torch.sqrt(torch.clamp(dist, 0.0, np.inf))   
