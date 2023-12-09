import numpy as np
from numpy.linalg import inv

def getBlock(i,j):
    return M[i*BSIZE:(i+1)*BSIZE,j*BSIZE:(j+1)*BSIZE]

def BRIA(k,r,c):  
    if k > 2:            
        A = BRIA(k-1,r[:k-1],c[:k-1]) 
        B = BRIB(k-1,r[:k-1],c[1:k])  
        C = BRIC(k-1,r[1:k],c[:k-1]) 
        D = BRID(k-1,r[1:k],c[1:k]) 
    else: 
        A = getBlock(r[0],c[0])    
        B = getBlock(r[0],c[1])
        C = getBlock(r[1],c[0])
        D = getBlock(r[1],c[1]) 
    return A-B@inv(D)@C 
    

def BRIB(k,r,c):    
    if k > 2:
        c[0], c[1] = c[1], c[0]     
        A = BRIA(k-1,r[:k-1],c[:k-1]) 
        B = BRIB(k-1,r[:k-1],c[1:k])  
        C = BRIC(k-1,r[1:k],c[:k-1]) 
        D = BRID(k-1,r[1:k],c[1:k]) 
    else: 
        A = getBlock(r[0],c[0])    
        B = getBlock(r[0],c[1])
        C = getBlock(r[1],c[0])
        D = getBlock(r[1],c[1])                
    return B-A@inv(C)@D      
    
    
def BRIC(k,r,c):   
    if k > 2:     
        r[0], r[1] = r[1], r[0]      
        A = BRIA(k-1,r[:k-1],c[:k-1]) 
        B = BRIB(k-1,r[:k-1],c[1:k])  
        C = BRIC(k-1,r[1:k],c[:k-1]) 
        D = BRID(k-1,r[1:k],c[1:k]) 
    else: 
        A = getBlock(r[0],c[0])    
        B = getBlock(r[0],c[1])
        C = getBlock(r[1],c[0])
        D = getBlock(r[1],c[1])       
    return C-D@inv(B)@A  


def BRID(k,r,c): 
    if k > 2:     
        r[0], r[1] = r[1], r[0] 
        c[0], c[1] = c[1], c[0]     
        A = BRIA(k-1,r[:k-1],c[:k-1]) 
        B = BRIB(k-1,r[:k-1],c[1:k])  
        C = BRIC(k-1,r[1:k],c[:k-1]) 
        D = BRID(k-1,r[1:k],c[1:k])  
    else: 
        A = getBlock(r[0],c[0])    
        B = getBlock(r[0],c[1])
        C = getBlock(r[1],c[0])
        D = getBlock(r[1],c[1]) 
    return D-C@inv(A)@B  


M = np.array(
    [   [ -6.6164911,  7.7555376,  15.739762,   12.443051,    9.5297758,   5.5297085,  8.2233156,  20.273332  ],
        [  7.3808367, -7.4988944,   3.5466968, -14.417208,   -5.8342174,  -6.2095338,  9.1451267,   4.2401619 ],
        [ -0.8848029,  5.2416103,  -9.3273577,  -5.7796722,  -8.2392579, -12.874077,  28.266681,   -6.123134  ],
        [-16.086774,  10.520307,    1.6987682, -27.759125,  -12.772844,    8.9151562, -0.7283695,   8.6290828 ],
        [ -5.0198685,  0.7605452, -14.619521,   -4.6371648,  -2.8886551,  -5.5947374, 7.543045,    -5.7730659 ], 
        [ -6.0044864,  4.3533879,  -7.2622331,   5.6943339,  12.32391,    -2.1097907, -1.3301313,  10.741226  ],
        [ -1.3439234, -0.8219373,  15.027612,   21.320067,    4.2687133,   7.8869843,  1.6135328, -20.894218  ],
        [ -0.8905257,  8.088748,    5.249849,   -0.4815748,  -6.4609905,   15.086619, -16.21103,    0.9530342 ]
    ]
)    

BSIZE = 2
r = np.arange(M.shape[0])    
c = np.arange(M.shape[0])  
k =  int(M.shape[0]/BSIZE) 
ir, ic = (0,2)
r[0], r[ir] = r[ir], r[0] 
c[0], c[ic] = c[ic], c[0]
N11 = inv(BRIA(k,r,c))  

print('M')    
print(M)   
print('inv(M)')
print(inv(M))
print('N11')
print(N11)
