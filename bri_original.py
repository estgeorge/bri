import numpy as np
from numpy.linalg import inv

def getBlock(p,q):
    return M[p*BSIZE:(p+1)*BSIZE,q*BSIZE:(q+1)*BSIZE]
    
def show_info(trace,i,j):
    global r,c
    #print('Node {0}'.format("".join(trace)));
    #print(f'r:{r}')
    #print(f'c:{c}')
    pass 
    
def show_shur(i,j,tipo):
    global r,c
    if tipo=='A':
        print('{0} {1},{2} - {3},{4} * {5},{6}^(-1) * {7},{8}'.format("".join(trace),r[i],c[j],r[i],c[j+1],r[i+1],c[j+1],r[i+1],c[j]));
    elif tipo=='B':
        print('{0} {1},{2} - {3},{4} * {5},{6}^(-1) * {7},{8}'.format("".join(trace),r[i], c[j+1], r[i], c[j], r[i+1], c[j], r[i+1], c[j+1]));    
    elif tipo=='C':
        print('{0} {1},{2} - {3},{4} * {5},{6}^(-1) * {7},{8}'.format("".join(trace),r[i+1], c[j], r[i+1], c[j+1], r[i], c[j+1], r[i], c[j])); 
    elif tipo=='D':    
        print('{0} {1},{2} - {3},{4} * {5},{6}^(-1) * {7},{8}'.format("".join(trace),r[i+1], c[j+1], r[i+1], c[j], r[i], c[j], r[i], c[j+1]));
    pass    
           
def BRIA(k,i,j,trace):
    global r,c
    trace.append("A")
    show_info(trace,i,j)
    if k > 2:
        B = BRIB(k-1,i,j+1,trace)
        D = BRID(k-1,i+1,j+1,trace)
        A = BRIA(k-1,i,j,trace)        
        C = BRIC(k-1,i+1,j,trace)        
    else:
        A = getBlock(r[i],c[j])
        B = getBlock(r[i],c[j+1])
        C = getBlock(r[i+1],c[j])
        D = getBlock(r[i+1],c[j+1])        
        show_shur(i,j,"A")  
        print(A-B@inv(D)@C)        
    trace.pop()        
    return A-B@inv(D)@C

def BRIB(k,i,j,trace):
    global r,c
    trace.append("B")
    show_info(trace,r,c)
    if k > 2:
        c[j], c[j+1] = c[j+1], c[j]              
        A = BRIA(k-1,i,j,trace)
        C = BRIC(k-1,i+1,j,trace)        
        B = BRIB(k-1,i,j+1,trace)
        D = BRID(k-1,i+1,j+1,trace)
        c[j+1], c[j] = c[j], c[j+1]  
    else:
        A = getBlock(r[i],c[j])
        B = getBlock(r[i],c[j+1])
        C = getBlock(r[i+1],c[j])
        D = getBlock(r[i+1],c[j+1])
        show_shur(i,j,"B") 
    trace.pop()       
    return B-A@inv(C)@D


def BRIC(k,i,j,trace):
    global r,c
    trace.append("C")
    show_info(trace,r,c)
    if k > 2:
        r[i], r[i+1] = r[i+1], r[i]
        D = BRID(k-1,i+1,j+1,trace)        
        B = BRIB(k-1,i,j+1,trace)
        C = BRIC(k-1,i+1,j,trace)
        A = BRIA(k-1,i,j,trace)   
        r[i+1], r[i] = r[i], r[i+1]
    else:
        A = getBlock(r[i],c[j])
        B = getBlock(r[i],c[j+1])
        C = getBlock(r[i+1],c[j])
        D = getBlock(r[i+1],c[j+1])
        show_shur(i,j,"C")   
    trace.pop()       
    return C-D@inv(B)@A


def BRID(k,i,j,trace):
    global r,c
    trace.append("D")
    show_info(trace,r,c)
    if k > 2:
        r[i], r[i+1] = r[i+1], r[i]
        c[j], c[j+1] = c[j+1], c[j]
        C = BRIC(k-1,i+1,j,trace)
        A = BRIA(k-1,i,j,trace)
        D = BRID(k-1,i+1,j+1,trace)        
        B = BRIB(k-1,i,j+1,trace)
        c[j+1], c[j] = c[j], c[j+1]
        r[i+1], r[i] = r[i], r[i+1]
    else:
        A = getBlock(r[i],c[j])
        B = getBlock(r[i],c[j+1])
        C = getBlock(r[i+1],c[j])
        D = getBlock(r[i+1],c[j+1])
        show_shur(i,j,"D")           
    trace.pop()        
    return D-C@inv(A)@B


M = np.array([
        [-6.6164911, 7.7555376, 15.739762, 12.443051, 9.5297758, 5.5297085, 8.2233156, 20.273332, 7.3808367, -7.4988944],
        [3.5466968, -14.417208, -5.8342174, -6.2095338, 9.1451267, 4.2401619, -0.8848029, 5.2416103, -9.3273577, -5.7796722],
        [-8.2392579, -12.874077, 28.266681, -6.123134, -16.086774, 10.520307, 1.6987682, -27.759125, -12.772844, 8.9151562],
        [-0.7283695, 8.6290828, -5.0198685, 0.7605452, -14.619521, -4.6371648, -2.8886551, -5.5947374, 7.543045, -5.7730659],
        [-6.0044864, 4.3533879, -7.2622331, 5.6943339, 12.32391, -2.1097907, -1.3301313, 10.741226, -1.3439234, -0.8219373],
        [15.027612, 21.320067, 4.2687133, 7.8869843, 1.6135328, -20.894218, -0.8905257, 8.088748, 5.249849, -0.4815748],
        [-6.4609905, 15.086619, -16.21103, 0.9530342, -16.157583, 2.842335, 2.023021, -9.3329297, -11.858605, 6.9711633],
        [7.1077842, -9.5678636, -2.7781823, 12.456173, 26.749105, -4.0055527, -3.5752841, 6.6231511, -5.4532615, -2.6478337],
        [1.1163911, 3.2327364, -13.344473, 18.36382, 18.487359, 19.651792, 2.6459116, -9.3589099, 10.182207, -8.355485],
        [1.2591184, 3.7204043, 8.9537031, -15.81167, -12.770621, 8.0427309, -8.0637384, 3.634747, 13.296215, -10.579022]
    ])

BSIZE = 2
trace = []
k =  int(M.shape[0]/BSIZE)
r = np.arange(k)
c = np.arange(k)
N11 = inv(BRIA(k,0,0,trace))

print('inv(M)')
print(inv(M)[0:BSIZE,0:BSIZE])
print('N11')
print(N11)
