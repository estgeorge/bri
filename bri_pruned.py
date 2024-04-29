import os
import shutil
import numpy as np
from numpy.linalg import inv,det

def get_filename(op,ac,bc,cc,dc):
    return f'{op}-{ac}-{bc}-{cc}-{dc}'
    
def getBlock(p,q):
    global r,c
    return M[r[p]*BSIZE:(r[p]+1)*BSIZE,c[q]*BSIZE:(c[q]+1)*BSIZE]
    
def read_block(op,ac,bc,cc,dc,node):
    path = output_dir+"/"+get_filename(op,ac,bc,cc,dc)+".npy"
    if os.path.isfile(path):        
        S = np.load(path)         
        print("read  "+node.show_path()+op+" - "+path)
        os.remove(path) 
        return S
    else:    
        raise Exception("Bloco não encontrado")

def write_block(op,ac,bc,cc,dc,S,node):  
    path = output_dir+"/"+get_filename(op,ac,bc,cc,dc)
    np.save(path,S)      
    print("write "+node.show_path()+" - "+path+".npy")
  
   
def getBlockInv(A,p,q):
    global r,c
    return A[c[p]*BSIZE:(c[p]+1)*BSIZE,r[q]*BSIZE:(r[q]+1)*BSIZE]    
   
def BRI(k,i,j,parent,op,ac,bc,cc,dc): 

    # node = BriTree(parent,op)  
    # node.info = get_filename(op,ac,bc,cc,dc)    
    # if op == "A":
        # parent.childA = node
    # elif op == "B":
        # parent.childB = node
    # elif op == "C":
        # parent.childC = node
    # elif op == "D":
        # parent.childD = node   
         
    if k > 2:
        if op > "A": 
            A = read_block("A",ac+1,bc,cc,dc,node) 
        else:        
            A = BRI(k-1,i,j,node,"A",ac+1,bc,cc,dc)      
        if op > "B":             
            B = read_block("B",ac,bc+1,cc,dc,node)  
        else:
            B = BRI(k-1,i,j+1,node,"B",ac,bc+1,cc,dc)                 
        if op > "C":      
            C = read_block("C",ac,bc,cc+1,dc,node) 
        else:
            C = BRI(k-1,i+1,j,node,"C",ac,bc,cc+1,dc)                   
        D = BRI(k-1,i+1,j+1,node,"D",ac,bc,cc,dc+1) 
    else:    
        A = getBlock(i,j)    
        B = getBlock(i,j+1)  
        C = getBlock(i+1,j)  
        D = getBlock(i+1,j+1)    
                   
    if op == "A":
        SA = A-B@inv(D)@C      
        return SA     
        
    if ac >= 2:   
        SA = A-B@inv(D)@C 
        write_block("A",ac,bc,cc,dc,SA,node)                
           
    if op == "B": 
        SB = B-A@inv(C)@D     
        return SB  

    if bc >= 1: 
        SB = B-A@inv(C)@D 
        write_block("B",ac,bc,cc,dc,SB,node)  
                    
    if op == "C": 
        SC = C-D@inv(B)@A    
        return SC

    if cc >= 1: 
        SC = C-D@inv(B)@A
        write_block("C",ac,bc,cc,dc,SC,node)         
        
    SD = D-C@inv(A)@B
    return SD         
               
  
    
M10 = np.array([
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

M8 = np.array(
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

M6 = np.array(
    [   [-6.6164911, 7.7555376, 15.739762, 12.443051, 9.5297758, 5.5297085],
        [7.3808367, -7.4988944, 3.5466968, -14.417208, -5.8342174, -6.2095338],
        [-0.8848029, 5.2416103, -9.3273577, -5.7796722, -8.2392579, -12.874077],
        [-16.086774, 10.520307, 1.6987682, -27.759125, -12.772844, 8.9151562],
        [-5.0198685, 0.7605452, -14.619521, -4.6371648, -2.8886551, -5.5947374],
        [-6.0044864, 4.3533879, -7.2622331, 5.6943339, 12.32391, -2.1097907]
    ]
)

M4 = np.array(
    [   [-6.6164911, 7.7555376, 15.739762, 12.443051],
        [7.3808367, -7.4988944, 3.5466968, -14.417208],
        [-0.8848029, 5.2416103, -9.3273577, -5.7796722],
        [-16.086774, 10.520307, 1.6987682, -27.759125]
    ]
)

class BriTree:

    def __init__(self,parent,op):   
        self.op = op
        self.parent = parent
        self.childA = None 
        self.childB = None 
        self.childC = None 
        self.childD = None
        self.info = None
        self.SA = None 
        self.SB = None 
        self.SC = None 
        self.SD = None 
        
    def children(self):
        lc = [] 
        if self.childA is not None:
            lc.append(self.childA)
        if self.childB is not None:
            lc.append(self.childB)
        if self.childC is not None:
            lc.append(self.childC)
        if self.childD is not None:
            lc.append(self.childD)
        return lc    

    def show_path(self):
        p = self
        path = []
        while p is not None:
            path.append(p.op)
            p = p.parent
        path.reverse()    
        return "".join(path)    


def conta_tree(tree):
    if tree == None:
        return 0;
    else:
        i = 1
        for c in tree.children():
            i = i + conta_tree(c)
        return i


M = np.loadtxt('matrix12.txt')
#M = np.loadtxt('matrix20.txt')
#M = M8
#M = M10

output_dir = "temp"
if os.path.isdir(output_dir):
    shutil.rmtree(output_dir)
os.mkdir(output_dir)

BSIZE = 2
k =  int(M.shape[0]/BSIZE)
r = np.arange(k)
c = np.arange(k)
br,bc = (1,2)
r[0],r[br] = r[br],r[0]
c[0],c[bc] = c[bc],c[0]
tree = BriTree(None,"A")
S = BRI(k,0,0,tree,'A',1,0,0,0)

os.system("dir temp")

N11 = inv(S)
print('inv(M)')
print(getBlockInv(inv(M),0,0))
print('N11')
print(N11)

# from PrettyPrint import PrettyPrintTree
# print()
# pt = PrettyPrintTree(lambda x: x.children(), lambda x: x.info, orientation=PrettyPrintTree.Horizontal)
# pt = PrettyPrintTree(lambda x: x.children(), lambda x: x.op)
# pt(tree.childA)
# print()

print(f"k={k} num_nodes={conta_tree(tree.childA)}") 



