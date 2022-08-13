from asyncio.windows_events import INFINITE, NULL
from math import inf
from cv2 import kmeans
import numpy as np
import random as rnd
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets    
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score as ARI
from sklearn.metrics import pairwise_distances as outer

def show2D(X,P):
    plt.scatter(X[:,0],X[:,1],c=P,s=12)
    plt.show()

def show3D(X,P):
    ax=plt.axes(projection ="3d")
    ax.scatter3D(X[:,0],X[:,1],X[:,2],c=P,s=12)
    plt.show()
    
def show_pcord(X,P):
    _,d=X.shape
    df=pd.DataFrame(X,columns=[a+str(b) for a,b in zip(['x']*d,list(range(1,d+1)))])
    df['Cluster']=P
    pd.plotting.parallel_coordinates(df,'Cluster')
    plt.show()
    
def init(X,k):
    N,_=X.shape
    U=[np.random.multinomial(1, [1/k]*k) for _ in range(N)]
    U=np.array(U)
    return U

def kernel(x,y,bandwidth=1,sigma=5):      #computes the distance is mapped hilbert space
    return np.exp(-np.linalg.norm(x-y)**2*bandwidth/(2*sigma**2))

def dist(U,K1,K2=None):    
    if K2 is None:
        K2=K1
    n,k=U.shape
    m,_=K2.shape
    S=U.sum(axis=0)             #column sum of membership matrix U
    t1=np.full((m,k),K1[0,0])                 #1st term 
    t2=-2*np.dot(K2,U)/S                        #2nd term
    t3=np.array([np.diag(U.T@K1@U)]*m)/(S**2)     #3rd term
    return t1+t2+t3             #return the distance matrix from the k Higher dim centroids of Y
    
def cost(X,U):
    n,k=U.shape
    
    K=np.array([[kernel(X[i]-X[j]) for j in range(n)] for i in range(n)])
    return np.matrix.trace((U.T)@K@U)

def update(U,K):
    D=dist(U,K)
    I=D.argmin(axis=1)
    U=np.zeros(U.shape)
    U[[*range(U.shape[0])],I]=1
    return U,D.sum()

def kernel_kmeans(K,X,k,plot=1,max_rep=1,max_iters=100,tol=1e-32):
    # print("Initial Clusters:-")
    # show2D(X,U.argmax(axis=1))
    mU=0
    merror=1e20
    for i in range(max_rep):
        U=init(X,k)
        lerror=inf
        for _ in range(max_iters):
            U,perror=update(U,K)     #Updates the membership matrix
            if np.abs(perror-lerror)<tol:    #stop if clusters stabilize 
                print(f'Iterations:{_}')
                break
            lerror=perror
        if merror>lerror:
            mU,merror=U,lerror
    P=mU.argmax(axis=1)
    if plot:
        print("Final Clusters:-")
        if X.shape[1]==2:
            show2D(X,P)
        elif X.shape[1]==3:
            show3D(X,P)
        else:
            show_pcord(X,P)
    return P,mU

def predict(K1,X,U,Y):
    K2=outer(Y,X,metric=kernel)
    D=dist(U,K1,K2)
    P=D.argmin(axis=1)
    Z=np.zeros([Y.shape[0],U.shape[1]])
    Z[[*range(Z.shape[0])],P]=1
    return P,Z
    
def make_blobs(C=[[1,2],[5,6],[10,4]],S=[[[1,0],[0,2]],[[3,0],[0,4]],[[3,0],[0,1]]],N=[60,80,100]):
    X=np.empty([0,2])
    for i in range(len(N)):
        x=np.random.multivariate_normal(C[i],S[i],N[i])
        X=np.append(X,x,axis=0)
    return X
    
def make_rings(C=[[0]*2]*2,R=[10,20],N=[100,200],noise=2):
    X=np.empty([0,2])
    for i in range(len(N)):
        theta=np.linspace(0,2*np.pi,num=N[i])
        x=[C[i]]*N[i]+R[i]*np.stack((np.cos(theta),np.sin(theta)),axis=1)+np.random.multivariate_normal([0]*2,np.diag([noise]*2),N[i])
        X=np.append(X,x,axis=0)
    return X

def make_moons(C=[[0]*2,[20,0]],R=[20,20],N=[200,200],A=[[-0.1,np.pi*1.1],[np.pi*0.9,2.1*np.pi]],noise=2):
    X=np.empty([0,2])
    for i in range(len(N)):
        theta=np.linspace(A[i][0],A[i][1],num=N[i])
        x=[C[i]]*N[i]+R[i]*np.stack((np.cos(theta),np.sin(theta)),axis=1)+np.random.multivariate_normal([0]*2,np.diag([noise]*2),N[i])
        X=np.append(X,x,axis=0)
    return X
    
def main():
    # # idx=np.random.rand(X.shape[0]).argsort()
    # # X=np.take(X,idx,axis=0)
    # L=[0]*60+[1]*80+[2]*100
    # # L=list(np.take(L,idx,axis=0))
    # #######################################Iris Dataset######################################
    # iris=datasets.load_iris()
    # X=iris.data
    # L=list(iris.target)
    # #################################Test Run#######################################
    # X=generate([[0]*2]*2,[10,20],[100,200])
    # L=[0]*40+[1]*60
    # n=60+80+100
    # k=3
    # P,U=kernel_kmeans(X,k)
    # show_pcord(X,P)
    # show_pcord(X,L)

    # # print(Z)
    # print(f'L:-\n{L}')
    # print(f'P:-\n{P}')
    # print(f"Adjusted Random Index is {ARI(L,P)}")
    # Z=np.array([[1 if L[i]==j else 0 for j in range(k) ] for i in range(n)])
    # print(Z)
    # print(dist(X,Z,X))
    # ############################Determining the no. of clusters#########################
    # # ari=[]
    # # for k in range(2,4):
    # #     m,_=soft_kmeans(X,k)
    # #     p=[predict(m,X[i]) for i in range(X.shape[0])]
    # #     ari.append(ARI(L,p))
    # # print(ari)
    # # plt.plot(range(10),ari)
    # # plt.show()
    # #####################################Using In-built function###########################
    # # model=kmeans.fit(X)
    # # fig,ax1,ax2=plt.subplot(1,2)
    # df2=pd.DataFrame(X,columns=[a+str(b) for a,b in zip(['x']*4,list(range(1,4+1)))])
    # df2['Cluster']=P
    # plt.subplot(1,2,2)
    # pd.plotting.parallel_coordinates(df2,'Cluster')
    # df1=pd.DataFrame(iris.data,columns=iris.feature_names)
    # df1['Flowers']=[iris.target_names[0] if i==0 else iris.target_names[1] if i==1 else iris.target_names[2] for i in iris.target]
    # plt.subplot(1,2,1)
    # pd.plotting.parallel_coordinates(df1,'Flowers')
    # plt.show()
    ######################################Ring Dataset####################################
    # X=make_rings([[0]*2]*2,[10,20],[500,1000])
    X=make_moons()
    # X=make_blobs()
    L=[0]*200+[1]*200
    n=400
    k=2
    K=outer(X,metric=kernel)
    mari=P=-inf
    for i in range(100):
        Z,U=kernel_kmeans(K,X,k,plot=0)
        ari=ARI(L,Z)
        print(f'ARI:{ari}')
        if ari>mari:
            P,mari=Z,ari
    # Z=np.array([[1 if P[i]==j else 0 for j in range(k) ] for i in range(n)])
    # show_pcord(X,P)
    # show_pcord(X,L)
    # show2D(X,Z)
    # print(predict(X,Z,X))
    # print(f'M:-\n{U}')
    # print(f'P:-\n{P}')
    # print(f'L:-\n{L}')
    show2D(X,P)
    print(f"Adjusted Random Index is {ARI(L,P)}")
    Y=np.array([[-20,5],[-10,15]])
    L,_=predict(K,X,U,Y)
    print(f"Predictions:-\n{L}")

if __name__=="__main__":
    main()