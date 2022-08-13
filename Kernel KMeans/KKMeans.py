from cv2 import kmeans
import numpy as np
import random as rnd
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets    
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score as ARI

def show2D(X,U):
    colors=np.random.random((U.shape[1],3))  #creates a random numpy array of shape[k,3] filled with iid unif(0,1) rvs
    colors=U.dot(colors)    #we take a dot product of the colour matrix with the membership matrix
    plt.scatter(X[:,0],X[:,1],c=colors,s=10)
    plt.show()

def show3D(X,U):
    colors=np.random.random((U.shape[1],3))  #creates a random numpy array of shape[k,3] filled with iid unif(0,1) rvs
    colors=U.dot(colors)    #we take a dot product of the colour matrix with the membership matrix
    ax=plt.axes(projection ="3d")
    ax.scatter3D(X[:,0],X[:,1],X[:,2],c=colors,s=10)
    plt.show()
    
def show_pcord(X,P):
    _,d=X.shape
    df=pd.DataFrame(X,columns=[a+str(b) for a,b in zip(['x']*d,list(range(1,d+1)))])
    df['Cluster']=P
    pd.plotting.parallel_coordinates(df,'Cluster')
    plt.show()
    
def init(X,K):
    N,_=X.shape
    U=[[rnd.uniform(0,1) for _ in range(K)] for _ in range(N)]
    U=np.array(U)
    U/=U.sum(axis=0)
    return U

def kernel(x,bandwidth=0.95,sigma=2.5):      #computes the distance is mapped hilbert space
    return np.exp(-(np.linalg.norm(x)**2*bandwidth)/(2*sigma**2))

def dist(X,U,Y):    
    n,k=U.shape
    m,_=Y.shape
    K1=np.array([[kernel(X[i]-X[j]) for j in range(n)] for i in range(n)])
    K2=np.array([[kernel(Y[i]-X[j]) for j in range(n)] for i in range(m)])
    S=U.sum(axis=0)             #column sum of membership matrix U
    t1=np.full((m,k),kernel(0))                 #1st term 
    t2=-2*np.dot(K2,U)/S                        #2nd term
    t3=np.array([np.diag(U.T@K1@U)]*m)/(S**2)     #3rd term
    return t1+t2+t3             #return the distance matrix from the k Higher dim centroids of Y

# def dist(X,U,Y):    
#     n,k=U.shape
#     m,_=Y.shape
#     K1=np.array([[kernel(X[i]-X[j]) for j in range(n)] for i in range(n)])
#     K2=np.array([[kernel(Y[i]-X[j]) for j in range(n)] for i in range(m)])
#     S=U.sum(axis=0)             #column sum of membership matrix U
#     D=np.zeros([m,k])
#     for i in range(n):
#         for j in range(k):
#             D[i,j]=kernel(0)-2*np.dot(U[:,j],K2[i,:])/S[j]
#             +np.sum([[U[u,j]*U[v,j]*K1[u,v] for v in range(n)] for u in range(n)])/S[j]**2 
#     return D             #return the distance matrix from the k Higher dim centroids of Y
    
def cost(X,U):
    n,k=U.shape
    
    K=np.array([[kernel(X[i]-X[j]) for j in range(n)] for i in range(n)])
    return np.matrix.trace((U.T)@K@U)

def update(X,U):
    U=np.exp(-dist(X,U,X))
    # U=1/dist(X,U,X)
    U/=U.sum(axis=1,keepdims=True)
    return U

def kernel_kmeans(X,k,max_iters=30,tol=1e-17):
    n,_=X.shape
    U=init(X,k)
    
    P,_=predict(X,U,X)
    Z=np.array([[1 if P[i]==j else 0 for j in range(k) ] for i in range(n)])
    show2D(X,Z)
    
    lcost=0
    for _ in range(max_iters):
        U=update(X,U)     #Updates the membership matrix
        pcost=cost(X,U)         #computes the likelihood
        # if np.abs(pcost-lcost)<tol:    #stop if clusters stabilize 
        #     print(f'Iterations:{_}')
        #     break
        lcost=pcost
    # if X.shape[1]==2:
    #     show2D(X,U)
    # elif X.shape[1]==3:
    #     show3D(X,U)
    # else:
    #     show_pcord(X,U)
    return predict(X,U,X)

def predict(X,U,Y):
    D=np.exp(-dist(X,U,Y))
    Z=D/D.sum(axis=1,keepdims=True)
    P=np.argmax(Z,axis=1)
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

def make_moons(C=[[0]*2,[20,0]],R=[20,20],N=[30,30],A=[[-0.1,np.pi*1.1],[np.pi*0.9,2.1*np.pi]],noise=2):
    X=np.empty([0,2])
    for i in range(len(N)):
        theta=np.linspace(A[i][0],A[i][1],num=N[i])
        x=[C[i]]*N[i]+R[i]*np.stack((np.cos(theta),np.sin(theta)),axis=1)+np.random.multivariate_normal([0]*2,np.diag([noise]*2),N[i])
        X=np.append(X,x,axis=0)
    return X
    
def main():
    # # """Generate data from gaussian mixture models and cluster them or use the iris dataset"""
    # x1=np.random.multivariate_normal([1,2],[[1,0],[0,2]],60)
    # x2=np.random.multivariate_normal([5,6],[[4,0],[0,4]],80)
    # x3=np.random.multivariate_normal([10,4],[[3,0],[0,1]],100)
    # X=np.vstack([x1,x2,x3])
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
    # L=[0]*100+[1]*200
    # n=60+80+100
    # k=3
    # P,U=kernel_kmeans(X,k)
    # show_pcord(X,P)
    # show_pcord(X,L)

    # # print(Z)
    # print(f'L:-\n{L}')
    # print(f'P:-\n{P}')
    # print(f'M:-\n{U}')
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
    X=make_rings([[0]*2]*2,[10,20],[20,40])
    # X=make_moons()
    L=[0]*20+[1]*40
    n=60
    k=2
    P,U=kernel_kmeans(X,k)
    Z=np.array([[1 if P[i]==j else 0 for j in range(k) ] for i in range(n)])
    # show_pcord(X,P)
    # show_pcord(X,L)
    show2D(X,Z)
    # print(predict(X,Z,X))
    # print(f'L:-\n{L}')
    # print(f'P:-\n{P}')
    print(f"Adjusted Random Index is {ARI(L,P)}")
    # print(x.shape)
    # plt.scatter(x[:,0],x[:,1])
    # plt.show()

if __name__=="__main__":
    main()