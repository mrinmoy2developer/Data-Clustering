import matplotlib.pyplot as plt
import numpy as np
bw=0.0095
k=2    #no of cluster

def mydistance(bw,x,y):
    return np.exp(-1*bw*np.sum((x-y)*(x-y)))

mycolor=["red","blue","green","yellow","orange","purple","tan","brown","pink","black"]

def listfinder(list1,a):
  n=len(list1)
  #print(a,type(a))
  for i in range(n):
    if a in list1[i]:
      break
  for j in range(len(list1[i])):
    if a==list1[i][j]:
      break
  #l=[i,j]
  #print(l,type(l))  
  return [i,j]


def kmean(data):
  M1=np.max(data[:,[0]])
  m1=np.min(data[:,[0]])
  M2=np.max(data[:,[1]])
  m2=np.min(data[:,[1]])
  n=len(data[:,[0]])
  i=0
  cluster_no=list()
  for i in range(k):
    cluster_no.append(np.array([i]))
  i=i+1
  while i < n:
    j=i%k
    cluster_no[j]=np.append(cluster_no[j],i)
    i=i+1
  gram_matrix=np.zeros((n,n))
  #print(cluster_no)
  for i in range(n):
    for j in range(n):
      gram_matrix[[i],[j]]=mydistance(bw,data[i,:],data[j,:])
  dist=np.array(range(k))
  #print(gram_matrix)
  for i in range(10):
    for j in range(n):
      s=0
      for l in range(k): 
        s=len(cluster_no[l])
        print(type(np.sum(gram_matrix[cluster_no[l],:][:,cluster_no[l]])))
        dist[l]=gram_matrix[[j],[j]]-2*(np.sum(gram_matrix[[j],:][:,cluster_no[l]]))/s+np.sum(gram_matrix[cluster_no[l],:][:,cluster_no[l]])/(s**2)
      #print(dist,type(dist))
      #print(cluster_no)
      p=listfinder(cluster_no,j)
      ll=np.argmin(dist)
      #print(dist)
      cluster_no[p[0]]=np.delete(cluster_no[p[0]],p[1])
      cluster_no[ll]=np.append(cluster_no[ll],j)
    #print(cluster_no)
  #t=int(999999/k)
  print(cluster_no)
  for i in range(k):
    pk=cluster_no[i]
    #plt.scatter(data[:,[0]][[pk],:],data[:,[1]][[pk],:],c='#%06X'%(t*k))
    plt.scatter(data[:,[0]][[pk],:],data[:,[1]][[pk],:],c=mycolor[i])
  plt.show()

def simulator():
  x=np.random.uniform(0,(2*np.pi),size=110)
  y=np.abs(np.random.normal(0,2,50))
  z=np.abs(np.random.normal(20,1,60))
  y=np.concatenate((y,z))
  x1=y*(np.cos(x))
  x2=y*(np.sin(x))
  x=np.concatenate((x1,x2))
  x=np.reshape(x,(2,110))
  x=np.transpose(x)
  plt.scatter(x[:,[0]],x[:,[1]],c='red')
  plt.show()
  return x

def make_rings(C=[[0]*2]*2,R=[10,20],N=[100,200],noise=2):
    X=np.empty([0,2])
    for i in range(len(N)):
        theta=np.linspace(0,2*np.pi,num=N[i])
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
    X=make_rings([[0]*2]*2,[10,20],[100,200])
    L=[0]*100+[1]*200
    n=300
    y=simulator()
    print(y.shape)
    print(X.shape)
    
    kmean(X)    
    # Z=np.array([[1 if P[i]==j else 0 for j in range(k) ] for i in range(n)])
    # show_pcord(X,P)
    # show_pcord(X,L)
    # show2D(X,Z)
    # print(predict(X,Z,X))
    # print(f'L:-\n{L}')
    # print(f'P:-\n{P}')
    # print(f"Adjusted Random Index is {ARI(L,P)}")
    # print(x.shape)
    # plt.scatter(x[:,0],x[:,1])
    # plt.show()

if __name__=="__main__":
    main()
