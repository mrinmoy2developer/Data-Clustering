##Deydstar
library(mvtnorm)

dat=file.choose()
M=dat
if(norm(M)>10e-6)
  M=M/norm(M)
norm_v <- function(x) sqrt(sum(x^2))

hardkmeans <- function(M,k,n)
{
  z<-matrix(runif(k*ncol(M),-1,1),ncol=ncol(M),nrow=k)
  N=nrow(M)
  for (l in 1:n)
  {
    sum1=matrix(0,ncol=ncol(M),nrow=k);C=c(0,k);
    for(i in 1:N)
    {
      mini=10e6;foo=0;
      for(j in 1:k)
      {
        if (norm_v(M[i,]-z[j,]) < mini)
        {
          mini=norm_v(M[i,]-z[j,]);
          foo=j;
        }
      }
      sum1[foo,]=sum1[foo,]+M[i,];
      C[foo]=C[foo]+1;
    }
    for(j in 1:k)
    {
      if(C[j]!=0)
       {z[j,]=sum1[j,]/C[j];}
    }
  }
  return(z)
}

mydata <- rbind(rmvnorm(50, c(1,1), matrix(c(.5^2,.20,.20,.5^2), ncol=2)),rmvnorm(50, c(4,4), matrix(c(1,0,0,1), ncol=2)))
plot(mydata,pch=19)
