import numpy as np

def SVD536(Mn, DebugMode = False):
  '''
  this function gets a full rank noisy nxd size Matrix and return noise free and possibly lower rank matrix size of nxd

  
  '''
  U, Si, Vt  = np.linalg.svd(Mn, full_matrices=False) #decomposing the given matrix 
  #to its singular vectors and singular values.

  S=np.diag(Si)#matrix form of the singular values.

  exsum=np.trace(np.matmul(S,np.transpose(S))) # sum of squares of all the singular values
  #print (S.shape)
  sum=0 #initials for the loop
  RankApp=0 #initials for the loop
  for i in range(S.shape[0]): 

    RankApp=RankApp+1 # increasing number of the used singular values in each loop
    sum=sum+pow(S[i,i],2) # sum of the squares of the singular values to be used construct M_res

    if sum/exsum>0.955: # conditon to loop terminate.

      break
    else: # else loop contiue to increase number of singular values to used

      continue

  
  #print (RankApp)
  Mres=np.matmul(U[:,:RankApp],np.matmul(np.diag(Si[:RankApp]),Vt[:RankApp,:]))
  PC=np.matmul(U[:,:RankApp],np.diag(Si[:RankApp]))
  # M_res constructed from reduced matrices.
  plt.plot(Si,'r*')
  return Mres,PC
