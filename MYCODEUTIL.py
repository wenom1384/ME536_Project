import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import matrix_rank  as rank
import PIL
import io
from base64 import b64decode, b64encode

from IPython.display import display, Javascript
from google.colab.output import eval_js
from base64 import b64decode
from scipy.spatial import distance

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
  RankApp=4
  Mres=np.matmul(U[:,:RankApp],np.matmul(np.diag(Si[:RankApp]),Vt[:RankApp,:]))
  PC=np.matmul(U[:,:RankApp],np.diag(Si[:RankApp]))
  # M_res constructed from reduced matrices.
  plt.plot(Si,'r*')
  return Mres,PC


def js_to_image(js_reply):
  """
  Params:
          js_reply: JavaScript object containing image from webcam
  Returns:
          img: OpenCV BGR image
  """
  # decode base64 image
  image_bytes = b64decode(js_reply.split(',')[1])
  # convert bytes to numpy array
  jpg_as_np = np.frombuffer(image_bytes, dtype=np.uint8)
  # decode numpy array into OpenCV BGR image
  img = cv2.imdecode(jpg_as_np, flags=1)

  return img

# function to convert OpenCV Rectangle bounding box image into base64 byte string to be overlayed on video stream
def bbox_to_bytes(bbox_array):
  """
  Params:
          bbox_array: Numpy array (pixels) containing rectangle to overlay on video stream.
  Returns:
        bytes: Base64 image byte string
  """
  # convert array into PIL image
  bbox_PIL = PIL.Image.fromarray(bbox_array, 'RGBA')
  iobuf = io.BytesIO()
  # format bbox into png for return
  bbox_PIL.save(iobuf, format='png')
  # format return string
  bbox_bytes = 'data:image/png;base64,{}'.format((str(b64encode(iobuf.getvalue()), 'utf-8')))

  return bbox_bytes
