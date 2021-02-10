import cv2
import face_recognition
import numpy as np
from scipy.linalg import orth


def myfacedetector(image,ave_list,ave_list_names,PC):
  rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  boxes = face_recognition.face_locations(rgb,model='hog')
  encodings = face_recognition.face_encodings(rgb, boxes)
  #print(boxes[0])
  
  #print(ave_list_names)
  #print(ave_list)
  name=[]
  res_list=[]
  score=[]
  unknown=0
  for j in encodings:

    C,a,b,c=np.linalg.lstsq(PC,np.transpose(j),rcond=None)
    #print(f'residual is {a}')

    C=C.reshape(PC.shape[1],1)

    
    dist_list=[]
    #print(C)

    for y in ave_list:
      
      dist=distance.euclidean(y,C[1:PC.shape[1],:])
      #print(C[1:4,:])
      #dist = np.linalg.norm(y-C[1:4,:])
      #print(dist)
      dist_list.append(dist)
    #print(dist_list)
    index=dist_list.index(min(dist_list))

    relative_dist_list = [x / (min(dist_list)) for x in dist_list]
    #print(relative_dist_list)

    if np.sum(relative_dist_list)-1<5.5*pow((a/0.2),2) or ave_list_names[index]=='Empty':
      name.append(ave_list_names[index]+'????')
      unknown=1  
    else:
      name.append(ave_list_names[index])
      
    res_list.append(a)
    scr=np.sum(relative_dist_list)-1
    score.append(scr)
  #print(name)
  
  
  count=0
  for t,r,b,l in boxes:
    image=cv2.rectangle(image, (l, t), (r,b), (0, 255, 0), 2)
    image=cv2.putText(image, name[count], (l,t), cv2.FONT_HERSHEY_SIMPLEX,0.75, (0, 255, 0), 2)
    image=cv2.putText(image, 'res'+np.array2string(res_list[count])+'scr'+np.array2string(score[count]), (l,t-20), cv2.FONT_HERSHEY_SIMPLEX,0.75, (0, 255, 0), 2)

    count+=1
  return boxes,name,score,res_list,unknown


def hellofriend(new_name,new_encodings,ave_list,ave_list_names,M,namesgrp):

  'Hello friend I want to meet you and know you.'
  count=0
  for i in range(len(new_encodings)):
    if new_encodings[i]:
      M =np.append(M,np.transpose(new_encodings[i]),axis=1)
      count+=1
      #print('DENEME')
    #print(M.shape)
    
  #print(M.shape)
  ave_list_names.append(new_name)
  Mres,PC=SVD536(M)
  namesgrp=np.append(namesgrp,count)
  result=np.empty((3,0))
  

  for i in range(M.shape[1]): #every indivual vector in the given Matrix M is investigated.

    C,a,b,c=np.linalg.lstsq(PC,M[:,i],rcond=None) #
    #print(np.transpose(C).shape)
    #print(type(C))
    
    C=C.reshape(4,1)
    #print(np.transpose(C).shape)
    result=np.append(result,C[1:4,:],axis=1)

  #print(f'Result shape {result.shape}')
  grp=[0]
  #print(namesgrp)
  for i in range(len(namesgrp)):
    if i==0:
      grp.append(namesgrp[0])
    else:
      grp.append(grp[i]+namesgrp[i])
  print(grp)
  print(result.shape)
  
  ave_list=[]
  for j in range(len(grp)-1):
    average=[]
    for i in range(result.shape[0]):
      average.append(np.average(result[i,grp[j]:grp[j+1]]))
    print(average)
    ave_list.append(average)

  #print(ave_list)
  #print(ave_list_names)
  return ave_list,ave_list_names,M,namesgrp,PC
