import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ion()
from VGG_model import *
import tensorflow as tf
import pdb
import numpy as np
import h5py
import scipy.io as scipy_io
import numpy as np
#from params import *
import cv2
import time
import cProfile
def crop_image_include_coundary(output,input,center):
  height,width = output.shape[0],output.shape[1]
  min_x = max(0,int(center[0]-height/2))
  min_out_x = max(0,int(height/2-center[0])) 

  min_y = max(0,int(center[1]-width/2))
  min_out_y = max(0,int(width/2-center[1]))

  max_x = min(input.shape[0],int(center[0]+height/2))
  max_out_x = min(height, height+ input.shape[0]- int(center[0]+height/2))
  max_y =  min(input.shape[1],int(center[1]+width/2))
  max_out_y = min(width, width+ input.shape[1]- int(center[1]+width/2))     
  
  try:
    output[min_out_x:max_out_x,min_out_y:max_out_y,:] = input[min_x:max_x,min_y:max_y,:]
  except:
    pdb.set_trace()

  return output
  
class data_handler_(object):
  def __init__(self,sess,length,batch_size=1):
    self.batch_size = batch_size
    self.length = length
    self.sess = sess
    #self.data_file_ = h5py.File('../data/butterfly.h5','r')
    self.list_files = [item[0:-1] for item in open('../data_origin/test_files.txt','r').readlines()]
    #self.list_files = [item[0:-1] for item in open('../data/test_files.txt','r').readlines()]
    self.list_len = len(self.list_files)
    self.data_file_ = h5py.File('../data_variable/'+re.sub('origin','',self.list_files[1]),'r')
    self.just_image_file_ = h5py.File('../data_origin/'+self.list_files[1])
	
    self.mask_file = h5py.File('/scratch/ys1297/LSTM_tracking/data_mask/'+ re.sub('origin','',self.list_files[1]),'r')

    #self.data_file_ =h5py.File('../data/ball1.h5','r')
    self.dataset_size = self.data_file_['image'].shape[0] - self.length-1
    self.height = self.data_file_['image'].shape[1]
    self.width = self.data_file_['image'].shape[2]
    np.random.seed(100)
    
    self.indices_ = np.arange(self.data_file_['image'].shape[0] - self.length)
    #np.random.shuffle(self.indices_)
    self.id_ = 0 # every hdf5 ids
    self.file_id = 0 # all file ids
    


  def Get_all(self):
    feature_0_mask = np.zeros((self.batch_size,14,14,512))
    feature_0_no_mask = np.zeros((self.batch_size,14,14,512))
    first_mask = np.zeros((self.batch_size,self.height,self.width,1))
    all_images = np.zeros((self.batch_size,self.length,self.height,self.width,3))
    all_masks = np.zeros((self.batch_size,self.length,self.height,self.width,3))

    #file_temp = h5py.File('../data_origin/'+self.list_files[22],'r')
    for i in range(self.batch_size):
      feature_0_mask[i,:,:,:] = self.data_file_['first_frame_feature'][self.indices_[self.id_]]
      feature_0_no_mask[i,:,:,:] = self.data_file_['feature'][self.indices_[self.id_]]
      #feature_0_mask[i,:,:,:] = file_temp['first_frame_feature'][519]
      #feature_0_no_mask[i,:,:,:] = file_temp['feature'][519] 
	 
      all_images[i,:,:,:,:] = self.just_image_file_['image'][self.indices_[self.id_]+1:self.indices_[self.id_]+1+self.length]
      all_masks[i,:,:,:,:] = self.mask_file['image'][self.indices_[self.id_]+1:self.indices_[self.id_]+1+self.length]>0 
      first_mask[i,:,:,:] = cv2.cvtColor(self.mask_file['image'][self.indices_[self.id_]], cv2.COLOR_BGR2GRAY).reshape(self.height,self.width,1)>0
    #self.data_file_.close()
    #self.just_image_file_.close()
    #self.mask_file.close()
    return feature_0_mask,feature_0_no_mask,first_mask,all_images,all_masks 
  

  def set_id(self,id):
    self.id_ = id   

  

  def set_file_id(self):
    self.data_file_.close()
    self.file_id +=1
    if self.file_id >= self.list_len :
      self.file_id = 0

    self.data_file_ = h5py.File('../data_variable/'+self.list_files[self.file_id],'r')
    self.id_ = 0
    self.dataset_size = self.data_file_['image'].shape[0] - self.length-1
    self.indices_ = np.arange(self.data_file_['image'].shape[0] - self.length-1)
    np.random.shuffle(self.indices_)

 
  def generate_group_image(self,images,predict,truth,file_name):
    for i in range(images.shape[0]):
      plt.subplot(2,1,1)
      plt.imshow(images[i,:,:,:]/255.0,alpha=0.9)
      plt.imshow(predict[i,:,:,:].squeeze(),alpha=0.3)
      plt.axis('off')
      plt.subplot(2,1,2)
      plt.imshow(images[i,:,:,:]/255.0,alpha=0.9)
      plt.imshow(cv2.cvtColor(np.uint8(truth[i,:,:,:].squeeze()),cv2.COLOR_BGR2GRAY)/255.0,alpha=0.3)
      plt.axis('off')
      plt.savefig(file_name+str(i),bbox_inches='tight')
      plt.clf()
 

  def generate_overlay_image(self,images,mask,file_name):
    for i in range(32):
      plt.subplot(4,8,i+1)
      plt.imshow(images[i,:,:,::-1]/255.0,alpha=0.9)
      plt.imshow(mask[i,:,:,:].squeeze(),alpha=0.3)
      plt.axis('off')
    plt.savefig(file_name, bbox_inches='tight')
    plt.clf()


if __name__ =="__main__":
   sess = tf.Session()
   data_handler = data_handler_(sess,2)
   for i in range(5): 
     t = time.time() 
     feature_0_mask,feature_0_no_mask, all_features, all_masks,all_images = data_handler.GetBatch()
     print time.time()-t
     #data_handler.generate_overlay_image(all_images[0,:,:,:,:],all_masks[0,:,:,:,:],'test'+str(i))
 
     print i



	
