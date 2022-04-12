import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import models

from scipy.ndimage import gaussian_filter

DIFFPATS_PATH  = './1_phase-recon/input/diffpats.npy'
POSITIONS_PATH = './1_phase-recon/input/positions.npy'
MASK_PATH      = './1_phase-recon/input/mask.npy'
WEIGHTS_PATH   = './1_phase-recon/input/020.hdf5'
OUTPUT_PATH    = './1_phase-recon/output/recon.npy'
NUM_ITER       = 3
ALPHA          = 0.1

def imresize_small(img, factor):
  img_small = np.zeros((int(img.shape[0]/factor),int(img.shape[1]/factor)))
  for i in range(0,img.shape[0]):
    for j in range(0,img.shape[1]):
      x = int(i/factor)
      y = int(j/factor)
      img_small[x,y] = img_small[x,y] + img[i,j]
  return img_small/factor/factor

def stitch(objs_pred,pos,mask,num_iter,alpha):
  mask_1d = mask.reshape((mask.shape[0]*mask.shape[1],1)).astype(bool)
  pos_r   = np.round(pos).astype('int')
  obj     = np.zeros((np.max(pos_r[:,0])+objs_pred.shape[1]+1,np.max(pos_r[:,1])+objs_pred.shape[2]+1))
  idxs    = list(range(objs_pred.shape[0]))
  np.random.shuffle(idxs)
  for i in range(0,NUM_ITER):
    for idx in range(0,objs_pred.shape[0]):
      p = idxs[idx]
      x = pos_r[p,0]
      y = pos_r[p,1]
      a = obj[x:x+objs_pred.shape[1],y:y+objs_pred.shape[2]]
      a_1d = a.reshape((mask.shape[0]*mask.shape[1],1))
      a_mean = np.mean(a_1d[mask_1d])
      b = objs_pred[p,:,:]
      obj_diff = ALPHA*(b-a+a_mean)*mask
      a = a + obj_diff
      obj[x:x+objs_pred.shape[1],y:y+objs_pred.shape[2]] = a
  return obj

def main():
  print("Generating objs predictions")
  mask     = np.load(MASK_PATH)
  mask_1d  = mask.reshape((mask.shape[0]*mask.shape[1],1)).astype(bool)
  diffpats = np.load(DIFFPATS_PATH)
  diffpats = diffpats[...,np.newaxis]
  model = models.load_model(WEIGHTS_PATH, compile=False)
  preds = model.predict(diffpats)
  objs_pred = np.zeros((preds.shape[0],preds.shape[1],preds.shape[2]))
  for i in range(0,preds.shape[0]):
    o = preds[i,:,:,0].reshape((mask.shape[0]*mask.shape[1],1))
    o_mean = np.mean(o[mask_1d])
    objs_pred[i,:,:] = (preds[i,:,:,0]-o_mean)*mask

  print("Stitching objs")
  pos       = np.load(POSITIONS_PATH)
  pos       = pos*4.0
  obj_final = stitch(objs_pred, pos, mask, NUM_ITER, ALPHA)
  obj_final = imresize_small(obj_final,4)
  np.save(OUTPUT_PATH,obj_final)
  plt.imshow(obj_final)
  plt.show()

if __name__ == "__main__":
  main()

