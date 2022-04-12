import numpy as np
import matplotlib.pyplot as plt
import subfunctions as sfns
import tensorflow as tf
from tensorflow.keras import callbacks, optimizers
from tensorflow.keras import backend as K

PROBE_PATH    = './0_train-cnn/input/probe.npy'
IMAGES_PATH   = './0_train-cnn/input/images.npy'
WEIGHTS_PATH  = './0_train-cnn/output/weights'
NUM_EPOCHS    = 100
LEARNING_RATE = 0.0001
DR_RATE       = 0.2

def gen_masks(probe):
  p = np.abs(np.conj(probe)*probe)
  mask_small = p>np.max(p)*0.16
  mask_zoom  = mask_small[12:20,12:20]
  mask       = sfns.imresize_big(mask_zoom,4)
  return mask_small, mask_zoom, mask

def calc_sim_data(probe, imgs, mask):
  sim_x = np.zeros((imgs.shape[0],probe.shape[0],probe.shape[1]))
  sim_y = np.zeros((imgs.shape[0],mask.shape[0],mask.shape[1]))
  for i in range(0,imgs.shape[0]):
    obj = imgs[i,:,:]
    dp  = np.exp(1.j*obj)*probe
    dp  = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(dp)))
    dp  = np.abs(np.conj(dp)*dp)
    dp  = np.random.poisson(dp)
    sim_x[i,:,:] = np.sqrt(dp)
    obj = obj*mask
    obj = obj[12:20,12:20]
    obj = sfns.imresize_big(obj, 4)
    sim_y[i,:,:] = obj
  return sim_x, sim_y

def l1_in_mask(y_true, y_pred):
  error = tf.multiply(y_true[0,:,:,0]-y_pred[0,:,:,0],mask)
  error = tf.abs(error)
  return K.sum(error)/np.sum(mask)

def main():
  #--------------------------------------------------
  print("Loading probe and stock images")
  probe = np.load(PROBE_PATH)
  imgs  = np.load(IMAGES_PATH)

  #--------------------------------------------------
  print("Generating masks")
  global mask
  mask_small, mask_zoom, mask = gen_masks(probe)
  np.save('./0_train-cnn/output/mask.npy', mask)
  mask = 1.*mask

  #--------------------------------------------------
  print("Generating simulated diffraction patterns")
  sim_x, sim_y = calc_sim_data(probe, imgs, mask_small)
  sim_x = sim_x[..., np.newaxis]
  sim_y = sim_y[..., np.newaxis]

  #--------------------------------------------------
  print("Training neural network")
  model = sfns.create_model(DR_RATE)
  opt = optimizers.SGD(learning_rate=LEARNING_RATE)
  model.compile(optimizer=opt,loss=l1_in_mask)
  model.summary()
  checkpoints = callbacks.ModelCheckpoint('%s/{epoch:03d}.hdf5' %WEIGHTS_PATH,
    save_weights_only=False, verbose=1, save_freq="epoch")
  history = model.fit(sim_x, sim_y, shuffle=True, batch_size=16, verbose=1,
    epochs=NUM_EPOCHS, validation_split=0.05, callbacks=[checkpoints])

if __name__ == "__main__":
  main()

