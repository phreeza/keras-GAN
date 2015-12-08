# IPython log file
import sys
sys.path.append('/home/mccolgan/PyCharm Projects/keras')
sys.path.insert(0,'/home/mccolgan/local/lib/python2.7/site-packages/')
from scipy.io import wavfile
import numpy as np

from keras.models import model_from_config
from keras.layers.convolutional import Convolution1D, UpSample1D

import json
generator_params = json.load(open('generator.json'))
generator_params['layers'][0]['input_length'] = 4096
generator_params['layers'][0]['input_shape'][0] = 4096
gen = model_from_config(generator_params,custom_objects={'UpSample1D':UpSample1D})
gen.load_weights('generator.h5')
zmb = np.random.normal(0., 1, size=(32, 4096, 16)).astype('float32')
fakes = gen.predict(zmb).squeeze()
for n in range(16):
    wavfile.write('fake_big'+str(n+1)+'.wav',44100,fakes[n,:])
