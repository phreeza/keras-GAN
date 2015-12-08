# Simple GAN implementation with keras
# adaptation of https://gist.github.com/Newmu/4ee0a712454480df5ee3
import sys
sys.path.append('/home/mccolgan/PyCharm Projects/keras')
sys.path.insert(0,'/home/mccolgan/local/lib/python2.7/site-packages/')
from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Flatten
from keras.layers.convolutional import Convolution1D, UpSample1D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD,RMSprop
from keras.layers.advanced_activations import LeakyReLU
from keras.initializations import normal
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde
from scipy.io import wavfile
import theano.tensor as T
import theano
import pydub

batch_size = 128
size_x = 4096
size_z = 256
size_z_fake = 256
assert (((size_z+64-1)*4 + 256 - 1)*4 - 2029 + 1) == size_x

print "loading data"

f = pydub.AudioSegment.from_mp3('../ml-music/07_-_Brad_Sucks_-_Total_Breakdown.mp3')
data = np.fromstring(f._data, np.int16)
data = data.astype(np.float32).reshape((-1,2))
print data.shape
data = data[:,0]+data[:,1]
#data = data[:,:subsample*int(len(data)/subsample)-1,:]
data -= data.min()
data /= data.max() / 2.
data -= 1.
print data.shape

print "Setting up decoder"

decoder = Sequential()
decoder.add(Dense(2048, input_dim = size_x, activation='relu'))
decoder.add(Dropout(0.5))
decoder.add(Dense(2048, activation='relu'))
decoder.add(Dropout(0.5))
decoder.add(Dense(1, activation='sigmoid'))

#sgd1 = SGD(lr=0.01, momentum=0.1)
sgd1 = RMSprop()
decoder.compile(loss='binary_crossentropy', optimizer=sgd1)

print "Setting up generator"
generator = Sequential()
generator.add(Convolution1D(16, 64, input_dim=16, input_length = size_z, border_mode='full'))
generator.add(LeakyReLU())
#generator.add(BatchNormalization())
generator.add(UpSample1D(4))
generator.add(Convolution1D(4, 256, border_mode='full'))
generator.add(LeakyReLU(alpha=0.0))
#generator.add(BatchNormalization())
generator.add(UpSample1D(4))
#(((2048+64-1)*4 + 256 - 1)*4 - 2029 + 1)
generator.add(Convolution1D(1, 2029, activation='linear'))

sgd = RMSprop()
#sgd = SGD(lr=0.01, momentum=0.1)
generator.compile(loss='binary_crossentropy', optimizer=sgd)

print "Setting up combined net"
gen_dec = Sequential()
gen_dec.add(generator)
gen_dec.add(Flatten())
decoder.trainable=False
gen_dec.add(decoder)

#def inverse_binary_crossentropy(y_true, y_pred):
#    if theano.config.floatX == 'float64':
#        epsilon = 1.0e-9
#    else:
#        epsilon = 1.0e-7
#    y_pred = T.clip(y_pred, epsilon, 1.0 - epsilon)
#    bce = T.nnet.binary_crossentropy(y_pred, y_true).mean(axis=-1)
#    return -bce
#
#gen_dec.compile(loss=inverse_binary_crossentropy, optimizer=sgd)

sgd3 = RMSprop()
#sgd3 = SGD(lr=0.01, momentum=0.1)
gen_dec.compile(loss='binary_crossentropy', optimizer=sgd3)

y_decode = np.ones(2*batch_size)
y_decode[:batch_size] = 0.
y_gen_dec = np.ones(batch_size)

def gaussian_likelihood(X, u=0., s=1.):
    return (1./(s*np.sqrt(2*np.pi)))*np.exp(-(((X - u)**2)/(2*s**2)))

#def vis(i):
#    s = 1.
#    u = 0.
#    zs = np.linspace(-1, 1, 500).astype('float32')[:,np.newaxis]
#    xs = np.linspace(-5, 5, 500).astype('float32')[:,np.newaxis]
#    ps = gaussian_likelihood(xs, 1.)
#
#    gs = generator.predict(zs)
#    print gs.mean(),gs.std()
#    preal = decoder.predict(xs)
#    kde = gaussian_kde(gs.flatten())
#
#    plt.clf()
#    plt.plot(xs, ps, '--', lw=2)
#    plt.plot(xs, kde(xs.T), lw=2)
#    plt.plot(xs, preal, lw=2)
#    plt.xlim([-5., 5.])
#    plt.ylim([0., 1.])
#    plt.ylabel('Prob')
#    plt.xlabel('x')
#    plt.legend(['P(data)', 'G(z)', 'D(x)'])
#    plt.title('GAN learning gaussian')
#    fig.canvas.draw()
#    plt.show(block=False)
#    if i%100 == 0:
#        plt.savefig('current.png')
#    plt.pause(0.01)

#fig = plt.figure()

with open('generator.json','w') as fh:
    fh.write(generator.to_json())

for i in range(100000):
    zmb = np.random.normal(0., 1, size=(batch_size, size_z, 16)).astype('float32')
    #xmb = np.random.normal(1., 1, size=(batch_size, 1)).astype('float32')
    xmb = np.array([data[n:n+size_x] for n in np.random.randint(0,data.shape[0]-size_x,batch_size)])
    #xmb = xmb[:,:,np.newaxis]
    if i % 10 == 0:
        err_E = 2
        while err_E > 0.9:
            r = gen_dec.fit(zmb,y_gen_dec,nb_epoch=1,verbose=0)
            err_E = r.totals['loss']/batch_size
            print 'E:',err_E
            zmb = np.random.normal(0., 1, size=(batch_size, size_z, 16)).astype('float32')
    elif i % 10 == 9:
        err_D = 2
        while err_D > 0.9:
            r = decoder.fit(np.vstack([generator.predict(zmb).squeeze(),xmb]),y_decode,nb_epoch=1,verbose=0)
            err_D = (r.totals['loss']/r.seen)
            print 'D:',err_D
            zmb = np.random.normal(0., 1, size=(batch_size, size_z, 16)).astype('float32')
            xmb = np.array([data[n:n+size_x] for n in np.random.randint(0,data.shape[0]-size_x,batch_size)])
    else:
        r = decoder.fit(np.vstack([generator.predict(zmb).squeeze(),xmb]),y_decode,nb_epoch=1,verbose=0)
        print 'D:',(r.totals['loss']/r.seen)
    if i % 10 == 0:
        print "saving fakes"
        zmb = np.random.normal(0., 1, size=(16, size_z_fake, 16)).astype('float32')
        fakes = generator.predict(zmb)
        wavfile.write('fake_epoch_'+str(i)+'.wav',44100,fakes[0,:])
        for n in range(16):
            wavfile.write('fake_'+str(n+1)+'.wav',44100,fakes[n,:])
            wavfile.write('real_'+str(n+1)+'.wav',44100,xmb[n,:])
#        vis(i)
        print "saving generator weights"
        generator.save_weights('generator.h5',overwrite=True)
