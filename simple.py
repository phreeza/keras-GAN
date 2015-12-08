# Simple GAN implementation with keras
# adaptation of https://gist.github.com/Newmu/4ee0a712454480df5ee3
import sys
sys.path.append('/home/mccolgan/PyCharm Projects/keras')
sys.path.insert(0,'/home/mccolgan/local/lib/python2.7/site-packages/')
from keras.models import Sequential
from keras.layers.core import Dense,Dropout
from keras.optimizers import SGD
from keras.initializations import normal
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde
from scipy.io import wavfile
import theano.tensor as T
import theano
import pydub

batch_size = 128

print "loading data"

f = pydub.AudioSegment.from_mp3('../ml-music/07_-_Brad_Sucks_-_Total_Breakdown.mp3')
data = np.fromstring(f._data, np.int16)
data = data.astype(np.float64).reshape((-1,2))
print data.shape
data = data[:,0]+data[:,1]
#data = data[:,:subsample*int(len(data)/subsample)-1,:]
data -= data.min()
data /= data.max() / 2.
data -= 1.
print data.shape

print "Setting up decoder"
decoder = Sequential()
decoder.add(Dense(2048, input_dim=32768, activation='relu'))
decoder.add(Dropout(0.5))
decoder.add(Dense(1024, activation='relu'))
decoder.add(Dropout(0.5))
decoder.add(Dense(1, activation='sigmoid'))

sgd = SGD(lr=0.01, momentum=0.1)
decoder.compile(loss='binary_crossentropy', optimizer=sgd)

print "Setting up generator"
generator = Sequential()
generator.add(Dense(2048*2, input_dim=2048, activation='relu'))
generator.add(Dense(1024*8, activation='relu'))
generator.add(Dense(32768, activation='linear'))

generator.compile(loss='binary_crossentropy', optimizer=sgd)

print "Setting up combined net"
gen_dec = Sequential()
gen_dec.add(generator)
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

gen_dec.compile(loss='binary_crossentropy', optimizer=sgd)

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

for i in range(100000):
    zmb = np.random.uniform(-1, 1, size=(batch_size, 2048)).astype('float32')
    #xmb = np.random.normal(1., 1, size=(batch_size, 1)).astype('float32')
    xmb = np.array([data[n:n+32768] for n in np.random.randint(0,data.shape[0]-32768,batch_size)])
    if i % 10 == 0:
        r = gen_dec.fit(zmb,y_gen_dec,nb_epoch=1,verbose=0)
        print 'E:',np.exp(r.totals['loss']/r.totals['seen'])
    else:
        r = decoder.fit(np.vstack([generator.predict(zmb),xmb]),y_decode,nb_epoch=1,verbose=0)
        print 'D:',np.exp(r.totals['loss']/r.totals['seen'])
    if i % 100 == 0:
        print "saving fakes"
        fakes = generator.predict(zmb[:16,:])
        wavfile.write('fake_epoch_'+str(i)+'.wav',44100,fakes[0,:])
        for n in range(16):
            wavfile.write('fake_'+str(n+1)+'.wav',44100,fakes[n,:])
            wavfile.write('real_'+str(n+1)+'.wav',44100,xmb[n,:])
#        vis(i)
