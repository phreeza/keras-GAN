# Simple GAN implementation with keras
# adaptation of https://gist.github.com/Newmu/4ee0a712454480df5ee3

from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from keras.initializations import normal
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde
import theano.tensor as T
import theano

batch_size = 128

print "Setting up decoder"
decoder = Sequential()
decoder.add(Dense(16, input_dim=1, activation='relu'))
decoder.add(Dense(16, activation='relu'))
decoder.add(Dense(1, activation='sigmoid'))

sgd = SGD(lr=0.01, momentum=0.1)
decoder.compile(loss='binary_crossentropy', optimizer=sgd)

print "Setting up generator"
generator = Sequential()
generator.add(Dense(16, input_dim=1, activation='relu'))
generator.add(Dense(16, activation='relu'))
generator.add(Dense(1, activation='linear'))

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

def vis(i):
    s = 1.
    u = 0.
    zs = np.linspace(-1, 1, 500).astype('float32')[:,np.newaxis]
    xs = np.linspace(-5, 5, 500).astype('float32')[:,np.newaxis]
    ps = gaussian_likelihood(xs, 1.)

    gs = generator.predict(zs)
    print gs.mean(),gs.std()
    preal = decoder.predict(xs)
    kde = gaussian_kde(gs.flatten())

    plt.clf()
    plt.plot(xs, ps, '--', lw=2)
    plt.plot(xs, kde(xs.T), lw=2)
    plt.plot(xs, preal, lw=2)
    plt.xlim([-5., 5.])
    plt.ylim([0., 1.])
    plt.ylabel('Prob')
    plt.xlabel('x')
    plt.legend(['P(data)', 'G(z)', 'D(x)'])
    plt.title('GAN learning gaussian')
    fig.canvas.draw()
    plt.show(block=False)
    if i%100 == 0:
        plt.savefig('current.png')
    plt.pause(0.01)

fig = plt.figure()

for i in range(100000):
    zmb = np.random.uniform(-1, 1, size=(batch_size, 1)).astype('float32')
    xmb = np.random.normal(1., 1, size=(batch_size, 1)).astype('float32')
    if i % 10 == 0:
        r = gen_dec.fit(zmb,y_gen_dec,nb_epoch=1,verbose=0)
        print np.exp(r.totals['loss']/128)
    else:
        decoder.fit(np.vstack([generator.predict(zmb),xmb]),y_decode,nb_epoch=1,verbose=0)
    if i % 10 == 0:
        print i
        vis(i)
