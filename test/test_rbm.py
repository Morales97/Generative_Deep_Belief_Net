from util import *
from rbm import RestrictedBoltzmannMachine
import pdb

image_size = [28,28]
train_imgs,train_lbls,test_imgs,test_lbls = read_mnist(dim=image_size, n_train=60000, n_test=10000)

vz_image(test_imgs[0])
vz_image(test_imgs[1])

rbm = RestrictedBoltzmannMachine(ndim_visible=image_size[0]*image_size[1],
                                     ndim_hidden=500,
                                     is_bottom=True,
                                     image_size=image_size,
                                     is_top=False,
                                     n_labels=10,
                                     batch_size=20
    )

rbm.cd1(visible_trainset=train_imgs[0:60000], n_iterations=10)

prob_h0, samples_h0 = rbm.get_h_given_v(test_imgs)
histogram_hidden_probs(prob_h0)

prob_h0, samples_h0 = rbm.get_h_given_v(test_imgs[0])
prob_v1, samples_v1 = rbm.get_v_given_h(samples_h0)
vz_image(prob_v1)

prob_h0, samples_h0 = rbm.get_h_given_v(test_imgs[1])
prob_v1, samples_v1 = rbm.get_v_given_h(samples_h0)
vz_image(prob_v1)