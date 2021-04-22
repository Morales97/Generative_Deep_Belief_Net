from util import *
from rbm import RestrictedBoltzmannMachine

image_size = [28,28]
train_imgs, train_lbls, test_imgs, test_lbls = read_mnist(dim=image_size, n_train=60000, n_test=10000)
N = 60000

rbm_bottom = RestrictedBoltzmannMachine(ndim_visible=image_size[0]*image_size[1],
                                     ndim_hidden=500,
                                     is_bottom=True,
                                     image_size=image_size,
                                     is_top=False,
                                     n_labels=10,
                                     batch_size=20
    )

rbm_bottom.cd1(visible_trainset=train_imgs[0:N], n_iterations=10)\

rbm_top = RestrictedBoltzmannMachine(ndim_visible=500,
                                     ndim_hidden=500,
                                     is_bottom=False,
                                     image_size=image_size,
                                     is_top=False,
                                     n_labels=10,
                                     batch_size=20
    )

prob_h, samples_h = rbm_bottom.get_h_given_v(train_imgs[0:N])
trainset_second_rbm = samples_h
rbm_top.cd1(trainset_second_rbm, n_iterations=10)