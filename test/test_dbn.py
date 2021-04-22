from util import *
from rbm import RestrictedBoltzmannMachine
from dbn import DeepBeliefNet

image_size = [28, 28]
train_imgs, train_lbls, test_imgs, test_lbls = read_mnist(dim=image_size, n_train=60000, n_test=10000)

vz_image(train_imgs[0, :])
dbn = DeepBeliefNet(sizes={"vis": image_size[0] * image_size[1], "hid": 500, "pen": 500, "top": 2000, "lbl": 10},
                    image_size=image_size,
                    n_labels=10,
                    batch_size=20,
                    image = np.reshape(train_imgs[0, :], (1, -1))
                    )

''' greedy layer-wise training '''

dbn.train_greedylayerwise(vis_trainset=train_imgs, lbl_trainset=train_lbls, n_iterations=10)
prob_h0, samples_h0 = dbn.rbm_stack['vis--hid'].get_h_given_v_dir(test_imgs)
histogram_hidden_probs(prob_h0)
histogram_weights(dbn.rbm_stack['vis--hid'].weight_h_to_v)

#dbn.recognize(train_imgs, train_lbls)

#dbn.recognize(test_imgs, test_lbls)

for digit in range(10):
    digit_1hot = np.zeros(shape=(1, 10))
    digit_1hot[0, digit] = 1
    #dbn.generate(digit_1hot, name="rbms")

# fine tuning
dbn.train_wakesleep_finetune(vis_trainset=train_imgs, lbl_trainset=train_lbls, n_iterations=1)

dbn.recognize(train_imgs, train_lbls)

dbn.recognize(test_imgs, test_lbls)

for digit in range(10):
    digit_1hot = np.zeros(shape=(1, 10))
    digit_1hot[0, digit] = 1
    dbn.generate(digit_1hot, name="rbms")