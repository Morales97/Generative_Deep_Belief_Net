from util import *
from rbm import RestrictedBoltzmannMachine
import pdb

class DeepBeliefNet():    

    ''' 
    For more details : Hinton, Osindero, Teh (2006). A fast learning algorithm for deep belief nets. https://www.cs.toronto.edu/~hinton/absps/fastnc.pdf

    network          : [top] <---> [pen] ---> [hid] ---> [vis] 
                               `-> [lbl] 
    lbl : label
    top : top
    pen : penultimate
    hid : hidden
    vis : visible
    '''
    
    def __init__(self, sizes, image_size, n_labels, batch_size, image):

        """
        Args:
          sizes: Dictionary of layer names and dimensions
          image_size: Image dimension of data
          n_labels: Number of label categories
          batch_size: Size of mini-batch
        """

        self.rbm_stack = {
            
            'vis--hid' : RestrictedBoltzmannMachine(ndim_visible=sizes["vis"], ndim_hidden=sizes["hid"],
                                                    is_bottom=True, image_size=image_size, batch_size=batch_size),
            
            'hid--pen' : RestrictedBoltzmannMachine(ndim_visible=sizes["hid"], ndim_hidden=sizes["pen"], batch_size=batch_size),
            
            'pen+lbl--top' : RestrictedBoltzmannMachine(ndim_visible=sizes["pen"]+sizes["lbl"], ndim_hidden=sizes["top"],
                                                        is_top=True, n_labels=n_labels, batch_size=batch_size)
        }
        
        self.sizes = sizes

        self.image_size = image_size

        self.batch_size = batch_size
        
        self.n_gibbs_recog = 15
        
        self.n_gibbs_gener = 300
        
        self.n_gibbs_wakesleep = 10

        self.print_period = 1

        self.image = image
        
        return

    def recognize(self,true_img,true_lbl):

        """Recognize/Classify the data into label categories and calculate the accuracy

        Args:
          true_imgs: visible data shaped (number of samples, size of visible layer)
          true_lbl: true labels shaped (number of samples, size of label layer). Used only for calculating accuracy, not driving the net
        """

        n_samples = true_img.shape[0]
        
        visible_set = true_img # visible layer gets the image data
        
        lbl = np.ones(true_lbl.shape)/10. # start the net by telling you know nothing about labels        
        
        # [TODO TASK 4.2] fix the image data in the visible layer and drive the network bottom to top. In the top RBM, run alternating Gibbs sampling \
        # and read out the labels (replace pass below and 'predicted_lbl' to your predicted labels).
        # NOTE : inferring entire train/test set may require too much compute memory (depends on your system). In that case, divide into mini-batches.

        # Drive the network bottom to top
        prob_hid, samples_hid = self.rbm_stack["vis--hid"].get_h_given_v_dir(visible_set)
        prob_pen, samples_pen = self.rbm_stack["hid--pen"].get_h_given_v_dir(prob_hid)

        prob_pen_lbl = np.concatenate((samples_pen, lbl), axis=1)   # Use binary samples in first step

        for _ in range(self.n_gibbs_recog):
            prob_top, _ = self.rbm_stack["pen+lbl--top"].get_h_given_v(prob_pen_lbl)    # TODO use probs or samples?
            prob_pen_lbl, _ = self.rbm_stack["pen+lbl--top"].get_v_given_h(prob_top)    # TODO use probs or samples?

        predicted_lbl = prob_pen_lbl[:, -true_lbl.shape[1]:]
            
        print ("accuracy = %.2f%%"%(100.*np.mean(np.argmax(predicted_lbl,axis=1)==np.argmax(true_lbl,axis=1))))
        
        return

    def generate(self,true_lbl,name):
        
        """Generate data from labels

        Args:
          true_lbl: true labels shaped (number of samples, size of label layer)
          name: string used for saving a video of generated visible activations
        """
        
        n_sample = true_lbl.shape[0]
        n_label = true_lbl.shape[1]
        n_images = 20
        
        records = []
        fig,ax = plt.subplots(1,1,figsize=(3,3))
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        ax.set_xticks([]); ax.set_yticks([])

        lbl = true_lbl

        # [TODO TASK 4.2] fix the label in the label layer and run alternating Gibbs sampling in the top RBM. From the top RBM, drive the network \ 
        # top to the bottom visible layer (replace 'vis' from random to your generated vi

        # To init randomly the hidden layer, we propagate a random image bottom up and draw a sample from the probabilites
        #prob_hid, _ = self.rbm_stack['vis--hid'].get_h_given_v_dir(np.random.binomial(n=1, p=0.1, size=(1, 784)))
        prob_hid, _ = self.rbm_stack['vis--hid'].get_h_given_v_dir(self.image)
        _, samples_pen = self.rbm_stack['hid--pen'].get_h_given_v_dir(prob_hid)

        #samples_pen = np.random.binomial(n=1, p=0.05, size=(n_sample, self.sizes["hid"]))
        samples_pen_lbl = np.concatenate((samples_pen, true_lbl), axis=1)
        for _ in range(self.n_gibbs_gener):
            samples_pen_lbl[:, -n_label:] = true_lbl
            prob_top, samples_top = self.rbm_stack['pen+lbl--top'].get_h_given_v(samples_pen_lbl)
            probs_pen_lbl, samples_pen_lbl = self.rbm_stack['pen+lbl--top'].get_v_given_h(samples_top)

        for _ in range(n_images):
            prob_hid, samples_hid = self.rbm_stack['hid--pen'].get_v_given_h_dir(probs_pen_lbl[:, :-n_label])
            prob_vis, samples_vis = self.rbm_stack['vis--hid'].get_v_given_h_dir(samples_hid)

            records.append([ax.imshow(prob_vis.reshape(self.image_size), cmap="bwr", vmin=0, vmax=1, animated=True, interpolation=None)])
        vz_image(prob_vis.T)

        anim = stitch_video(fig, records).save("%s.generate%d.mp4"%(name,np.argmax(true_lbl)))
            
        return

    def train_greedylayerwise(self, vis_trainset, lbl_trainset, n_iterations):

        """
        Greedy layer-wise training by stacking RBMs. This method first tries to load previous saved parameters of the entire RBM stack. 
        If not found, learns layer-by-layer (which needs to be completed) .
        Notice that once you stack more layers on top of a RBM, the weights are permanently untwined.

        Args:
          vis_trainset: visible data shaped (size of training set, size of visible layer)
          lbl_trainset: label data shaped (size of training set, size of label layer)
          n_iterations: number of iterations of learning (each iteration learns a mini-batch)
        """

        try :

            self.loadfromfile_rbm(loc="trained_rbm",name="vis--hid")
            self.rbm_stack["vis--hid"].untwine_weights()            
            
            self.loadfromfile_rbm(loc="trained_rbm",name="hid--pen")
            self.rbm_stack["hid--pen"].untwine_weights()
            
            self.loadfromfile_rbm(loc="trained_rbm",name="pen+lbl--top")        

        except IOError :

            # [DONE TASK 4.2] use CD-1 to train all RBMs greedily
        
            print ("training vis--hid")
            self.rbm_stack["vis--hid"].cd1(vis_trainset, n_iterations)
            self.savetofile_rbm(loc="trained_rbm",name="vis--hid")

            print ("training hid--pen")
            prob_h, samples_h =  self.rbm_stack["vis--hid"].get_h_given_v(vis_trainset)
            self.rbm_stack["vis--hid"].untwine_weights()

            hid_trainset = samples_h    # TODO samples or probs
            self.rbm_stack["hid--pen"].cd1(hid_trainset, n_iterations)
            self.savetofile_rbm(loc="trained_rbm",name="hid--pen")            

            print ("training pen+lbl--top")
            prob_h, samples_h =  self.rbm_stack["hid--pen"].get_h_given_v(hid_trainset)
            self.rbm_stack["hid--pen"].untwine_weights()

            top_trainset = np.concatenate((samples_h, lbl_trainset[:samples_h.shape[0], :]), axis=1)
            self.rbm_stack["pen+lbl--top"].cd1(top_trainset, n_iterations)
            self.savetofile_rbm(loc="trained_rbm",name="pen+lbl--top")            

        return    

    def train_wakesleep_finetune(self, vis_trainset, lbl_trainset, n_iterations):

        """
        Wake-sleep method for learning all the parameters of network. 
        First tries to load previous saved parameters of the entire network.

        Args:
          vis_trainset: visible data shaped (size of training set, size of visible layer)
          lbl_trainset: label data shaped (size of training set, size of label layer)
          n_iterations: number of iterations of learning (each iteration learns a mini-batch)
        """
        
        print ("\ntraining wake-sleep..")

        try :
            
            self.loadfromfile_dbn(loc="trained_dbn",name="vis--hid")
            self.loadfromfile_dbn(loc="trained_dbn",name="hid--pen")
            self.loadfromfile_rbm(loc="trained_dbn",name="pen+lbl--top")
            
        except IOError :

            n_samples = vis_trainset.shape[0]
            n_label = lbl_trainset.shape[1]
            n_batches = np.ceil(n_samples / self.batch_size).astype(int)
            mini_batches = list(vis_trainset[self.batch_size * i: min(self.batch_size * (i + 1), n_samples)]
                                for i in range(n_batches))
            mini_batches_lbl = list(lbl_trainset[self.batch_size * i: min(self.batch_size * (i + 1), n_samples)]
                                for i in range(n_batches))

            for it in range(n_iterations):

                for i in range(n_batches):
                    samples_vis_wake = mini_batches[i]
                    samples_lbl_wake = mini_batches_lbl[i]

                    # [TODO TASK 4.3] wake-phase : drive the network bottom to top using fixing the visible and label data.
                    prob_hid_wake, samples_hid_wake = self.rbm_stack['vis--hid'].get_h_given_v_dir(samples_vis_wake)
                    prob_hid_wake, samples_pen_wake = self.rbm_stack['hid--pen'].get_h_given_v_dir(samples_hid_wake)
                    samples_pen_lbl_wake = np.concatenate((samples_pen_wake, samples_lbl_wake), axis=1)

                    # [TODO TASK 4.3] alternating Gibbs sampling in the top RBM for k='n_gibbs_wakesleep' steps, also store neccessary information for learning this RBM.
                    _, samples_top = self.rbm_stack['pen+lbl--top'].get_h_given_v(samples_pen_lbl_wake)
                    v_0 = samples_pen_lbl_wake
                    h_0 = samples_top
                    prob_pen_lbl, _ = self.rbm_stack['pen+lbl--top'].get_v_given_h(samples_top)
                    for _ in range(self.n_gibbs_wakesleep):
                        _, samples_top = self.rbm_stack['pen+lbl--top'].get_h_given_v(prob_pen_lbl)
                        prob_pen_lbl, samples_pen_lbl = self.rbm_stack['pen+lbl--top'].get_v_given_h(samples_top)

                    # [TODO TASK 4.3] sleep phase : from the activities in the top RBM, drive the network top to bottom.
                    samples_pen_sleep = samples_pen_lbl[:, :-n_label]
                    prob_hid_sleep, samples_hid_sleep = self.rbm_stack['hid--pen'].get_v_given_h_dir(samples_pen_sleep)
                    prob_vis_sleep, _ = self.rbm_stack['vis--hid'].get_v_given_h_dir(samples_hid_sleep)

                    # [TODO TASK 4.3] compute predictions : compute generative predictions from wake-phase activations, and recognize predictions from sleep-phase activations.
                    # Note that these predictions will not alter the network activations, we use them only to learn the directed connections.

                    # Generation hid->vis
                    gen_vis_target = samples_vis_wake
                    gen_vis_input = prob_hid_wake
                    gen_vis_prediction, _ = self.rbm_stack['vis--hid'].get_v_given_h_dir(gen_vis_input)

                    # Generation pen->hid
                    gen_hid_target = prob_hid_wake
                    gen_hid_input = prob_hid_wake
                    gen_hid_prediction, _ = self.rbm_stack['hid--pen'].get_v_given_h_dir(gen_hid_input)

                    # Top RBM
                    v_0 = v_0
                    h_0 = h_0
                    v_k = prob_pen_lbl
                    h_k, _ = self.rbm_stack['pen+lbl--top'].get_h_given_v(prob_pen_lbl)

                    # Recognition vis->hid
                    rec_vis_target = prob_hid_sleep
                    rec_vis_input = prob_vis_sleep
                    rec_vis_prediction, _ = self.rbm_stack['vis--hid'].get_h_given_v_dir(rec_vis_input)

                    # Recognition hid->pen
                    rec_hid_target = prob_pen_lbl[:, :-n_label]
                    rec_hid_input = prob_hid_sleep
                    rec_hid_prediction, _ = self.rbm_stack['hid--pen'].get_h_given_v_dir(rec_hid_input)


                    # [TODO TASK 4.3] update generative parameters : here you will only use 'update_generate_params' method from rbm class.
                    self.rbm_stack['vis--hid'].update_generate_params(gen_vis_input, gen_vis_target, gen_vis_prediction)
                    self.rbm_stack['hid--pen'].update_generate_params(gen_hid_input, gen_hid_target, gen_hid_prediction)

                    # [TODO TASK 4.3] update parameters of top rbm : here you will only use 'update_params' method from rbm class.
                    self.rbm_stack['pen+lbl--top'].update_top_params(v_0, h_0, v_k, h_k)

                    # [TODO TASK 4.3] update recognize parameters : here you will only use 'update_recognize_params' method from rbm class.
                    self.rbm_stack['vis--hid'].update_recognize_params(rec_vis_input, rec_vis_target, rec_vis_prediction)
                    self.rbm_stack['hid--pen'].update_recognize_params(rec_hid_input, rec_hid_target, rec_hid_prediction)

                    if i % 50 == 0: print("batch=%7d" % i)

                if it % self.print_period == 0 : print ("----iteration=%7d----"%it)
                        
            self.savetofile_dbn(loc="trained_dbn",name="vis--hid")
            self.savetofile_dbn(loc="trained_dbn",name="hid--pen")
            self.savetofile_rbm(loc="trained_dbn",name="pen+lbl--top")            

        return

    
    def loadfromfile_rbm(self,loc,name):
        
        self.rbm_stack[name].weight_vh = np.load("%s/rbm.%s.weight_vh.npy"%(loc,name))
        self.rbm_stack[name].bias_v    = np.load("%s/rbm.%s.bias_v.npy"%(loc,name))
        self.rbm_stack[name].bias_h    = np.load("%s/rbm.%s.bias_h.npy"%(loc,name))
        print ("loaded rbm[%s] from %s"%(name,loc))
        return
        
    def savetofile_rbm(self,loc,name):
        
        np.save("%s/rbm.%s.weight_vh"%(loc,name), self.rbm_stack[name].weight_vh)
        np.save("%s/rbm.%s.bias_v"%(loc,name),    self.rbm_stack[name].bias_v)
        np.save("%s/rbm.%s.bias_h"%(loc,name),    self.rbm_stack[name].bias_h)
        return
    
    def loadfromfile_dbn(self,loc,name):
        
        self.rbm_stack[name].weight_v_to_h = np.load("%s/dbn.%s.weight_v_to_h.npy"%(loc,name))
        self.rbm_stack[name].weight_h_to_v = np.load("%s/dbn.%s.weight_h_to_v.npy"%(loc,name))
        self.rbm_stack[name].bias_v        = np.load("%s/dbn.%s.bias_v.npy"%(loc,name))
        self.rbm_stack[name].bias_h        = np.load("%s/dbn.%s.bias_h.npy"%(loc,name))
        print ("loaded rbm[%s] from %s"%(name,loc))
        return
        
    def savetofile_dbn(self,loc,name):
        
        np.save("%s/dbn.%s.weight_v_to_h"%(loc,name), self.rbm_stack[name].weight_v_to_h)
        np.save("%s/dbn.%s.weight_h_to_v"%(loc,name), self.rbm_stack[name].weight_h_to_v)
        np.save("%s/dbn.%s.bias_v"%(loc,name),        self.rbm_stack[name].bias_v)
        np.save("%s/dbn.%s.bias_h"%(loc,name),        self.rbm_stack[name].bias_h)
        return
    
