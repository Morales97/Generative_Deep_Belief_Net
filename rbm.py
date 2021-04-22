from util import *
import pdb

class RestrictedBoltzmannMachine():
    '''
    For more details : A Practical Guide to Training Restricted Boltzmann Machines https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf
    '''
    def __init__(self, ndim_visible, ndim_hidden, is_bottom=False, image_size=[28,28], is_top=False, n_labels=10, batch_size=10):

        """
        Args:
          ndim_visible: Number of units in visible layer.
          ndim_hidden: Number of units in hidden layer.
          is_bottom: True only if this rbm is at the bottom of the stack in a deep belief net. Used to interpret visible layer as image data with dimensions "image_size".
          image_size: Image dimension for visible layer.
          is_top: True only if this rbm is at the top of stack in deep beleif net. Used to interpret visible layer as concatenated with "n_label" unit of label data at the end. 
          n_label: Number of label categories.
          batch_size: Size of mini-batch.
        """
       
        self.ndim_visible = ndim_visible

        self.ndim_hidden = ndim_hidden

        self.is_bottom = is_bottom

        if is_bottom : self.image_size = image_size
        
        self.is_top = is_top

        if is_top : self.n_labels = 10

        self.batch_size = batch_size        
                
        self.delta_bias_v = 0

        self.delta_weight_vh = 0

        self.delta_bias_h = 0

        self.bias_v = np.random.normal(loc=0.0, scale=0.01, size=self.ndim_visible)

        self.weight_vh = np.random.normal(loc=0.0, scale=0.01, size=(self.ndim_visible, self.ndim_hidden))

        self.bias_h = np.random.normal(loc=0.0, scale=0.01, size=self.ndim_hidden)
        
        self.delta_weight_v_to_h = 0

        self.delta_weight_h_to_v = 0        
        
        self.weight_v_to_h = None
        
        self.weight_h_to_v = None

        self.learning_rate = 0.01
        
        self.momentum = 0.7

        self.weight_decay_rate = 0.0001

        self.print_period = 1

        # Sparsity parameters
        self.q_estimate_prob = 0.5 * np.ones([ndim_hidden])
        self.sparsity_target = 0.2 * np.ones([ndim_hidden])
        self.q_update_coef = 0.9
        self.sparsity_cost = 0.001    # Try different values

        self.rf = { # receptive-fields. Only applicable when visible layer is input data
            "period" : 9, #5000, # iteration period to visualize
            "grid" : [5,5], # size of the grid
            "ids" : np.random.randint(0,self.ndim_hidden,25) # pick some random hidden units
            }
        
        return

        
    def cd1(self, visible_trainset, n_iterations=10000):
        
        """Contrastive Divergence with k=1 full alternating Gibbs sampling

        Args:
          visible_trainset: training data for this rbm, shape is (size of training set, size of visible layer)
          n_iterations: number of iterations of learning (each iteration learns a mini-batch)
        """

        print ("learning CD1")
        
        n_samples = visible_trainset.shape[0]
        n_batches = np.ceil(n_samples/self.batch_size).astype(int)
        mini_batches = list(visible_trainset[self.batch_size * i : min(self.batch_size * (i+1), n_samples)]
                            for i in range(n_batches))

        reconstruction_error_batch = []
        reconstruction_error_iteration = np.zeros(n_iterations)
        for it in range(n_iterations):

            for i in range(n_batches):
                samples_v0 = mini_batches[i]
                prob_h0, samples_h0 = self.get_h_given_v(samples_v0)
                prob_v1, samples_v1 = self.get_v_given_h(samples_h0)
                prob_h1, samples_h1 = self.get_h_given_v(prob_v1) # TODO use samples_v1 or prob_v1? For what I understand from book in 3.2, use prob

                self.update_params(samples_v0, samples_h0, prob_v1, prob_h1) # TODO slides say to use prob_v0 ??

                error = np.mean(np.square(prob_v1 - samples_v0))
                reconstruction_error_batch.append(error)
                reconstruction_error_iteration[it] += error / n_batches
	    # [ TASK 4.1] run k=1 alternating Gibbs sampling : v_0 -> h_0 ->  v_1 -> h_1.
            # you may need to use the inference functions 'get_h_given_v' and 'get_v_given_h'.
            # note that inference methods returns both probabilities and activations (samples from probablities) and you may have to decide when to use what.

            # [ TASK 4.1] update the parameters using function 'update_params'
            
            # visualize once in a while when visible layer is input images
            
            if it % self.rf["period"] == 0 and self.is_bottom:
                # DM: visualize 25 pictures, each of them is the output when activating one random hidden unit
                viz_rf(weights=self.weight_vh[:,self.rf["ids"]].reshape((self.image_size[0],self.image_size[1],-1)), it=it, grid=self.rf["grid"])

            # print progress
            
            if it % self.print_period == 0 :
                #histogram_weights(self.weight_vh)
                print ("iteration=%7d recon_loss=%4.4f"%(it, np.linalg.norm(visible_trainset - visible_trainset)))

        plot_recons_error(reconstruction_error_batch, 'Number of mini-batches trained')
        plot_recons_error(reconstruction_error_iteration, 'Number of epochs')

        return

    def update_params(self, v_0, h_0, v_k, h_k):

        """Update the weight and bias parameters.

        You could also add weight decay and momentum for weight updates.

        Args:
           v_0: activities or probabilities of visible layer (data to the rbm)
           h_0: activities or probabilities of hidden layer
           v_k: activities or probabilities of visible layer
           h_k: activities or probabilities of hidden layer
           all args have shape (size of mini-batch, size of respective layer)
        """

        # [ TASK 4.1] get the gradients from the arguments (replace the 0s below) and update the weight and bias parameters
        # TODO: try weight decay and momentum

        # strategy = "dont_use_momentum"
        # strategy = "use_momentum_dani"
        # strategy = "use_momentum_book"
        strategy = "use_momentum_book_weight_decay_pepino"
        encourage_sparcity = True

        n_mini_batch = v_0.shape[0]

        statistics_data = np.matmul(v_0.T, h_0)
        statistics_model = np.matmul(v_k.T, h_k)
        step_weight_vh = 1/n_mini_batch * (statistics_data - statistics_model)

        step_bias_v = 1/n_mini_batch * np.sum(v_0 - v_k, axis=0)
        step_bias_h = 1/n_mini_batch * np.sum(h_0 - h_k, axis=0)

        # Dont use momentum or learning rate. BAD RESULTS
        if strategy == "dont_use_momentum":
            self.bias_v += step_bias_v
            self.weight_vh += step_weight_vh
            self.bias_h += step_bias_h

        # This is what I understand for momentum. Then you have to multiply by learning rate
        if strategy == "use_momentum_dani":
            self.delta_bias_v = self.momentum * self.delta_bias_v + (1 - self.momentum) * step_bias_v
            self.delta_weight_vh = self.momentum * self.delta_weight_vh + (1 - self.momentum) * step_weight_vh
            self.delta_bias_h = self.momentum * self.delta_bias_h + (1 - self.momentum) * step_bias_h

            self.bias_v += self.learning_rate * self.delta_bias_v
            self.weight_vh += self.learning_rate * self.delta_weight_vh
            self.bias_h += self.learning_rate * self.delta_bias_h

        # This is the formula from the book.
        if strategy == "use_momentum_book":
            self.delta_bias_v = self.momentum * self.delta_bias_v + self.learning_rate * step_bias_v
            self.delta_weight_vh = self.momentum * self.delta_weight_vh + self.learning_rate * step_weight_vh
            self.delta_bias_h = self.momentum * self.delta_bias_h + self.learning_rate * step_bias_h

            self.bias_v += self.delta_bias_v
            self.weight_vh += self.delta_weight_vh
            self.bias_h += self.delta_bias_h

        if strategy == "use_momentum_book_weight_decay_pepino":
            step_weight_vh = step_weight_vh - self.weight_decay_rate * self.weight_vh

            self.delta_bias_v = self.momentum * self.delta_bias_v + self.learning_rate * step_bias_v
            self.delta_weight_vh = self.momentum * self.delta_weight_vh + self.learning_rate * step_weight_vh
            self.delta_bias_h = self.momentum * self.delta_bias_h + self.learning_rate * step_bias_h

            self.bias_v += self.delta_bias_v
            self.weight_vh += self.delta_weight_vh
            self.bias_h += self.delta_bias_h

        if encourage_sparcity:
            mean_prob_h = np.mean(h_k, axis=0)
            self.q_estimate_prob = self.q_update_coef * self.q_estimate_prob + (1-self.q_update_coef) * mean_prob_h
            self.bias_h -= self.sparsity_cost * (self.q_estimate_prob - self.sparsity_target)
            sparsity_weight_penalty = np.repeat(np.reshape((self.q_estimate_prob - self.sparsity_target), (-1,1)), self.ndim_visible, axis=1).T
            self.weight_vh -= self.sparsity_cost * sparsity_weight_penalty

        return

    def get_h_given_v(self,visible_minibatch):
        
        """Compute probabilities p(h|v) and activations h ~ p(h|v) 

        Uses undirected weight "weight_vh" and bias "bias_h"
        
        Args: 
           visible_minibatch: shape is (size of mini-batch, size of visible layer)
        Returns:        
           tuple ( p(h|v) , h) 
           both are shaped (size of mini-batch, size of hidden layer)
        """
        
        assert self.weight_vh is not None
        # TASK 4.1 compute probabilities and activations (samples from probabilities) of hidden layer (replace the zeros below)

        x = self.bias_h + np.matmul(visible_minibatch, self.weight_vh)
        prob_h_given_v = sigmoid(x)
        activations = sample_binary(prob_h_given_v)

        return prob_h_given_v, activations


    def get_v_given_h(self,hidden_minibatch):
        
        """Compute probabilities p(v|h) and activations v ~ p(v|h)

        Uses undirected weight "weight_vh" and bias "bias_v"
        
        Args: 
           hidden_minibatch: shape is (size of mini-batch, size of hidden layer)
        Returns:        
           tuple ( p(v|h) , v) 
           both are shaped (size of mini-batch, size of visible layer)
        """
        assert self.weight_vh is not None

        x = self.bias_v + np.matmul(hidden_minibatch, self.weight_vh.T)

        if self.is_top:
            """
            Here visible layer has both data and labels. Compute total input for each unit (identical for both cases), \ 
            and split into two parts, something like support[:, :-self.n_labels] and support[:, -self.n_labels:]. \
            Then, for both parts, use the appropriate activation function to get probabilities and a sampling method \
            to get activities. The probabilities as well as activities can then be concatenated back into a normal visible layer.
            """
            prob_v_given_h = np.empty_like(x)
            units = x[:, :-self.n_labels]
            labels = x[:, -self.n_labels:]
            # Use adequate activation function for each part
            prob_v_given_h[:, :-self.n_labels] = sigmoid(units)
            prob_v_given_h[:, -self.n_labels:] = softmax(labels)
            # Get all activations
            activations = sample_binary(prob_v_given_h)

        else:
            prob_v_given_h = sigmoid(x)
            activations = sample_binary(prob_v_given_h)

        return prob_v_given_h, activations


    
    """ rbm as a belief layer : the functions below do not have to be changed until running a deep belief net """

    

    def untwine_weights(self):
        
        self.weight_v_to_h = np.copy( self.weight_vh )
        self.weight_h_to_v = np.copy( np.transpose(self.weight_vh) )
        self.weight_vh = None

    def get_h_given_v_dir(self, visible_minibatch):

        """Compute probabilities p(h|v) and activations h ~ p(h|v)

        Uses directed weight "weight_v_to_h" and bias "bias_h"
        
        Args: 
           visible_minibatch: shape is (size of mini-batch, size of visible layer)
        Returns:        
           tuple ( p(h|v) , h) 
           both are shaped (size of mini-batch, size of hidden layer)
        """
        
        assert self.weight_v_to_h is not None

        # [DONE TASK 4.2] perform same computation as the function 'get_h_given_v' but with directed connections (replace the zeros below)

        x = self.bias_h + np.matmul(visible_minibatch, self.weight_v_to_h)
        prob_h_given_v = sigmoid(x)
        activations = sample_binary(prob_h_given_v)

        return prob_h_given_v, activations


    def get_v_given_h_dir(self, hidden_minibatch):


        """Compute probabilities p(v|h) and activations v ~ p(v|h)

        Uses directed weight "weight_h_to_v" and bias "bias_v"
        
        Args: 
           hidden_minibatch: shape is (size of mini-batch, size of hidden layer)
        Returns:        
           tuple ( p(v|h) , v) 
           both are shaped (size of mini-batch, size of visible layer)
        """
        
        assert self.weight_h_to_v is not None

        x = self.bias_v + np.matmul(hidden_minibatch, self.weight_h_to_v)

        if self.is_top:

            """
            Here visible layer has both data and labels. Compute total input for each unit (identical for both cases), \ 
            and split into two parts, something like support[:, :-self.n_labels] and support[:, -self.n_labels:]. \
            Then, for both parts, use the appropriate activation function to get probabilities and a sampling method \
            to get activities. The probabilities as well as activities can then be concatenated back into a normal visible layer.
            """
            
            # DONE TASK 4.2] Note that even though this function performs same computation as 'get_v_given_h' but with directed connections,
            # this case should never be executed : when the RBM is a part of a DBN and is at the top, it will have not have directed connections.
            # Appropriate code here is to raise an error (replace pass below)
            raise Exception("This function should never be called")

        else:
            # [DONE TASK 4.2] performs same computaton as the function 'get_v_given_h' but with directed connections (replace the pass and zeros below)
            prob_v_given_h = sigmoid(x)
            activations = sample_binary(prob_v_given_h)
            
        return prob_v_given_h, activations
        
    def update_generate_params(self,inps,trgs,preds):
        
        """Update generative weight "weight_h_to_v" and bias "bias_v"
        
        Args:
           inps: activities or probabilities of input unit
           trgs: activities or probabilities of output unit (target)
           preds: activities or probabilities of output unit (prediction)
           all args have shape (size of mini-batch, size of respective layer)
        """

        # [TODO TASK 4.3] find the gradients from the arguments (replace the 0s below) and update the weight and bias parameters.
        n_mini_batch = inps.shape[0]

        delta_weight_h_to_v = 1/n_mini_batch * inps.T @ (trgs - preds)
        delta_bias_v = 1/n_mini_batch * np.sum((trgs - preds), axis=0)
        
        self.weight_h_to_v += self.learning_rate * delta_weight_h_to_v
        self.bias_v += self.learning_rate * delta_bias_v
        
        return

    def update_top_params(self, v_0, h_0, v_k, h_k):
        n_mini_batch = v_0.shape[0]

        statistics_data = np.matmul(v_0.T, h_0)
        statistics_model = np.matmul(v_k.T, h_k)
        step_weight_vh = 1/n_mini_batch * (statistics_data - statistics_model)
        step_bias_v = 1/n_mini_batch * np.sum(v_0 - v_k, axis=0)
        step_bias_h = 1/n_mini_batch * np.sum(h_0 - h_k, axis=0)

        self.weight_vh += self.learning_rate * step_weight_vh
        self.bias_h += self.learning_rate * step_bias_h
        self.bias_v += self.learning_rate * step_bias_v
    
    def update_recognize_params(self,inps,trgs,preds):
        
        """Update recognition weight "weight_v_to_h" and bias "bias_h"
        
        Args:
           inps: activities or probabilities of input unit
           trgs: activities or probabilities of output unit (target)
           preds: activities or probabilities of output unit (prediction)
           all args have shape (size of mini-batch, size of respective layer)
        """

        # [TODO TASK 4.3] find the gradients from the arguments (replace the 0s below) and update the weight and bias parameters.
        n_mini_batch = inps.shape[0]

        delta_weight_v_to_h = 1/n_mini_batch * inps.T @ (trgs - preds)
        delta_bias_h = 1/n_mini_batch * np.sum((trgs - preds), axis=0)

        self.weight_v_to_h += self.learning_rate * delta_weight_v_to_h
        self.bias_h += self.learning_rate * delta_bias_h

        return    
