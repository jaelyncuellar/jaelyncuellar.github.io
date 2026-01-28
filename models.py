import nn
from nn import Constant
import numpy as np 


class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        # perceptron weights. 1xdimensions
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w


    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        # nn.DotProduct(features, weights)
        dotProduct = nn.DotProduct(x, self.w)
        return dotProduct



    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"

        dotProduct = PerceptronModel.run(self, x)
        val = nn.as_scalar(dotProduct)
        if val < 0: # is negative 
            return -1
        return 1 # non-neagtive 


    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        batch_size = 1 

        updated = True

        while updated == True:
            updated = False 
            for x,correct_label in dataset.iterate_once(batch_size): 
                predicted_label = PerceptronModel.get_prediction(self, x)
                if nn.as_scalar(correct_label) != predicted_label: # if misclassified 
                    direction = nn.Constant(x.data * nn.as_scalar(correct_label)) # Constant node 
                    self.w.update(direction, multiplier= 1) # update weights 
                    updated = True

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"

        # num neurons in 1 layer
        self.hidden_layer_size = 128 # 10-400
        self.lr = 0.05

        # batch_size = 10 # 1-size dataset. % dataset = 0 
        # learning_rate = 0.5 #0.001-1.0
        # num_hidden_layers = 2 #1-3

        self.w1 = nn.Parameter(1, self.hidden_layer_size)
        self.b1 = nn.Parameter(1, self.hidden_layer_size )

        self.w2 = nn.Parameter(self.hidden_layer_size, self.hidden_layer_size//2)
        self.b2 = nn.Parameter(1, self.hidden_layer_size//2)

        self.w3 = nn.Parameter(self.hidden_layer_size//2, 1)
        self.b3 = nn.Parameter(1, 1)

        self.params = [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3]





    # polynomial linear regression predicts Y based on single predictor X 
    def run(self, x): # forward propagation 
        # each hidden layer has its own forward pass 
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            # x is the input data fed into neural network 
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
            # model's goal is to predict the corresponding output vals in data 
        """
        "*** YOUR CODE HERE ***"

        # forward pass thru the network -- propagating 
        # nn.Linear = applies linear transformation (matrix mult.)
        # nn.AddBias(features, bias)
        # nn.ReLu(features)

        # xw1 = nn.Linear(x, self.w1)
        # predicted_w1 = nn.AddBias(xw1, self.b1)
        # h1 = nn.ReLU(predicted_w1)
        # xw2 = nn.Linear(h1, self.w2)
        # predicted_w2 = nn.AddBias(xw2, self.b2)
        # h2 = nn.ReLU(predicted_w2)
        # xw3 = nn.Linear(h2, self.w3)
        # predicted_w3 = nn.AddBias(xw3, self.b3)
        # output = nn.ReLU(predicted_w3)

        h1 = nn.ReLU(nn.AddBias(nn.Linear(x, self.w1), self.b1))
        h2 = nn.ReLU(nn.AddBias(nn.Linear(h1, self.w2), self.b2))
        output = nn.AddBias(nn.Linear(h2, self.w3), self.b3)

        return output


    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1) # predicted
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"

        predicted = self.run(x)
        return nn.SquareLoss(predicted,y) 
    


    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        # train model using gradient based updates 
        batch_size = 50

        loss = float('inf')

        while (loss >= 0.018):

            for x, y in dataset.iterate_once(batch_size): 

                # compute loss 
                loss = self.get_loss(x, y)
                print(nn.as_scalar(loss))

                gradients = nn.gradients(loss, self.params)
                
                loss = nn.as_scalar(loss)

                #update parameters 
                for i in range(len(self.params)): 
                    self.params[i].update(gradients[i], -self.lr)

            # loss = nn.as_scalar(loss)





class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"

        self.w1 = nn.Parameter(784, 256)
        self.b1 = nn.Parameter(1,256)

        self.w2 = nn.Parameter(256, 128)
        self.b2 = nn.Parameter(1,128)

        self.w3 = nn.Parameter(128, 10)
        self.b3 = nn.Parameter(1,10)

        self.lr = 0.05

        self.params = [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3]


    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)

        # for prediction 
        """
        "*** YOUR CODE HERE ***"

        l1 = nn.ReLU(nn.AddBias(nn.Linear(x, self.w1), self.b1))
        l2 = nn.ReLU(nn.AddBias(nn.Linear(l1, self.w2), self.b2))
        l3 = nn.AddBias(nn.Linear(l2, self.w3), self.b3)
        return l3


    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"

        logits = self.run(x)
        return nn.SoftmaxLoss(logits,y)

    def train(self, dataset):
        """
        Trains the model.

        # the model learns 
        # updates model's parameters 
        # learns to min. predefined loss fn 
        # forward and backward passes thru network, 
            # computing gradients & update weights to min loss 

        # for optimizing the parameters 
        
        """
        "*** YOUR CODE HERE ***"

        batch_size = 50
        num_epochs = 5
        for epoch in range(num_epochs): 

                while (dataset.get_validation_accuracy() < 0.97):

                    for x, y in dataset.iterate_once(batch_size): 

                        loss = self.get_loss(x, y)

                        grad_params = nn.gradients(loss, self.params)

                        for i in range(len(grad_params)): 
                            self.params[i].update(grad_params[i], -self.lr)





class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"

        self.hidden_size = 128 # 10-400

        self.wxh = nn.Parameter(self.num_chars, self.hidden_size)
        self.whh = nn.Parameter(self.hidden_size, self.hidden_size)
        self.why = nn.Parameter(self.hidden_size, 5) # what should this size be ? 

        self.bh = nn.Parameter(1, self.hidden_size)
        self.by = nn.Parameter(1, 5) # 5 langs 

        self.params = [self.wxh, self.whh, self.why, self.bh, self.by]

        self.lr = 0.05



    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"

        # what is batch_size? 
        # what is d? 
        h = nn.Parameter(len(xs), self.hidden_size)

        z_initial = nn.AddBias(nn.Linear(xs[0], self.wxh), self.bh)
        h = nn.ReLU(z_initial)

        for L in range(1, len(xs)):
            x = xs[L] 
            z_rec = nn.AddBias(nn.Add(nn.Linear(x, self.wxh), nn.Linear(h, self.whh)), self.bh)
            h = nn.ReLU(z_rec)

        mult = nn.Linear(h, self.why)
        logits = nn.AddBias(mult, self.by)

        return logits





    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"

        yhat = self.run(xs)
        return nn.SoftmaxLoss(yhat,y) 



    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"


        batch_size = 50
        num_epochs = 5
        for epoch in range(num_epochs): 

                while (dataset.get_validation_accuracy() < 0.85):

                    for xs, y in dataset.iterate_once(batch_size): 

                        loss = self.get_loss(xs, y)

                        grad_params = nn.gradients(loss, self.params)

                        for i in range(len(grad_params)): 
                            self.params[i].update(grad_params[i], -self.lr)

