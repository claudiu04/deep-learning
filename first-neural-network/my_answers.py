import numpy as np
import sys

class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        np.random.seed(123)
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes**-0.5, 
                                       (self.input_nodes, self.hidden_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes**-0.5, 
                                       (self.hidden_nodes, self.output_nodes))
        self.lr = learning_rate
        
        self.activation_function = lambda x : 1 / (1 + np.exp(-x))
                  

    def train(self, features, targets):
        ''' Train the network on batch of features and targets. 
        
            Arguments
            ---------
            
            features: 2D array, each row is one data record, each column is a feature
            targets: 1D array of target values
        
        '''

#        features = np.array(features, ndmin=2)
#        targetss = np.array(targets, ndmin=2)
        if len(features.shape) == 1:
            features = features[:, None]
        if len(targets.shape) == 1:
            targets = targets[:, None]

        n_records = features.shape[0]
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)

        final_outputs, hidden_outputs = self.forward_pass_train(features)
        delta_weights_i_h, delta_weights_h_o = self.backpropagation(
            final_outputs, hidden_outputs, features, targets, 
            np.zeros(self.weights_input_to_hidden.shape), 
            np.zeros(self.weights_hidden_to_output.shape))
        self.update_weights(delta_weights_i_h, delta_weights_h_o, n_records)


    def forward_pass_train(self, X):
        ''' Implement forward pass here 
         
            Arguments
            ---------
            X: features batch

        '''
        #### Implement the forward pass here ####
        ### Forward pass ###

        hidden_inputs = np.dot(X, self.weights_input_to_hidden)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output)
        final_outputs = final_inputs

        return final_outputs, hidden_outputs

    def backpropagation(self, final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o):
        ''' Implement backpropagation
         
            Arguments
            ---------
            final_outputs: output from forward pass
            y: target (i.e. label) batch
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers

        '''
        #### Implement the backward pass here ####
        ### Backward pass ###

        X = np.array(X, ndmin=2)
        y = np.array(y, ndmin=2)
        hidden_outputs = np.array(hidden_outputs, ndmin=2)
    
        # Add dimension if y is 1-dimensional
#        yy = y[:, None] if len(y.shape) == 1 or len(y.shape) < len(final_outputs.shape) else y
#        fo = final_outputs[:, None] if len(final_outputs.shape) == 1 else final_outputs 
#        X2 = X[None, :] if len(X.shape) == 1 else X
#        ho = hidden_outputs[None, :] if len(hidden_outputs.shape) == 1 else hidden_outputs

        yy = y
        fo = final_outputs
        X2 = X
        ho = hidden_outputs
 
        error = yy - fo

        output_error_term = error

        hidden_error = np.dot(output_error_term, self.weights_hidden_to_output.T)

        hidden_error_term = hidden_error * hidden_outputs * (1 - hidden_outputs)

        try:
        
            # Weight step (input to hidden)
            delta_weights_i_h += np.dot(X2.T, hidden_error_term)

            # Weight step (hidden to output)
            delta_weights_h_o += np.dot(ho.T, output_error_term)

#        self.input_nodes = input_nodes
#        self.hidden_nodes = hidden_nodes
#        self.output_nodes = output_nodes
#
#        # Initialize weights
#        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes**-0.5,
#                                       (self.input_nodes, self.hidden_nodes))
#
#        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes**-0.5,
#                                       (self.hidden_nodes, self.output_nodes))


        except ValueError as ve:
            s = ve.__str__() + "\n"
            s = s + "y.shape = %s\n" % str(y.shape)
            s  = s + "yy.shape = %s\n" % str(yy.shape)
            s = s + "final_outputs.shape = %s\n" % str(final_outputs.shape)
            s = s + "fo.shape = %s\n" % str(fo.shape)
            s = s + "output_error_term.shape = %s\n" % str(output_error_term.shape)
            s = s + "hidden_error.shape = %s\n" % str(hidden_error.shape)
            s = s + "hidden_error_term.shape = %s\n" % str(hidden_error_term.shape)
            s = s + "X.T.shape = %s\n" % str(X.T.shape)
            s = s + "delta_weights_i_h += np.dot(X.T, hidden_error_term)\n"
            s = s + "self.input_nodes = %s\n" % str(self.input_nodes)
            s = s + "self.hidden_nodes = %s\n" % str(self.hidden_nodes)
            s = s + "self.weights_input_to_hidden.shape = %s\n" % str(self.weights_input_to_hidden.shape)
            s = s + "self.weights_hidden_to_output.shape = %s\n" % str(self.weights_hidden_to_output.shape)
            s = s + "delta_weights_i_h.shape = %s\n" % str(delta_weights_i_h.shape)
            s = s + "delta_weights_h_i.shape = %s" % str(delta_weights_h_o.shape)

            raise ValueError(s)


        # TODO: Update weights
#        delta_weights_i_h = delta_weights_i_h
#        delta_weights_h_o = delta_weights_h_o        
       
        return delta_weights_i_h, delta_weights_h_o

    def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):
        ''' Update weights on gradient descent step
         
            Arguments
            ---------
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers
            n_records: number of records

        '''
        self.weights_hidden_to_output += self.lr * delta_weights_h_o / n_records
        self.weights_input_to_hidden += self.lr * delta_weights_i_h / n_records

    def run(self, features):
        ''' Run a forward pass through the network with input features 
        
            Arguments
            ---------
            features: 1D array of feature values
        '''
        num_records, num_features = features.shape

        final_outputs, _ = self.forward_pass_train(features)
        return final_outputs

#########################################################
# Set your hyperparameters here
##########################################################
iterations = 6000
learning_rate = 0.625
hidden_nodes = 10
output_nodes = 1
