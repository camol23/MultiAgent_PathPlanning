import numpy as np
import tensorflow as tf
from tensorflow import keras





class dense_np:
    def __init__(self, weights, bias):
        
        self.w = weights
        self.b = bias

        self.output = None


    def compute(self, input):        
        output = np.matmul(np.transpose(self.w), input, dtype=np.float32) + self.b
        # print("layer type ", self.output.dtype) # 32
        # return self.output

        return output



def relu_fnct(input):
    
    zero_mask = (input >= 0)
    zero_mask = zero_mask.astype(np.float32)
    output = zero_mask*input

    return output
    


def sigmoid_fnct(input):
    output = 1/(1 + np.exp(-input, dtype=np.float32))
    # print("sig type ", output.dtype) # 32

    return output


class model_dummy:
    def __init__(self):
        
        self.w_params = []
        self.b_params = []
        self.layers_list = []


    def initialization(self, w_params, b_params):
        
        self.w_params = w_params
        self.b_params = b_params

        # Layers init.
        for i, wi in enumerate(self.w_params):
            layer_i = dense_np(weights=wi, bias=self.b_params[i])

            self.layers_list.append(layer_i)
            
    
    def compute(self, input):
        '''
            Define the model
        '''

        fc_1 = self.layers_list[0].compute(input)
        relu_1 = relu_fnct(fc_1)
        fc_2 = self.layers_list[1].compute(relu_1)
        relu_2 = relu_fnct(fc_2)
        output = sigmoid_fnct(relu_2)

        return output


class model_all_18:
    def __init__(self):
        
        self.w_params = []
        self.b_params = []
        self.layers_list = []


    def initialization(self, w_params, b_params):
        
        self.w_params = w_params
        self.b_params = b_params

        # Layers init.
        for i, wi in enumerate(self.w_params):
            layer_i = dense_np(weights=wi, bias=self.b_params[i])

            self.layers_list.append(layer_i)
            
    
    def compute(self, input):
        '''
            Define the model
            keras.Input(shape=(9,))
        '''

        input_prev = input
        num_layers = len(self.layers_list)
        for i, layer_i in enumerate(self.layers_list):

            fc_i = layer_i.compute(input_prev)
            # fc_i_cp = np.copy(fc_i)

            if i != (num_layers-1):
                relu_i = relu_fnct(fc_i)
                # print("relu type ", relu_i.dtype) # 32
                # input_prev = np.copy(relu_i)
                input_prev = relu_i
            else:
                input_prev = fc_i


        output = sigmoid_fnct(input_prev)

        return output

