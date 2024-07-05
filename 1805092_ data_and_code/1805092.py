import torchvision.datasets as ds
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import copy
from sklearn.model_selection import train_test_split
import pickle
from sklearn.metrics import f1_score

from sklearn.metrics import confusion_matrix
import seaborn as sn


# Function to perform one-hot encoding

def one_hot_encode(letter):
    # Define the alphabet
    alphabet = 'abcdefghijklmnopqrstuvwxyz'

    # Create a dictionary to map each letter to its corresponding index
    letter_to_index = {letter: index for index, letter in enumerate(alphabet)}

    # Check if the input letter is in the alphabet
    if letter in letter_to_index:
        one_hot_vector = np.zeros(len(alphabet))
        one_hot_vector[letter_to_index[letter]] = 1
        return one_hot_vector
    else:
        # Handle the case where the input letter is not in the alphabet
        raise ValueError(f"Invalid letter: {letter}. The letter must be in the alphabet.")


def load_preprocess_training_dataset():
    train_validation_dataset = ds.EMNIST(root='./data', split='letters',
                                train=True,
                                transform=transforms.ToTensor(),
                                download=True)


    
    
    # Get the number of samples in the dataset
   # num_samples = len(train_validation_dataset)
    #print(f"Number of samples in the dataset: {num_samples}")

    # Show 2 example
    #for i in range(2):
    #    index = np.random.randint(0, num_samples)
    #    image, label = train_validation_dataset[index]
    #    print(f"Image shape: {image.shape}")
    #   print(f"Type of the image: {type(image)}")
    #   image = image.squeeze()
    #    image = image.numpy()
    #    image = image.T
    #    plt.imshow(image, cmap="gray")
    #    plt.show()

    
    letters = train_validation_dataset.classes[1:]
    labels = copy.deepcopy(train_validation_dataset.targets) -1
   # labels = np.array([one_hot_encode(label) for label in labels])
    labels = np.array(labels)
    train_set = train_validation_dataset.data.view(len(train_validation_dataset),28,28).float()
    train_set = train_set.reshape(len(train_validation_dataset),784)
    train_set = train_set.numpy()
    train_set= train_set/255.0
    
    
    return train_set, labels, letters
    
def load_preprocess_testing_dataset():
    independent_test_dataset = ds.EMNIST(root='./data',
                        split='letters',
                                train=False,
                                transform=transforms.ToTensor())
    
    test_set= independent_test_dataset.data.view(len(independent_test_dataset),28,28).float()
    test_set = test_set.reshape(len(independent_test_dataset),784)
    test_set = test_set.numpy()
    test_set= test_set/255.0
    letters = independent_test_dataset.classes[1:]
    
    labels = copy.deepcopy(independent_test_dataset.targets) -1
   # labels = np.array([one_hot_encode(label) for label in labels])
    labels = np.array(labels)
    
    return test_set, labels, letters
    


class dense_layer:
    def __init__(self, input_nums, output_nums):
        # Xavier initialization
        self.weights = np.sqrt(2 / (input_nums + output_nums)) * np.random.randn(input_nums, output_nums)
        #self.weights = np.random.randn(input_nums, output_nums) * 0.01
        self.biases = np.zeros((1, output_nums))
        self.d_weights = np.zeros((input_nums, output_nums))
        self.d_biases = np.zeros((1, output_nums))

    def forward(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.biases
        return self.output

    def backward(self, d_values):
        self.learning_rate = 0.005
        self.d_weights = np.dot(self.input.T, d_values)
        self.d_biases = np.sum(d_values, axis=0, keepdims=True)
      #  print(" d_values shape ",d_values.shape)
       # print(" weights shape ",self.weights.shape)
        self.d_inputs = np.dot(d_values, self.weights.T)
        self.weights -= self.d_weights* self.learning_rate
        self.biases -= self.d_biases* self.learning_rate
        return self.d_inputs

    
    

class relu_activation:
    def forward(self, input):
        self.input = input
        self.output = np.maximum(0, input)
    
    def backward(self, dvalues):
        dv = dvalues.copy()
        self.d_inputs = np.maximum(0, dv)
        
    
class activation_softmax:
    def forward(self, inputs):
        self.inputs = inputs
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)
    
    def backward(self, d_values):
        self.d_inputs = np.empty_like(d_values)
        for index, (output_single, dvalues_single) in enumerate(zip(self.output, d_values)):
            output_single = output_single.reshape(-1, 1)
            mat_jacobian = np.diagflat(output_single) - np.dot(output_single, output_single.T)
            self.d_inputs[index] = np.dot(mat_jacobian, dvalues_single)
    



class layer_dropout:
    def __init__(self, rate):
        self.rate = rate
    
    def forward(self, input):
        self.input = input
        self.binary_mask = np.random.binomial(1, self.rate, size=input.shape) / self.rate
        self.output = input * self.binary_mask
    
    def backward(self, dvalues):
        self.d_inputs = dvalues * self.binary_mask
        
        
class loss:
    def __init__(self, loss_function):
        self.loss_function = loss_function

    def calculate(self, output, y):
        sample_losses = self.loss_function.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss
        
 
    
class loss_categorical_crossentropy(loss):
    def __init__(self):
        super().__init__(self)
        
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        clipped_y_pred = np.clip(y_pred, 1e-7, 1-1e-7)
        if len(y_true.shape) == 1:
            confidences_correct = clipped_y_pred[range(samples), y_true]
        elif len(y_true.shape) == 2:
            confidences_correct = np.sum(clipped_y_pred * y_true, axis=1)
        negative_log_likelihoods = -np.log(confidences_correct )
        return negative_log_likelihoods
    
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
       
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        # Copy so we can safely modify
        self.dinputs = dvalues.copy()
        # Calculate gradient    
        self.dinputs[range(samples), y_true] -= 1
        # Normalize gradient
        self.dinputs = self.dinputs / samples
        return self.dinputs
    
        
        

class model_FNN:
    def __init__(self):
        self.layers = []
        self.cross_enthalpy_loss = loss_categorical_crossentropy()
        
    
    def add_a_layer(self, layer):
        self.layers.append(layer)

    def forward(self, input):
        for layer in self.layers:
            layer.forward(input)
            input = layer.output
        self.output = input
        return self.output
    
    def backward(self, d_values):
        for layer in reversed(self.layers):
            layer.backward(d_values)
            d_values = layer.d_inputs
        return d_values
    

    def get_full_dataset_loss(self, x, y):
        samples = len(x)
        output = self.forward(x)
        return self.cross_enthalpy_loss.calculate(output, y)
    

        
    def prediction_of_whole_dataset(self, x):
        softmax_output = self.forward(x)
        predicted_labels = np.argmax(softmax_output, axis=1)
        return predicted_labels
    

    def get_accuracy(self, predicted_labels, actual_labels):
        return np.sum(predicted_labels == actual_labels) / len(actual_labels)
    
    def fit_minibatch(self, x, y, x_validation, y_validation,mini_batch_size, epochs,  learning_rate):
        data_x = []
        data_y = []
        train_loss = []
        val_loss = []
        train_accuracy = [] 
        val_accuracy = []
        validation_f1_macro_score = []
        max_f1score = 0
        best_model = None
        for e in range(epochs):
            indices = np.random.permutation(len(x))
            shuffled_x = x[indices]
            shuffled_y = y[indices]
            for i in range(0, len(shuffled_x), mini_batch_size):
                batch_x = shuffled_x[i:i + mini_batch_size]
                batch_y = shuffled_y[i:i + mini_batch_size]
               # print("batch_x shape ", batch_x.shape)
               # print("batch_y shape ", batch_y.shape)

                # Forward pass
                output = self.forward(batch_x)

                # Calculate loss
                loss = self.cross_enthalpy_loss.calculate(output, batch_y)
                

                # Backward pass
                self.backward(self.cross_enthalpy_loss.backward(output, batch_y))

                # Store data for plotting
                data_x.append(e + i / len(shuffled_x))
                data_y.append(loss)

            # Evaluate on validation set after each epoch
            predicted_val_labels = self.prediction_of_whole_dataset(x_validation)
            tAccuracy = self.get_accuracy(self.prediction_of_whole_dataset(x), y)
            vAccuracy = self.get_accuracy(predicted_val_labels, y_validation)
           # print(f"Epoch: {e + 1}, Validation Accuracy: {accuracy:.3f}")
            train_accuracy.append(tAccuracy)
            val_accuracy.append(vAccuracy)
            print(f"Epoch: {e + 1}, Train accuracy: {tAccuracy:.3f} Validation Accuracy: {vAccuracy:.3f}")
            
            train_loss.append( self.get_full_dataset_loss(x, y))
            val_loss.append(self.get_full_dataset_loss(x_validation, y_validation))
            validation_f1_macro_score.append(f1_score( y_validation, predicted_val_labels, average='macro'))
        
        predicted_labels = self.prediction_of_whole_dataset(x_validation)
        f1score = f1_score( y_validation, predicted_labels, average='macro')
        print(f"Epoch: {e + 1}, Validation F1 macro score: {f1score:.3f}")
        if f1score > max_f1score:
            max_f1score = f1score
            best_model = copy.deepcopy(self)
            
        #plot epoch VS loss
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
    
        plt.plot(train_loss, label='train_loss')
        plt.plot(val_loss, label='val_loss')
        plt.legend()
        plt.show()
        
        
        # #plot epoch VS accuracy
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        
        plt.plot(train_accuracy, label='train_accuracy')
        plt.plot(val_accuracy, label='val_accuracy')
        plt.legend()
        plt.show()
        
        # #plot epoch VS f1 macro score
        plt.xlabel('Epochs')
        plt.ylabel('F1 macro score')
        plt.plot(validation_f1_macro_score, label='train_f1_macro_score')
        plt.legend()
        plt.show()
        #show confusion  matrix

        cm = confusion_matrix(y_validation, predicted_labels)
        plt.figure(figsize=(10,7))
        sn.heatmap(cm, annot=True)
        plt.xlabel('Predicted')
        plt.ylabel('Truth')
        plt.show()
        
        
        return data_x, data_y, best_model, max_f1score


    '''
    def find_the_best_model(self, x_train, x_validation, y_train, y_validation, mini_batch_size, epochs,learning_rate):
        max_f1score = 0
        best_model = None
        for i in range(epochs):
            self.fit_minibatch(x_train, y_train, x_validation, y_validation, epochs, mini_batch_size,learning_rate)
            predicted_labels = self.prediction_of_whole_dataset(x_validation)
            f1score = calculate_f1_score(predicted_labels, y_validation)
            if f1score > max_f1score:
                max_f1score = f1score
                best_model = copy.deepcopy(self)
        return best_model, max_f1score
    
    '''
    
        
            
        
    def set(self, loss, optimizer, accuracy):
        self.loss = loss
        self.optimizer = optimizer
        self.accuracy = accuracy
        

      
class optimizer_adam:
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))
            
    def update_params(self, layer):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)
            
        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.d_weights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.d_biases
        
        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        
        layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.d_weights ** 2
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.d_biases ** 2
        
        weight_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** (self.iterations + 1))
        
        layer.weights += -self.current_learning_rate * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += -self.current_learning_rate * bias_momentums_corrected / (np.sqrt(bias_cache_corrected) + self.epsilon)
        
    def post_update_params(self):
        self.iterations += 1
    
    
    
if __name__ == '__main__':
    training_set, labels, letters = load_preprocess_training_dataset()
    testing_set, test_labels, test_letters = load_preprocess_testing_dataset()
    x_train = training_set
    y_train = labels
    x_test = testing_set
    y_test = test_labels
    
   # print("x_train.shape ", x_train.shape)
   # print("y_train.shape ", y_train.shape)    
    x_train, x_validation , y_train, y_validation = train_test_split(x_train, y_train, test_size=0.15, random_state=42)
    
    epochs= 100
    learning_rate = 0.005
    mini_batch_size = 221
    rate_dropout= 0.3
    
   # model1 = model_FNN()
   # adam = optimizer_adam(learning_rate=learning_rate, decay=5e-7)
    
   # model1.set(loss=loss_categorical_crossentropy(), optimizer= adam, accuracy=None)
    #dense1= dense_layer(784, 1248)
   # dense2= dense_layer(1248, 26)
    
   # model1.add_a_layer(dense1)
   # model1.add_a_layer(relu_activation())
   # model1.add_a_layer(layer_dropout(rate_dropout))
   # model1.add_a_layer(dense2)
  #  model1.add_a_layer(activation_softmax())
    
    #adam.pre_update_params()
   # adam.update_params(dense1)
   # adam.update_params(dense2)
   # adam.post_update_params()
    '''
    model2 = model_FNN()
   # adam = optimizer_adam(learning_rate=learning_rate, decay=5e-7)
    
    model2.set(loss=loss_categorical_crossentropy(), optimizer= None, accuracy=None)
    dense1= dense_layer(784, 100)
    dense2= dense_layer(100, 26)
    
    model2.add_a_layer(dense1)
    model2.add_a_layer(relu_activation())
    model2.add_a_layer(layer_dropout(rate_dropout))
    model2.add_a_layer(dense2)
    model2.add_a_layer(activation_softmax())
    '''
    
    
    model3 = model_FNN()
    adam = optimizer_adam(learning_rate=learning_rate, decay=5e-7)
    
    model3.set(loss=loss_categorical_crossentropy(), optimizer= adam, accuracy=None)
    dense1= dense_layer(784, 100)
    dense2= dense_layer(100, 26)
    
    model3.add_a_layer(dense1)
    #model3.add_a_layer(relu_activation())
    #model3.add_a_layer(layer_dropout(rate_dropout))
    model3.add_a_layer(dense2)
    #model3.add_a_layer(activation_softmax())
    
  
   
    #use optimizer adam
    
    #model.set(loss=loss_categorical_crossentropy(), optimizer=optimizer_adam, accuracy=None)
    
    
   # model1.fit_minibatch(x_train, y_train, x_validation, y_validation, epochs, mini_batch_size, learning_rate)
   # predicted_labels = model1.prediction_of_whole_dataset(x_test)
   # accuracy = model1.get_accuracy(predicted_labels, y_test)
    
    data_x, data_y, best_model, max_f1 = model3.fit_minibatch(x_train,  y_train,x_validation, y_validation, mini_batch_size, epochs,learning_rate)
   # with open('model_FNN.pickle', 'wb') as f:
    #    pickle.dump(best_model, f)
    
    #save the model in pickle
    with open('model_FNN3.pickle', 'wb') as f:
        pickle.dump(best_model, f)
    
    model = pickle.load(open('model_FNN3.pickle', 'rb'))
    predicted_labels = model.prediction_of_whole_dataset(x_test)
    accuracy = model.get_accuracy(predicted_labels, y_test)
    print(f"Accuracy on test set: {accuracy:.3f}")
   
    
    #graph plot train and validation loss VS epoch
    
     
     
    
    

    
    
    
