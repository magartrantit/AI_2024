import numpy as np

class Cost:
    @staticmethod
    def function(network_output, expected):
        raise NotImplementedError("Cost function not implemented.")

    @staticmethod
    def derivative(weighted_sums, network_output, expected):
        raise NotImplementedError("Cost derivative not implemented.")

class CrossEntropyCost(Cost):
    @staticmethod
    def function(network_output, expected):
        return -np.sum(np.nan_to_num(expected * np.log(network_output) + (1 - expected) * np.log(1 - network_output)))

    @staticmethod
    def derivative(weighted_sums, network_output, expected):
        return network_output - expected

class Activation:
    @staticmethod
    def function(weighted_sums):
        raise NotImplementedError("Activation function not implemented.")

    @staticmethod
    def derivative(weighted_sums):
        raise NotImplementedError("Activation derivative not implemented.")

class RELu(Activation):
    @staticmethod
    def function(weighted_sums):
        return np.maximum(0, weighted_sums)

    @staticmethod
    def derivative(weighted_sums):
        return (weighted_sums > 0).astype(float)

class Network:
    def __init__(self, input_size, hidden_sizes, output_size, cost, activation):
        self.sizes = [input_size] + hidden_sizes + [output_size]
        self.biases = [np.random.randn(y) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x) / np.sqrt(x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        self.cost = cost
        self.activation = activation
        
    def activation_function(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def feedforward(self, x):

        for b, w in zip(self.biases, self.weights):
            x = self.activation_function(np.dot(w, x) + b)
        return x

    def forward_propagation(self, data_input):
        outputs, inputs = [data_input], []
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            z = np.dot(w, outputs[-1]) + b
            inputs.append(z)
            outputs.append(self.activation.function(z))
        z = np.dot(self.weights[-1], outputs[-1]) + self.biases[-1]
        inputs.append(z)
        outputs.append(self.softmax(z))
        return outputs, inputs

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z))
        return exp_z / np.sum(exp_z)

    def backpropagation(self, data_input, data_output):
        outputs, inputs = self.forward_propagation(data_input)
        delta = self.cost.derivative(inputs[-1], outputs[-1], data_output)
        weight_gradients = [np.outer(delta, outputs[-2])]
        bias_gradients = [delta]
        for l in range(2, len(self.sizes)):
            z = inputs[-l]
            delta = np.dot(self.weights[-l + 1].T, delta) * self.activation.derivative(z)
            weight_gradients.insert(0, np.outer(delta, outputs[-l - 1]))
            bias_gradients.insert(0, delta)
        return weight_gradients, bias_gradients

    def update_parameters(self, mini_batch, lr, reg_param, n):
        weight_changes = [np.zeros_like(w) for w in self.weights]
        bias_changes = [np.zeros_like(b) for b in self.biases]
        for data_input, data_output in mini_batch:
            weight_grad, bias_grad = self.backpropagation(data_input, data_output)
            for wc, wg in zip(weight_changes, weight_grad):
                wc += wg
            for bc, bg in zip(bias_changes, bias_grad):
                bc += bg
        self.weights = [(1 - lr * reg_param / n) * w - (lr / len(mini_batch)) * dw for w, dw in zip(self.weights, weight_changes)]
        self.biases = [b - (lr / len(mini_batch)) * db for b, db in zip(self.biases, bias_changes)]

    def train(self, training_data, epochs: int, batch_size, lr, reg_param=0.0, validation_data=None):
        # Transform training data to one-hot encoded format
        training_data = [(x, np.eye(self.sizes[-1])[y]) for x, y in training_data]
        
        for epoch in range(epochs):
            np.random.shuffle(training_data)
            
            mini_batches = [training_data[k:k + batch_size] for k in range(0, len(training_data), batch_size)]
            
            for mini_batch in mini_batches:
                self.update_parameters(mini_batch, lr, reg_param, len(training_data))
            
            if validation_data:
                training_accuracy = self.evaluate([(x, np.argmax(y)) for x, y in training_data])
                validation_accuracy = self.evaluate(validation_data)
                print(f"Epoch {epoch + 1}: Validation Accuracy = {validation_accuracy} / {len(validation_data)}, Training Accuracy = {training_accuracy} / {len(training_data)}")

    def evaluate(self, data): 
        results = []
        for x, y in data:
            predicted_class = np.argmax(self.forward_propagation(x)[0][-1])
            true_class = np.argmax(y) if isinstance(y, np.ndarray) else y
            results.append(predicted_class == true_class)
        return sum(results)
