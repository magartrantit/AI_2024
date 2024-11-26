import numpy as np
from abc import ABC, abstractmethod
from typing import Type, List, Tuple

class Cost(ABC):
    @staticmethod
    @abstractmethod
    def function(network_output, expected):
        pass

    @staticmethod
    @abstractmethod
    def derivative(weighted_sums, network_output, expected):
        pass

class CrossEntropyCost(Cost):
    @staticmethod
    def function(network_output, expected):
        return -np.sum(expected * np.log(network_output + 1e-8))

    @staticmethod
    def derivative(weighted_sums, network_output, expected):
        return network_output - expected

class Activation(ABC):
    @staticmethod
    @abstractmethod
    def function(weighted_sums):
        pass

    @staticmethod
    @abstractmethod
    def derivative(weighted_sums):
        pass

class RELu(Activation):
    @staticmethod
    def function(weighted_sums):
        return np.maximum(0, weighted_sums)

    @staticmethod
    def derivative(weighted_sums):
        return (weighted_sums > 0).astype(float)

class LeakyRELU(Activation):
    @staticmethod
    def function(weighted_sums):
        return np.where(weighted_sums > 0, weighted_sums, 0.01 * weighted_sums)

    @staticmethod
    def derivative(weighted_sums):
        return np.where(weighted_sums > 0, 1, 0.01)

class Network:
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int, cost: Type[Cost] = CrossEntropyCost, activation: Type[Activation] = LeakyRELU):
        self.sizes = [input_size] + hidden_sizes + [output_size]
        self.biases = [np.random.randn(y) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x) * np.sqrt(2 / x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        self.cost = cost
        self.activation = activation

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
        for data_input, data_output in mini_batch:
            weight_grad, bias_grad = self.backpropagation(data_input, data_output)
            for i in range(len(self.weights)):
                self.weights[i] -= lr * (weight_grad[i] + reg_param * self.weights[i])
                self.biases[i] -= lr * bias_grad[i]

    def train(self, training_data: List[Tuple[np.ndarray, int]], epochs: int, batch_size: int, lr: float, reg_param: float = 0.0, validation_data=None):
        training_data = [(x, np.eye(self.sizes[-1])[y]) for x, y in training_data]
        for epoch in range(epochs):
            np.random.shuffle(training_data)
            mini_batches = [training_data[k:k + batch_size] for k in range(0, len(training_data), batch_size)]
            for mini_batch in mini_batches:
                self.update_parameters(mini_batch, lr, reg_param, len(training_data))
            if validation_data:
                accuracy = self.evaluate(validation_data)
                print(f"Epoch {epoch + 1}: {accuracy} / {len(validation_data)}")

    def evaluate(self, data: List[Tuple[np.ndarray, int]]):
        results = [(np.argmax(self.forward_propagation(x)[0][-1]), y) for x, y in data]
        return sum(int(x == y) for x, y in results)
