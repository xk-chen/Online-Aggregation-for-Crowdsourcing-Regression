import numpy as np
import matplotlib.pyplot as plt

class Base_with_weights:
    
    def __init__(self, base_learners_collection, variance_dict):
        self.base_learners_collection = base_learners_collection
        self.num_base_learners = np.sum(np.array(base_learners_collection))
        self.weight = 1.0 / self.num_base_learners * np.ones(self.num_base_learners)
        self.cumulative_loss = 0
        self.MAE_collection = []
        self.weight_collection = []
        self.step_counter = 0
        self.variance_dict = variance_dict
        
    def training(self, X, learning_rate):
        self.X = X
        self.learning_rate = learning_rate
        num_observations = len(self.X)
        for t in range(num_observations):
            variance_collection = []
            for i in range(len(self.base_learners_collection)):
                for j in range(self.base_learners_collection[i]):
                    variance_collection.append(np.random.choice(self.variance_dict[str(i)], 1, replace=True)[0])
            x_annotation = np.random.normal(self.X[t], variance_collection, len(variance_collection))
            predict = self.predict(x_annotation)
            self.cumulative_loss += np.abs(predict - self.X[t])
            gradient = (predict - self.X[t]) * x_annotation
            numerator = self.weight * np.exp(-self.learning_rate * gradient)
            self.weight = numerator * 1.0 / np.sum(numerator)
            self.weight_collection.append(self.weight)
            self.step_counter += 1
            self.MAE_collection.append(self._get_MAE())
            
    def predict(self, x):
        return np.dot(self.weight, x)
    
    def _get_MAE(self): 
        return self.cumulative_loss * 1.0 / self.step_counter
    
    def plot_MAE_curve(self, x_label = "Number of samples", y_label = "MAE"):
        plt.plot(self.MAE_collection)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.show()
    
    def plot_weights_curve(self, x_label = "Number of samples", y_label = "Weights"):
        weight_collection = np.array(self.weight_collection)
        for i in range(weight_collection.shape[-1]):
            plt.plot(weight_collection[:, i], label = "Worker{}".format(i + 1))
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend()
        plt.show()


class Base_with_average:

    def __init__(self, base_learners_collection, variance_dict):
        self.base_learners_collection = base_learners_collection
        self.num_base_learners = np.sum(np.array(base_learners_collection))
        self.weight = 1.0 / self.num_base_learners * np.ones(self.num_base_learners)
        self.cumulative_loss = 0
        self.MAE_collection = []
        self.step_counter = 0
        self.variance_dict = variance_dict
    
    def training(self, X, learning_rate):
        self.X = X
        self.learning_rate = learning_rate
        num_observations = len(self.X)
        for t in range(num_observations):
            variance_collection = []
            for i in range(len(self.base_learners_collection)):
                for j in range(self.base_learners_collection[i]):
                    variance_collection.append(np.random.choice(self.variance_dict[str(i)], 1, replace=True)[0])
            x_annotation = np.random.normal(self.X[t], variance_collection, len(variance_collection))
            predict = self.predict(x_annotation)
            self.cumulative_loss += np.abs(predict - self.X[t])
            gradient = (predict - self.X[t]) * x_annotation
            self.step_counter += 1
            self.MAE_collection.append(self._get_MAE())
            
    def predict(self, x):
        return np.dot(self.weight, x)
    
    def _get_MAE(self): 
        return self.cumulative_loss * 1.0 / self.step_counter
    
    def plot_MAE_curve(self, x_label = "Number of samples", y_label = "MAE"):
        plt.plot(self.MAE_collection)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.show()


class Base_with_sampling:
 
    def __init__(self, base_learners_collection, variance_dict):
        self.base_learners_collection = base_learners_collection
        self.num_base_learners = np.sum(np.array(base_learners_collection))
        self.weight = 1.0 / self.num_base_learners * np.ones(self.num_base_learners)
        self.cumulative_loss = 0
        self.MAE_collection = []
        self.weight_collection = []
        self.step_counter = 0
        self.variance_dict = variance_dict
        
    def sampling(self):
        outcome = np.argmax(np.random.multinomial(1, self.weight, size = 1))
        return outcome
    
    def training(self, X, learning_rate):
        self.X = X
        self.learning_rate = learning_rate
        num_observations = len(self.X)
        for t in range(num_observations):
            variance_collection = []
            for i in range(len(self.base_learners_collection)):
                for j in range(self.base_learners_collection[i]):
                    variance_collection.append(np.random.choice(self.variance_dict[str(i)], 1, replace=True)[0])
            x_annotation = np.random.normal(self.X[t], variance_collection, len(variance_collection))
            predict = x_annotation[self.sampling()]
            self.cumulative_loss += np.abs(predict - self.X[t])
            gradient = (predict - self.X[t]) * x_annotation
            numerator = self.weight * np.exp(-self.learning_rate * gradient)
            self.weight = numerator * 1.0 / np.sum(numerator)
            self.weight_collection.append(self.weight)
            self.step_counter += 1
            self.MAE_collection.append(self._get_MAE())
            
    def predict(self, x):
        return x[self.sampling()]
    
    def _get_MAE(self): 
        return self.cumulative_loss * 1.0 / self.step_counter
    
    def plot_MAE_curve(self, x_label = "Number of samples", y_label = "MAE"):
        plt.plot(self.MAE_collection)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.show()
    
    def plot_weights_curve(self, x_label = "Number of samples", y_label = "Weights"):
        weight_collection = np.array(self.weight_collection)
        for i in range(weight_collection.shape[-1]):
            plt.plot(weight_collection[:, i], label = "Worker {}".format(i + 1))
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.show()


class Base_with_stepsize:

    def __init__(self):
        self.cumulative_loss = 0
        self.MAE_collection = []
        self.weight_collection = []
        self.step_counter = 0
    
    def training(self, X, y, learning_rate, strategy):
        self.X = X
        self.y = y
        self.num_base_learners = self.X.shape[-1]
        num_observations = len(self.X)
        self.weight = 1.0 / self.num_base_learners * np.ones(self.num_base_learners)
        self.learning_rate = learning_rate

        for t in range(num_observations):
            predict = self.predict(self.X[t])
            self.cumulative_loss += np.abs(predict - self.y[t])
            gradient = (predict - self.y[t]) * self.X[t]
            self.learning_rate_adjusting(strategy)
            numerator = self.weight * np.exp(-self.learning_rate * gradient)
            self.weight = numerator * 1.0 / np.sum(numerator)
            self.weight_collection.append(self.weight)
            self.step_counter += 1
            self.MAE_collection.append(self._get_MAE())
            
    def is_two_power(self, x):
        if (x & (x-1)) == 0:
            return True
        else:
            return False
    
    def learning_rate_adjusting(self, strategy):
        if strategy == "constant":
            pass
        elif strategy == "oracle":
            pass
        elif strategy == "doubling":
            if self.is_two_power(self.step_counter + 1):
                self.learning_rate = self.learning_rate / np.sqrt(2)
        else:
            raise NameError, "No such an option."
        
    def predict(self, x):
        return np.dot(self.weight, x)
    
    def _get_MAE(self): 
        return self.cumulative_loss * 1.0 / self.step_counter
    
    def plot_MAE_curve(self, x_label = "Number of samples", y_label = "MAE"):
        plt.plot(self.MAE_collection)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.show()
    
    def plot_weights_curve(self, x_label =  "Number of samples", y_label = "MAE"):
        weight_collection = np.array(self.weight_collection)
        for i in range(weight_collection.shape[-1]):
            plt.plot(weight_collection[:, i], label = "Worker {}".format(i + 1))
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend()

def shuffle(X, labels):

    randomize = np.arange(len(labels))
    np.random.shuffle(randomize)
    X_shuffle = X[randomize]
    labels_shuffle = labels[randomize]
    return X_shuffle, labels_shuffle
