import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
import numpy as np
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)

# TODO: finish this class


class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(6, 10)  # 6 input features, 10 hidden units
        self.fc2 = nn.Linear(10, 10)  # 10 hidden units, 10 hidden units
        self.fc3 = nn.Linear(10, 10)  # 10 hidden units, 10 hidden units
        self.fc4 = nn.Linear(10, 10)  # 10 hidden units, 10 hidden units
        self.fc5 = nn.Linear(10, 3)  # 10 hidden units, 3 output units

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # ReLU activation for the first layer
        x = torch.relu(self.fc2(x))  # ReLU activation for the second layer
        x = torch.relu(self.fc3(x))  # ReLU activation for the third layer
        x = torch.relu(self.fc4(x))  # ReLU activation for the fourth layer
        x = self.fc5(x)  # Final output layer, no activation
        return x


class Regression:
    def __init__(self):
        # ANN attributes
        self.model = SimpleNN()
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        self.train_loss_history = []
        self.prediction_errors = []

        # RLS attributes
        self.rls_weights = None
        self.train_loss_history_rls = []

        # ELM attributes
        self.elm_input_weights = None
        self.elm_output_weights = None
        self.train_loss_history_elm = []

        self.current_model = None

    def load_data(self, filepath):
        data = np.load(filepath)
        np.random.shuffle(data.T)  # Shuffle each column randomly, before splitting
        train_size = int(0.9 * len(data.T))
        val_size = len(data.T) - train_size
        self.train_inputs = torch.tensor(data[:6, :train_size]).t()
        self.train_targets = torch.tensor(data[6:, :train_size]).t()
        self.val_inputs = torch.tensor(data[:6, train_size:]).t()
        self.val_targets = torch.tensor(data[6:, train_size:]).t()

    def train_ann(self, num_epochs=200000):
        self.current_model = "ann"
        for epoch in range(num_epochs):
            self.model.train()
            self.optimizer.zero_grad()
            outputs = self.model(self.train_inputs)
            loss = self.criterion(outputs, self.train_targets)
            loss.backward()
            self.optimizer.step()
            self.train_loss_history.append(loss.item())

            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(self.val_inputs)
                prediction_error = (
                    100
                    * torch.abs(val_outputs - self.val_targets).mean().item()
                    / torch.abs(self.val_targets.mean())
                )
                self.prediction_errors.append(prediction_error)

            if (epoch + 1) % 100 == 0:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.14f}, Prediction Relative Percent Error: {prediction_error:.14f}"
                )

    def save_ann_model(self, filepath):
        torch.save(self.model.state_dict(), filepath)

    def plot_ann_training_loss(self, filepath):
        plt.figure()
        plt.semilogy(self.train_loss_history, color="blue")
        plt.xlabel(r"Training Epoch [-]")
        plt.ylabel(r"Loss [-]")
        plt.grid(True, which="both", linestyle="--")
        plt.savefig(filepath, format="pdf")
        plt.show()

    def plot_ann_prediction_errors(self, filepath):
        plt.figure()
        plt.plot(self.prediction_errors, color="red")
        plt.xlabel(r"Training Epoch [-]")
        plt.ylabel(r"Prediction Relative Percent Error [-]")
        plt.grid(True, which="both", linestyle="--")
        plt.savefig(filepath, format="pdf")
        plt.show()

    def rls_regression(self, lambda_, delta, num_epochs=100):
        self.current_model = "rls"
        n_features = self.train_inputs.shape[1]
        n_outputs = self.train_targets.shape[1]

        # Initialize weights and covariance matrix
        self.rls_weights = np.zeros((n_features, n_outputs))
        P = np.eye(n_features) / delta

        for epoch in range(num_epochs):
            for i in range(self.train_inputs.shape[0]):
                x = self.train_inputs[i].numpy().reshape(-1, 1)
                y = self.train_targets[i].numpy().reshape(-1, 1)

                # Compute gain vector
                Px = np.dot(P, x)
                g = Px / (lambda_ + np.dot(x.T, Px))

                # Update weights
                y_hat = np.dot(self.rls_weights.T, x)
                self.rls_weights += np.dot(g, (y - y_hat).T)

                # Update covariance matrix
                P = (P - np.dot(g, x.T.dot(P))) / lambda_

            # Calculate training loss
            y_pred = np.dot(self.train_inputs.numpy(), self.rls_weights)
            loss = np.mean((self.train_targets.numpy() - y_pred) ** 2)
            self.train_loss_history_rls.append(loss)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss:.14f}")

    def plot_rls_training_loss(self, filepath):
        plt.figure()
        plt.plot(self.train_loss_history_rls, color="blue")
        plt.xlabel(r"Training Epoch [-]")
        plt.ylabel(r"Loss [-]")
        plt.grid(True, which="both", linestyle="--")
        plt.savefig(filepath, format="pdf")
        plt.show()

    def train_elm(self, hidden_units=10):
        self.current_model = "elm"
        n_samples = self.train_inputs.shape[0]
        n_features = self.train_inputs.shape[1]
        n_outputs = self.train_targets.shape[1]

        # Initialize input weights randomly and calculate hidden layer output
        self.elm_input_weights = np.random.randn(n_features, hidden_units)
        H = np.tanh(np.dot(self.train_inputs.numpy(), self.elm_input_weights))

        # Calculate output weights using Moore-Penrose pseudoinverse
        H_pinv = np.linalg.pinv(H)
        self.elm_output_weights = np.dot(H_pinv, self.train_targets.numpy())

        # Calculate training loss
        y_pred = np.dot(H, self.elm_output_weights)
        loss = np.mean((self.train_targets.numpy() - y_pred) ** 2)
        self.train_loss_history_elm.append(loss)

        print(f"ELM Training Loss: {loss:.14f}")

    def plot_elm_training_loss(self, filepath):
        plt.figure()
        plt.plot(self.train_loss_history_elm, color="blue")
        plt.xlabel(r"Training Iteration [-]")
        plt.ylabel(r"Loss [-]")
        plt.grid(True, which="both", linestyle="--")
        plt.savefig(filepath, format="pdf")
        plt.show()

    def compute_jacobian(self, input_data):
        """
        Compute the Jacobian matrix of the current model with respect to the input data.

        Parameters:
            input_data (np.ndarray): The input data for which to compute the Jacobian.

        Returns:
            np.ndarray: The computed Jacobian matrix.
        """
        if self.current_model == "ann":
            self.model.eval()
            input_tensor = torch.tensor(input_data, requires_grad=True)
            output = self.model(input_tensor)
            jacobian = torch.zeros(output.size(0), input_tensor.size(0))
            for i in range(output.size(0)):
                gradients = grad(output[i], input_tensor, create_graph=True)[0]
                jacobian[i, :] = gradients
            return jacobian.detach().numpy()

        elif self.current_model == "rls":

            def rls_forward(x):
                return np.dot(x, self.rls_weights)

            input_tensor = torch.tensor(
                input_data, requires_grad=True, dtype=torch.float64
            )
            rls_weights_tensor = torch.tensor(self.rls_weights, dtype=torch.float64)
            output = torch.matmul(input_tensor, rls_weights_tensor)
            jacobian = torch.zeros(output.size(0), input_tensor.size(0))
            for i in range(output.size(0)):
                gradients = grad(output[i], input_tensor, create_graph=True)[0]
                jacobian[i, :] = gradients
            return jacobian.detach().numpy()

        elif self.current_model == "elm":

            def elm_forward(x):
                H = np.tanh(np.dot(x, self.elm_input_weights))
                return np.dot(H, self.elm_output_weights)

            input_tensor = torch.tensor(
                input_data, requires_grad=True, dtype=torch.float64
            )
            elm_input_weights_tensor = torch.tensor(
                self.elm_input_weights, dtype=torch.float64
            )
            elm_output_weights_tensor = torch.tensor(
                self.elm_output_weights, dtype=torch.float64
            )
            H = torch.tanh(torch.matmul(input_tensor, elm_input_weights_tensor))
            output = torch.matmul(H, elm_output_weights_tensor)
            jacobian = torch.zeros(output.size(0), input_tensor.size(0))
            for i in range(output.size(0)):
                gradients = grad(output[i], input_tensor, create_graph=True)[0]
                jacobian[i, :] = gradients
            return jacobian.detach().numpy()

        else:
            raise ValueError("No model has been trained yet.")
