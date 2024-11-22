import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
import os
from functools import partial
from matplotlib.patches import Circle

result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

# Define a simple MLP class
class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr, activation='tanh'):
        np.random.seed(0)
        self.lr = lr # learning rate
        self.activation_fn = activation # activation function

        self.w_1 = np.random.uniform(-0.5, 0.5, (input_dim, hidden_dim)) 
        self.w_2 = np.random.uniform(-0.5, 0.5, (hidden_dim, output_dim)) 

        
        self.b_1 = np.random.randn(1, hidden_dim) * 0.01
        self.b_2 = np.random.randn(1, output_dim) * 0.01

    def forward(self, X):
        def act(v, fn=self.activation_fn):
            if fn == 'tanh':
                return np.tanh(v)
            elif fn == 'relu':
                return np.maximum(0, v)
            elif fn == 'sigmoid':
                return 1 / (1 + np.exp(-v))
        
        self.layer1 = np.dot(X, self.w_1) + self.b_1
        self.activated1 = act(self.layer1)
        self.layer2 = np.dot(self.activated1, self.w_2) + self.b_2
        self.activated2 = act(self.layer2, fn='tanh')
        return self.activated2

    def backward(self, X, y):
        def act(v, fn=self.activation_fn):
            if fn == 'tanh':
                return 1 - np.tanh(v)**2
            elif fn == 'relu':
                return (v > 0).astype(float)
            elif fn == 'sigmoid':
                s = 1 / (1 + np.exp(-v))
                return s * (1 - s)
        m = X.shape[0]
        self.dz2 = self.activated2 - y
        self.dw2 = np.dot(self.activated1.T, self.dz2) / m
        self.db2 = np.sum(self.dz2, axis=0, keepdims=True) / m

        self.dz1 = np.dot(self.dz2, self.w_2.T) * act(self.layer1)
        self.dw1 = np.dot(X.T, self.dz1) / m
        self.db1 = np.sum(self.dz1, axis=0, keepdims=True) / m

        max_grad = 1.0
        self.dw1 = np.clip(self.dw1, -max_grad, max_grad)
        self.dw2 = np.clip(self.dw2, -max_grad, max_grad)

        # Gradient Descent
        self.w_2 -= self.lr * self.dw2
        self.b_2 -= self.lr * self.db2
        self.w_1 -= self.lr * self.dw1
        self.b_1 -= self.lr * self.db1

def generate_data(n_samples=100):
    np.random.seed(0)
    # Generate input
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1).astype(int) * 2 - 1  # Circular boundary
    y = y.reshape(-1, 1)
    return X, y

# Visualization update function
def update(frame, mlp, ax_input, ax_hidden, ax_gradient, X, y):
    ax_hidden.clear()
    ax_input.clear()
    ax_gradient.clear()

    # perform training steps by calling forward and backward function
    for _ in range(10):
        # Perform a training step
        mlp.forward(X)
        mlp.backward(X, y)
        
    # Hidden Layer Features
    hidden_features = mlp.activated1
    ax_hidden.scatter(hidden_features[:, 0], hidden_features[:, 1], hidden_features[:, 2],
                      c=y.ravel(), cmap='bwr', alpha=0.7)
    ax_hidden.set_title(f"Hidden Space at Step {frame*10}")

    # Dynamic Yellow Decision Boundary Plane
    hidden_x = np.linspace(hidden_features[:, 0].min(), hidden_features[:, 0].max(), 20)
    hidden_y = np.linspace(hidden_features[:, 1].min(), hidden_features[:, 1].max(), 20)
    hidden_xx, hidden_yy = np.meshgrid(hidden_x, hidden_y)
    hidden_grid = np.c_[hidden_xx.ravel(), hidden_yy.ravel(), np.zeros_like(hidden_xx.ravel())]

    w2 = mlp.w_2
    b2 = mlp.b_2
    hidden_zz = -(hidden_grid[:, 0] * w2[0, 0] + hidden_grid[:, 1] * w2[1, 0] + b2[0, 0]) / w2[2, 0]
    hidden_zz = hidden_zz.reshape(hidden_xx.shape)

    ax_hidden.plot_surface(hidden_xx, hidden_yy, hidden_zz, alpha=0.3, color='orange', edgecolor='none')

    # Blue Distortion Plane
    input_x = np.linspace(X[:, 0].min(), X[:, 0].max(), 20)
    input_y = np.linspace(X[:, 1].min(), X[:, 1].max(), 20)
    input_xx, input_yy = np.meshgrid(input_x, input_y)
    input_grid = np.c_[input_xx.ravel(), input_yy.ravel()]  # Shape (400, 2)

    mlp.forward(input_grid)  # Update mlp.activated1
    hidden_transformed = mlp.activated1.reshape((20, 20, 3))

    ax_hidden.plot_surface(hidden_transformed[:, :, 0],  # Hidden Unit 1
                           hidden_transformed[:, :, 1],  # Hidden Unit 2
                           hidden_transformed[:, :, 2],  # Hidden Unit 3
                           alpha=0.3, color='blue', edgecolor='none')
    # Decision Boundary in Input Space
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    predictions = mlp.forward(grid_points)
    predicted_labels = (predictions > 0.5).astype(int).reshape(xx.shape)
    ax_input.contourf(xx, yy, predicted_labels, alpha=0.6, cmap='bwr')
    ax_input.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='bwr', edgecolor='k')
    ax_input.set_title(f"Input Space at Step {frame*10}")

    # Gradients Visualization
    ax_gradient.clear()

    # Node positions
    input_nodes = [(0, i+.5) for i in range(2)]  # Input nodes: x1, x2
    hidden_nodes = [(1, i) for i in range(3)]  # Hidden layer nodes: h1, h2, h3
    output_node = [(2, 1)]  # Output node: y

    # Labels for nodes
    input_labels = ["x1", "x2"]
    hidden_labels = ["h1", "h2", "h3"]
    output_label = ["y"]


    # Plot connections from inputs to hidden layer with gradient thickness
    grad_magnitude_input_hidden = np.abs(mlp.dw1)  # Gradients from input to hidden layer
    for i, (x_in, y_in) in enumerate(input_nodes):  # Input nodes
        for j, (x_h, y_h) in enumerate(hidden_nodes):  # Hidden layer nodes
            line_width = min(5, grad_magnitude_input_hidden[i, j] * 50)  # Scale line width
            ax_gradient.plot([x_in, x_h], [y_in, y_h], "#FC6b3a", alpha=0.9, linewidth=line_width)

    # Plot connections from hidden layer to output node with gradient thickness
    grad_magnitude_hidden_output = np.abs(mlp.dw2)  # Gradients from hidden to output layer
    for i, (x_h, y_h) in enumerate(hidden_nodes):  # Hidden layer nodes
        for j, (x_out, y_out) in enumerate(output_node):  # Output node
            line_width = min(5, grad_magnitude_hidden_output[i, j] * 50)  # Scale line width
            ax_gradient.plot([x_h, x_out], [y_h, y_out], "#FC6B3A", alpha=0.9, linewidth=line_width)
            
    # Draw input nodes
    for idx, (x, y) in enumerate(input_nodes):
        ax_gradient.scatter(x, y, s=800, c="#ED5443", alpha=1)
        ax_gradient.text(x, y, input_labels[idx], ha="center", va="center", color="white", fontsize=12)

    # Draw hidden nodes
    for idx, (x, y) in enumerate(hidden_nodes):
        ax_gradient.scatter(x, y, s=800, c="#ED5443", alpha=1)
        ax_gradient.text(x, y, hidden_labels[idx], ha="center", va="center", color="white", fontsize=12)

    # Draw output node
    for idx, (x, y) in enumerate(output_node):
        ax_gradient.scatter(x, y, s=800, c="#ED5443", alpha=1)
        ax_gradient.text(x, y, output_label[idx], ha="center", va="center", color="white", fontsize=12)

    # Adjust plot limits and labels
    ax_gradient.set_xlim(-0.5, 2.5)  # Left to right for input, hidden, output
    ax_gradient.set_ylim(-0.5, max(len(input_nodes), len(hidden_nodes), len(output_node)) - 0.5)
    ax_gradient.set_title(f"Gradients at Step {frame*10}")
    ax_gradient.axis("off")  # Turn off axes



def visualize(activation, lr, step_num):
    X, y = generate_data()
    mlp = MLP(input_dim=2, hidden_dim=3, output_dim=1, lr=lr, activation=activation)

    # Set up visualization
    matplotlib.use('agg')
    fig = plt.figure(figsize=(21, 7))
    ax_hidden = fig.add_subplot(131, projection='3d')
    ax_input = fig.add_subplot(132)
    ax_gradient = fig.add_subplot(133)

    # Create animation
    ani = FuncAnimation(fig, partial(update, mlp=mlp, ax_input=ax_input, ax_hidden=ax_hidden, ax_gradient=ax_gradient, X=X, y=y), frames=step_num//10, repeat=False)

    # Save the animation as a GIF
    ani.save(os.path.join(result_dir, "visualize.gif"), writer='pillow', fps=10)
    plt.close()

if __name__ == "__main__":
    activation = "tanh"
    lr = 0.2
    step_num = 2000
    visualize(activation, lr, step_num)