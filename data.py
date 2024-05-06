# Importing necessary libraries
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import torch
import matplotlib.pyplot as plt
import numpy as np

# Set data parameters
N_SAMPLES = 1000 
NUM_CLASSES = 5
NUM_FEATURES = 2
TEST_SIZE = 0.2
RANDOM_SEED = 7

# Create the data
X, y = make_blobs(n_samples = N_SAMPLES, n_features = NUM_FEATURES, centers = NUM_CLASSES, cluster_std = 1.5, random_state = RANDOM_SEED)

# Data into Tensors
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float) 

# Split data into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = TEST_SIZE)  

# Visualize the data in a scatter graph
def visualize_scatter(X, y):
    plt.scatter(x = X[:, 0], y = X[:, 1], c = y, cmap = plt.cm.RdYlBu)
    plt.show()

visualize_scatter(X, y)

# Set the device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Data to device
X_train = X_train.to(device)
X_test = X_test.to(device)
y_train = y_train.to(device)
y_test = y_test.to(device)

def plot_decision_boundary(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor):
    # Put everything to CPU (works better with NumPy + Matplotlib)
    model.to("cpu")
    X, y = X.to("cpu"), y.to("cpu")

    # Setup prediction boundaries and grid
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101), np.linspace(y_min, y_max, 101))

    # Make features
    X_to_pred_on = torch.from_numpy(np.column_stack((xx.ravel(), yy.ravel()))).float()

    # Make predictions
    model.eval()
    with torch.inference_mode():
        y_logits = model(X_to_pred_on)

    # Test for multi-class or binary and adjust logits to prediction labels
    if len(torch.unique(y)) > 2:
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)  # mutli-class
    else:
        y_pred = torch.round(torch.sigmoid(y_logits))  # binary

    # Reshape preds and plot
    y_pred = y_pred.reshape(xx.shape).detach().numpy()
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

# Visualizer function
def visualize_data(model, X_train, X_test, y_train, y_test):
    plt.figure(figsize = (12,6))
    plt.subplot(1,2,1)
    plt.title("Train")
    plot_decision_boundary(model, X_train, y_train)
    plt.subplot(1,2,2)
    plt.title("Test")
    plot_decision_boundary(model, X_test, y_test)
    plt.show()