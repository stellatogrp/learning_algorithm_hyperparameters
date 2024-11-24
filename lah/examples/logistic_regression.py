import jax.numpy as jnp
import numpy as np
import cvxpy as cp
import yaml
from lah.launcher import Workspace
from lah.examples.solve_script import gd_setup_script
import os
import hydra
import gzip
import time
import matplotlib.pyplot as plt


def run(run_cfg, model='lah'):
    example = "logistic_regression"
    data_yaml_filename = 'data_setup_copied.yaml'

    # read the yaml file
    with open(data_yaml_filename, "r") as stream:
        try:
            setup_cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            setup_cfg = {}

    # set the seed
    np.random.seed(setup_cfg['seed'])
    # n_orig = setup_cfg['n_orig']
    # m_orig = setup_cfg['m_orig']

    num_points = setup_cfg['num_points']
    # A = np.random.normal(size=(m_orig, n_orig))
    # P = A.T @ A + lambd * np.identity(n_orig)

    gd_step = 0.01

    static_dict = dict(gd_step=gd_step, num_points=num_points)

    # we directly save q now
    static_flag = True
    if model == 'lah':
        algo = 'lah_logisticgd'
    elif model == 'l2ws':
        algo = 'logisticgd'
    else:
        algo = 'lm_logisticgd'
    workspace = Workspace(algo, run_cfg, static_flag, static_dict, example)

    # run the workspace
    workspace.run()


# Function to create binary problems
def create_binary_problems(X, y, N, samples_per_problem=100):
    binary_problems = []
    classes = np.unique(y)
    num_classes = len(classes)
    
    for _ in range(N):
        # Randomly select two classes
        class1, class2 = np.random.choice(classes, size=2, replace=False)
        
        # Select samples for the two classes
        idx_class1 = np.where(y == class1)[0]
        idx_class2 = np.where(y == class2)[0]
        
        # Randomly sample 50 datapoints from each class to form a problem with 100 datapoints
        selected_idx_class1 = np.random.choice(idx_class1, size=samples_per_problem // 2, replace=False)
        selected_idx_class2 = np.random.choice(idx_class2, size=samples_per_problem // 2, replace=False)
        
        selected_idx = np.concatenate([selected_idx_class1, selected_idx_class2])
        
        X_binary = X[selected_idx]
        y_binary = y[selected_idx]
        
        # Relabel classes to 0 and 1
        y_binary = np.where(y_binary == class1, 0, 1)
        
        binary_problems.append((X_binary, y_binary))
    
    return binary_problems


def setup_probs(setup_cfg):
    cfg = setup_cfg
    N_train, N_test = cfg.N_train, cfg.N_test
    N = N_train + N_test
    np.random.seed(setup_cfg['seed'])

    num_points = cfg.num_points

    # get the mnist data
    # Load the MNIST dataset
    # (X_train_full, y_train_full), (X_test_full, y_test_full) = tf.keras.datasets.mnist.load_data()
    x_train, x_test, y_train, y_test = get_mnist(emnist=False)

    # # Flatten the images
    # X_train_full = X_train_full.reshape((X_train_full.shape[0], -1))
    # X_test_full = X_test_full.reshape((X_test_full.shape[0], -1))

    # # Normalize the data
    # X_train_full = X_train_full / 255.0
    # X_test_full = X_test_full / 255.0

    # # the parameter for each problem is the data
    # # Create N binary logistic regression problems each with 100 datapoints
    samples_per_problem = num_points + 20 #120
    binary_problems = create_binary_problems(x_train, y_train, N, samples_per_problem=samples_per_problem)

    # import pdb
    # pdb.set_trace()

    # Example of training logistic regression on the generated binary problems
    train_size = num_points
    z_stars = np.zeros((N, 784 + 1))
    theta_mat = np.zeros((N, train_size * (784 + 1)))
    for i, (X_binary, y_binary) in enumerate(binary_problems):
        # Manually split into train and test sets
        
        X_train, X_test = X_binary[:train_size], X_binary[train_size:]
        y_train, y_test = y_binary[:train_size], y_binary[train_size:]
        
        # Train logistic regression model
        w, b = train_logistic_regression(X_train, y_train)

        # fill z_stars
        z_stars[i, :784] = w
        z_stars[i, -1] = b

        # fill theta_mat
        theta_mat[i, :784 * train_size] = X_train.flatten()
        theta_mat[i, 784 * train_size:] = y_train
        
        # Predict and evaluate
        y_pred = predict(X_test, w, b)
        accuracy = np.mean(y_pred == y_test)
        print(f'Problem {i+1} Accuracy: {accuracy}')


    # save output to output_filename
    output_filename = f"{os.getcwd()}/data_setup"


    # gd_setup_script(c_mat, P, theta_mat, output_filename)

    # save the data
    print("final saving final data...")
    t0 = time.time()
    jnp.savez(
        output_filename,
        # q_mat=jnp.array(c_mat),
        thetas=jnp.array(theta_mat),
        z_stars=jnp.array(z_stars),
    )

    save_time = time.time()
    print(f"finished saving final data... took {save_time-t0}'")

    # save plot of first 5 solutions
    for i in range(5):
        plt.plot(z_stars[i, :])
    plt.savefig("z_stars.pdf")
    plt.clf()


    # save plot of first 5 parameters
    for i in range(5):
        plt.plot(theta_mat[i, :])
    plt.savefig("thetas.pdf")
    plt.clf()


def obj(P, c, z):
    return .5 * z.T @ P @ z  + c @ z

def obj_diff(obj, true_obj):
    return (obj - true_obj)


def solve_many_probs_cvxpy(P, c_mat):
    """
    solves many unconstrained qp problems where each problem has a different b vector
    """
    P_inv = jnp.linalg.inv(P)
    z_stars = -P_inv @ c_mat
    objvals = obj(P, c_mat, z_stars)
    return z_stars, objvals

def get_mnist(emnist=True):
    orig_cwd = hydra.utils.get_original_cwd()
    if emnist:
        images, labels = extract_training_samples('letters')
        images = images[:20000, :, :]
        x_train = np.reshape(images, (images.shape[0], 784)) / 255
        x_test = None

    else:
        # Load MNIST dataset
        # x_train, y_train = load_mnist('mnist_data', kind='train')
        # x_test, y_test = load_mnist('mnist_data', kind='t10k')
        x_train, y_train = load_mnist(f"{orig_cwd}/examples/mnist_data", kind='train')
        x_test, y_test = load_mnist(f"{orig_cwd}/examples/mnist_data", kind='t10k')

        # Normalize pixel values
        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255
    return x_train, x_test, y_train, y_test

def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path, '%s-labels-idx1-ubyte.gz' % kind)
    images_path = os.path.join(path, '%s-images-idx3-ubyte.gz' % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)

    return images, labels


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def compute_loss(y, y_hat):
    m = y.shape[0]
    return -1/m * (np.dot(y, np.log(1e-6 + y_hat)) + np.dot((1 - y), np.log(1 + 1e-6 - y_hat)))


def compute_gradient(X, y, y_hat):
    m = y.shape[0]
    dw = 1/m * np.dot(X.T, (y_hat - y))
    db = 1/m * np.sum(y_hat - y)
    return dw, db


def train_logistic_regression(X, y, learning_rate=0.02, beta1=0.9, beta2=0.999, epochs=2000, epsilon=1e-8):
    m, n = X.shape
    w = np.zeros(n)
    b = 0
    
    # for epoch in range(epochs):
    #     z = np.dot(X, w) + b
    #     y_hat = sigmoid(z)
    #     loss = compute_loss(y, y_hat)
        
    #     dw, db = compute_gradient(X, y, y_hat)
        
    #     w -= learning_rate * dw
    #     b -= learning_rate * db
        
    #     if epoch % 1000 == 0:
    #         print(f'Epoch {epoch}, Loss: {loss}')
    # Adam optimizer variables
    m_w, v_w = np.zeros(n), np.zeros(n)
    m_b, v_b = 0, 0
    t = 0

    for epoch in range(epochs):
        t += 1
        z = np.dot(X, w) + b
        y_hat = sigmoid(z)
        loss = compute_loss(y, y_hat)
        
        dw, db = compute_gradient(X, y, y_hat)
        
        # Update moments
        m_w = beta1 * m_w + (1 - beta1) * dw
        v_w = beta2 * v_w + (1 - beta2) * (dw ** 2)
        
        m_b = beta1 * m_b + (1 - beta1) * db
        v_b = beta2 * v_b + (1 - beta2) * (db ** 2)
        
        # Bias correction
        m_w_hat = m_w / (1 - beta1 ** t)
        v_w_hat = v_w / (1 - beta2 ** t)
        
        m_b_hat = m_b / (1 - beta1 ** t)
        v_b_hat = v_b / (1 - beta2 ** t)
        
        # Update weights and bias
        w -= learning_rate * m_w_hat / (np.sqrt(v_w_hat) + epsilon)
        b -= learning_rate * m_b_hat / (np.sqrt(v_b_hat) + epsilon)
        
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss}')
    
    return w, b

def predict(X, w, b):
    z = np.dot(X, w) + b
    y_hat = sigmoid(z)
    return np.where(y_hat >= 0.5, 1, 0)
