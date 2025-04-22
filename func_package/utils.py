import numpy as np

def initialize_parameters(layer_dims, initialization="random"):
    """
    Initialize weights and biases for a multilayer neural network.

    Arguments:
    layer_dims -- list of integers, the dimensions of each layer (including input and output)
    initialization -- string, one of:
        "random"  : small random values
        "zeros"   : all zeros

    Returns:
    parameters -- dict containing:
        W1, b1, W2, b2, ..., WL, bL
    """
    import numpy as np

    np.random.seed(1)
    parameters = {}
    L = len(layer_dims)  # number of layers in the network

    for l in range(1, L):
        if initialization == "zeros":
            W = np.zeros((layer_dims[l], layer_dims[l-1]))
        else:  # "random"
            W = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01

        b = np.zeros((layer_dims[l], 1))

        parameters[f"W{l}"] = W
        parameters[f"b{l}"] = b

    return parameters

def split_parameters(parameters):
    """
    Given a dict with keys "W1", "b1", "W2", "b2", … 
    returns (weights_dict, biases_dict).
    """
    weights = {k: v for k, v in parameters.items() if k.startswith("W")}
    biases  = {k: v for k, v in parameters.items() if k.startswith("b")}
    return weights, biases

def get_minibatches(X, Y, batch_size, shuffle=True):
    """
    Yield successive (X_batch, Y_batch) pairs from the dataset.

    Arguments:
    X           -- input data, shape (n_x, m)
    Y           -- labels, shape (n_y, m)
    batch_size  -- size of each mini‑batch
    shuffle     -- whether to shuffle examples before splitting

    Yields:
    X_batch     -- shape (n_x, batch_len)
    Y_batch     -- shape (n_y, batch_len)
    """
    m = X.shape[1]
    
    
    indices = np.arange(m)
    
    if shuffle:
        np.random.shuffle(indices)

    for start in range(0, m, batch_size):
        end = start + batch_size
        batch_idx = indices[start:end]
        X_batch = X[:, batch_idx]
        Y_batch = Y[:, batch_idx]
        yield X_batch, Y_batch

def compute_cost(AL, Y):
    """
    Compute binary cross‑entropy cost and its derivative w.r.t. AL.

    Arguments:
    AL -- probability vector from forward propagation, shape (1, m)
    Y  -- true labels vector (0/1), shape (1, m)

    Returns:
    cost -- scalar value of the loss
    dAL  -- derivative of the loss w.r.t. AL, shape (1, m)
    """
    m = Y.shape[1]
    # Compute cost
    cost = - (1/m) * np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL))
    cost = np.squeeze(cost)
    # Derivative of cost w.r.t. AL
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    return cost, dAL
