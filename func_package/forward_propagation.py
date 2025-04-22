import numpy as np

def linear_forward(A_prev, W, b):
    """
    Compute the linear part of a layer’s forward propagation.

    Arguments:
    A_prev -- activations from previous layer, shape (n_prev, m)
    W      -- weights matrix for current layer, shape (n_curr, n_prev)
    b      -- bias vector for current layer, shape (n_curr, 1)

    Returns:
    Z      -- pre-activation parameter, shape (n_curr, m)
    cache  -- tuple (A_prev, W, b) for backprop
    """
    Z = np.dot(W, A_prev) + b
    cache = (A_prev, W, b)
    return Z, cache

def relu(Z):
    """
    ReLU activation.

    Arguments:
    Z -- pre-activation parameter

    Returns:
    A      -- post-activation, same shape as Z
    cache  -- Z, for backprop
    """
    A = np.maximum(0, Z)
    cache = Z
    return A, cache

def sigmoid(Z):
    """
    Sigmoid activation.

    Arguments:
    Z -- pre-activation parameter

    Returns:
    A      -- post-activation, same shape as Z
    cache  -- Z, for backprop
    """
    A = 1 / (1 + np.exp(-Z))
    cache = Z
    return A, cache

def activation_forward(A_prev, W, b, activation):
    """
    Forward propagation for a single layer: linear -> activation.

    Arguments:
    A_prev     -- activations from previous layer
    W, b       -- parameters for this layer
    activation -- "relu" or "sigmoid"

    Returns:
    A      -- post-activation output
    cache  -- (linear_cache, activation_cache)
    """
    Z, linear_cache = linear_forward(A_prev, W, b)

    if activation == "relu":
        A, activation_cache = relu(Z)
    elif activation == "sigmoid":
        A, activation_cache = sigmoid(Z)

    cache = (linear_cache, activation_cache)

    return A, cache

def forward_propagation(X, weights, biases,
                        hidden_activation="relu",
                        output_activation="sigmoid"):
    """
    Implements forward propagation for the whole network.

    Arguments:
    X                 -- input data, shape (n_x, m)
    weights           -- dict of weight matrices W1...WL
    biases            -- dict of bias vectors b1...bL
    hidden_activation -- activation for hidden layers ("relu")
    output_activation -- activation for output layer ("sigmoid")

    Returns:
    AL      -- last post-activation value (prediction), shape (n_L, m)
    caches  -- list of caches for each layer
    """
    caches = []
    A = X
    L = len(weights)

    # Hidden layers
    for l in range(1, L):
        A_prev = A
        A, cache = activation_forward(
            A_prev,
            weights[f"W{l}"],
            biases[f"b{l}"],
            hidden_activation
        )
        caches.append(cache)

    # Output layer
    AL, cache = activation_forward(
        A,
        weights[f"W{L}"],
        biases[f"b{L}"],
        output_activation
    )
    caches.append(cache)

    return AL, caches
