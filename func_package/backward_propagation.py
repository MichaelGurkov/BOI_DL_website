import numpy as np

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


def linear_backward(dZ, linear_cache):
    """
    Linear portion of backward propagation for a single layer.

    Arguments:
    dZ           -- gradient of the cost w.r.t. the linear output Z, shape (n_curr, m)
    linear_cache -- tuple of values (A_prev, W, b) from forward propagation

    Returns:
    dA_prev -- gradient of the cost w.r.t. the activation from previous layer, shape (n_prev, m)
    dW      -- gradient of the cost w.r.t. W, shape (n_curr, n_prev)
    db      -- gradient of the cost w.r.t. b, shape (n_curr, 1)
    """
    A_prev, W, b = linear_cache
    m = A_prev.shape[1]

    dW = (1/m) * np.dot(dZ, A_prev.T)
    db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db

def relu_backward(dA, activation_cache):
    """
    Backward pass for a ReLU activation.

    Arguments:
    dA               -- post-activation gradient, same shape as Z
    activation_cache -- Z stored during forward pass

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    Z = activation_cache
    dZ = np.array(dA, copy=True)  # just propagate where Z > 0
    dZ[Z <= 0] = 0
    return dZ

def sigmoid_backward(dA, activation_cache):
    """
    Backward pass for a sigmoid activation.

    Arguments:
    dA               -- post-activation gradient, same shape as Z
    activation_cache -- Z stored during forward pass

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    Z = activation_cache
    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)
    return dZ

def activation_backward(dA, cache, activation):
    """
    Backward propagation for the linear->activation layer.

    Arguments:
    dA         -- post-activation gradient for current layer
    cache      -- tuple of (linear_cache, activation_cache) from forward pass
    activation -- "relu" or "sigmoid"

    Returns:
    dA_prev -- Gradient of the cost w.r.t. activation from previous layer
    dW      -- Gradient of the cost w.r.t. W (current layer)
    db      -- Gradient of the cost w.r.t. b (current layer)
    """
    linear_cache, activation_cache = cache
    

    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)


    dA_prev, dW, db = linear_backward(dZ, linear_cache)
    
    return dA_prev, dW, db


def backward_propagation(AL, Y, caches,
                         hidden_activation="relu",
                         output_activation="sigmoid"):
    """
    Implements the backward propagation for the entire network.

    Arguments:
    AL                 -- probability vector from forward propagation, shape (n_L, m)
    Y                  -- true labels vector, shape (n_L, m)
    caches             -- list of caches from forward_propagation, of length L
    hidden_activation  -- activation used in hidden layers ("relu")
    output_activation  -- activation used in output layer ("sigmoid")

    Returns:
    grads -- dictionary with the gradients
             grads["dW1"], ..., grads["dWL"]
             grads["db1"], ..., grads["dbL"]
    """
    grads = {}
    L = len(caches)            # number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)    # ensure Y has the same shape as AL

    # 1) Initial gradient from the cost with respect to AL
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    # 2) Backprop through output layer (layer L)
    current_cache = caches[L-1]
    dA_prev, dW, db = activation_backward(dAL,
                                          current_cache,
                                          activation=output_activation)
    grads[f"dW{L}"] = dW
    grads[f"db{L}"] = db
    grads[f"dA{L-1}"] = dA_prev

    # 3) Backprop through hidden layers (layers L-1 to 1)
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_curr = grads[f"dA{l+1}"]
        dA_prev, dW, db = activation_backward(dA_curr,
                                              current_cache,
                                              activation=hidden_activation)
        grads[f"dW{l+1}"] = dW
        grads[f"db{l+1}"] = db
        grads[f"dA{l}"] = dA_prev

    return grads
