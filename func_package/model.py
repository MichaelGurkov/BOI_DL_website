import numpy as np

from func_package.utils import (initialize_parameters,split_parameters,
get_minibatches, compute_cost)

from func_package.forward_propagation import forward_propagation

from func_package.backward_propagation import backward_propagation

from func_package.optimization import (initialize_optimizer_state,
update_parameters)


def model(X, Y, layer_dims,
          optimizer="gd",
          learning_rate=0.01,
          num_epochs=1000,
          batch_size=None,
          print_cost=False,
          print_every=100,
          **hyperparams):
  """
    Trains a L‑layer neural network.

    Arguments:
    X, Y            -- input data and labels, shapes (n_x, m), (n_y, m)
    layer_dims      -- list of layer dimensions [n_x, n_h1, ..., n_y]
    optimizer       -- "gd", "momentum", "rmsprop", or "adam"
    learning_rate   -- step size
    num_epochs      -- number of full passes over the data
    batch_size      -- size of mini‑batches; if None, uses full batch
    print_cost      -- if True, print cost every `print_every` epochs
    print_every     -- frequency (in epochs) of printing the cost
    **hyperparams   -- extra optimizer settings (beta, beta1, beta2, epsilon)

    Returns:
    parameters -- learned parameters dict (W1…WL, b1…bL)
    costs      -- list of costs printed
    """
  m = X.shape[1]
  
  if batch_size is None:
    batch_size = m
  
  # 1) Initialize parameters & optimizer state
  parameters = initialize_parameters(layer_dims)
  
  opt_state  = initialize_optimizer_state(parameters, optimizer, **hyperparams)
  costs = []
  
  # 2) Training loop
  for epoch in range(1, num_epochs+1):
    
    epoch_cost = 0
    
    minibatches = get_minibatches(X, Y, batch_size, shuffle=True)
  
    for X_batch, Y_batch in minibatches:
      # Forward
      weights, biases = split_parameters(parameters)
      AL, caches = forward_propagation(
      X_batch, weights, biases,
      hidden_activation="relu",
      output_activation="sigmoid"
      )
    
      # Compute cost & backward
      cost, _ = compute_cost(AL, Y_batch)
      grads = backward_propagation(
        AL, Y_batch, caches,
        hidden_activation="relu",
        output_activation="sigmoid"
      )
      
  
      # Update parameters
      parameters, opt_state = update_parameters(
        parameters, grads, opt_state,
        optimizer=optimizer,
        learning_rate=learning_rate,
        **hyperparams
      )
      epoch_cost += cost * (X_batch.shape[1] / m)
    
    # Record / print cost
    if epoch % print_every == 0:
      costs.append(epoch_cost)
    if print_cost:
      print(f"Cost after epoch {epoch}: {epoch_cost:.6f}")
  
  return parameters, costs
