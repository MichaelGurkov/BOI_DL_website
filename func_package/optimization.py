import numpy as np

def initialize_optimizer_state(parameters, optimizer="gd", **hyperparams):
  """
    Create and return the optimizer_state dict for the chosen optimizer.
    parameters -- dict with W1...WL, b1...bL
    optimizer  -- "gd", "momentum", "rmsprop", or "adam"
    hyperparams-- any extra params like beta, beta1, beta2, epsilon
    """
  state = {}
  L = len(parameters) // 2
  
  if optimizer == "momentum":
    # zero velocities
    velocity = {}
    for l in range(1, L+1):
      velocity[f"dW{l}"] = np.zeros_like(parameters[f"W{l}"])
      velocity[f"db{l}"] = np.zeros_like(parameters[f"b{l}"])
    state["velocity"] = velocity
  
  elif optimizer == "rmsprop":
    # zero cache for squared grads
    s = {}
    for l in range(1, L+1):
      s[f"dW{l}"] = np.zeros_like(parameters[f"W{l}"])
      s[f"db{l}"] = np.zeros_like(parameters[f"b{l}"])
    state["s"]       = s
    state["beta2"]   = hyperparams.get("beta2", 0.999)
    state["epsilon"] = hyperparams.get("epsilon", 1e-8)
  
  elif optimizer == "adam":
    # zero v and s, plus timestep
    v = {}
    s = {}
    for l in range(1, L+1):
      v[f"dW{l}"] = np.zeros_like(parameters[f"W{l}"])
      v[f"db{l}"] = np.zeros_like(parameters[f"b{l}"])
      s[f"dW{l}"] = np.zeros_like(parameters[f"W{l}"])
      s[f"db{l}"] = np.zeros_like(parameters[f"b{l}"])
    state["v"] = v
    state["s"] = s
    state["t"] = 0
    state["beta1"] = hyperparams.get("beta1", 0.9)
    state["beta2"] = hyperparams.get("beta2", 0.999)
    state["epsilon"] = hyperparams.get("epsilon", 1e-8)
  

  return state


def update_parameters(parameters, grads, optimizer_state,
                      optimizer="gd", learning_rate=0.01, **hyperparams):
  """
    Public dispatcher to update parameters with the chosen optimizer.
    Returns (updated_parameters, updated_optimizer_state).
    """
  if optimizer == "gd":
    new_params = _update_gd(parameters, grads, learning_rate)
    return new_params, optimizer_state

  elif optimizer == "momentum":
    new_params, new_velocity = _update_momentum(
      parameters, grads,
      optimizer_state["velocity"],
      learning_rate,
      beta=hyperparams.get("beta", 0.9)
    )
    optimizer_state["velocity"] = new_velocity
    return new_params, optimizer_state

  elif optimizer == "rmsprop":
    new_params, new_s = _update_rmsprop(
      parameters, grads,
      optimizer_state["s"],
      learning_rate,
      beta2=optimizer_state.get("beta2", hyperparams.get("beta2", 0.999)),
      epsilon=optimizer_state.get("epsilon", hyperparams.get("epsilon", 1e-8))
    )
    optimizer_state["s"] = new_s
    return new_params, optimizer_state

  elif optimizer == "adam":
    new_params, new_v, new_s, new_t = _update_adam(
      parameters, grads,
      optimizer_state["v"],
      optimizer_state["s"],
      optimizer_state["t"],
      learning_rate,
      beta1=optimizer_state.get("beta1", hyperparams.get("beta1", 0.9)),
      beta2=optimizer_state.get("beta2", hyperparams.get("beta2", 0.999)),
      epsilon=optimizer_state.get("epsilon", hyperparams.get("epsilon", 1e-8))
    )
  optimizer_state["v"] = new_v
  optimizer_state["s"] = new_s
  optimizer_state["t"] = new_t
  return new_params, optimizer_state

# ---- Private helper functions ----

def _update_gd(parameters, grads, learning_rate):
  
  L = len(parameters) // 2
  
  for l in range(1, L+1):
    parameters[f"W{l}"] -= learning_rate * grads[f"dW{l}"]
    parameters[f"b{l}"] -= learning_rate * grads[f"db{l}"]
  
  return parameters

def _update_momentum(parameters, grads, velocity, learning_rate, beta):
  L = len(parameters) // 2
  for l in range(1, L+1):
    velocity[f"dW{l}"] = beta * velocity[f"dW{l}"] \
  + (1 - beta) * grads[f"dW{l}"]
  
  velocity[f"db{l}"] = beta * velocity[f"db{l}"] \
  + (1 - beta) * grads[f"db{l}"]
  
  parameters[f"W{l}"] -= learning_rate * velocity[f"dW{l}"]
  parameters[f"b{l}"] -= learning_rate * velocity[f"db{l}"]
  
  return parameters, velocity

def _update_rmsprop(parameters, grads, s, learning_rate, beta2, epsilon):
  
  L = len(parameters) // 2
  
  for l in range(1, L+1):
    s[f"dW{l}"] = beta2 * s[f"dW{l}"] + (1 - beta2) * (grads[f"dW{l}"]**2)
    s[f"db{l}"] = beta2 * s[f"db{l}"] + (1 - beta2) * (grads[f"db{l}"]**2)
    
    parameters[f"W{l}"] -= learning_rate * grads[f"dW{l}"] \
    / (np.sqrt(s[f"dW{l}"]) + epsilon)
    
    parameters[f"b{l}"] -= learning_rate * grads[f"db{l}"] \
    / (np.sqrt(s[f"db{l}"]) + epsilon)
    
  return parameters, s

def _update_adam(parameters, grads, v, s, t,
                 learning_rate, beta1, beta2, epsilon):
  
  L = len(parameters) // 2
  
  t += 1
  
  v_corr = {}
  
  s_corr = {}
  
  for l in range(1, L+1):
    
    # Update biased first moment estimate
    v[f"dW{l}"] = beta1 * v[f"dW{l}"] + (1 - beta1) * grads[f"dW{l}"]
    v[f"db{l}"] = beta1 * v[f"db{l}"] + (1 - beta1) * grads[f"db{l}"]
  
  # Update biased second moment estimate
    s[f"dW{l}"] = beta2 * s[f"dW{l}"] + (1 - beta2) * (grads[f"dW{l}"]**2)
    s[f"db{l}"] = beta2 * s[f"db{l}"] + (1 - beta2) * (grads[f"db{l}"]**2)
  
  # Compute bias-corrected estimates
    v_corr[f"dW{l}"] = v[f"dW{l}"] / (1 - beta1**t)
    v_corr[f"db{l}"] = v[f"db{l}"] / (1 - beta1**t)
    s_corr[f"dW{l}"] = s[f"dW{l}"] / (1 - beta2**t)
    s_corr[f"db{l}"] = s[f"db{l}"] / (1 - beta2**t)
  
  # Update parameters
    parameters[f"W{l}"] -= learning_rate * v_corr[f"dW{l}"] \
    / (np.sqrt(s_corr[f"dW{l}"]) + epsilon)
    
    parameters[f"b{l}"] -= learning_rate * v_corr[f"db{l}"] \
    / (np.sqrt(s_corr[f"db{l}"]) + epsilon)
    
  return parameters, v, s, t
