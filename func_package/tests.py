import numpy as np

from func_package.utils import initialize_parameters,split_parameters

from func_package.forward_propagation import forward_propagation

from func_package.backward_propagation import backward_propagation

from func_package.model import model

def test_forward_propagation():
  np.random.seed(42)
  # 1) toy data: 5 features, 3 examples
  X = np.random.randn(5, 3)
  
  # 2) toy network: 5→4→1
  layer_dims = [5, 4, 1]
  
  # --- Test zeros initialization ---
  params = initialize_parameters(layer_dims,initialization="zeros")
  weights_z, biases_z = split_parameters(params)
  
  AL_z, caches_z = forward_propagation(X, weights_z, biases_z)
  
  # Output should be sigmoid(0)=0.5 for every example
  assert AL_z.shape == (1, 3)
  assert np.allclose(AL_z, 0.5), f"Expected all 0.5, but got {AL_z}"
  
  # caches: one per layer (2 layers here)
  assert len(caches_z) == 2
  # check shapes in cache for layer 1
  A_prev1, W1, b1 = caches_z[0][0]
  assert A_prev1.shape == (5, 3)
  assert W1.shape      == (4, 5)
  assert b1.shape      == (4, 1)
    
  print("Forward_propagation passes basic sanity checks.")


def test_backward_propagation():
    np.random.seed(1)
    # 1) Toy dataset: 3 features, 4 examples
    X = np.random.randn(3, 4)
    Y = np.array([[1, 0, 1, 0]])  # binary labels

    # 2) Tiny network: 3 → 2 → 1
    layer_dims = [3, 2, 1]

    # 3) Initialize parameters to zeros
    params_z = initialize_parameters(layer_dims, initialization="zeros")
    weights_z, biases_z = split_parameters(params_z)

    # 4) Forward pass
    AL, caches = forward_propagation(X, weights_z, biases_z)

    # 5) Backward pass
    grads = backward_propagation(AL, Y, caches)

    # 6) Sanity checks:
    #    - hidden-layer grads (dW1, db1) should be all zeros
    assert np.allclose(grads["dW1"], np.zeros_like(grads["dW1"]))
    assert np.allclose(grads["db1"], np.zeros_like(grads["db1"]))

    #    - output-layer weight grad dW2 should be zeros since A1 was zeros
    assert np.allclose(grads["dW2"], np.zeros_like(grads["dW2"]))

    #    - output-layer bias grad db2 should equal -(1/m) * sum(AL - Y)
    m = Y.shape[1]
    expected_db2 = - (1/m) * np.sum(AL - Y)
    assert np.allclose(grads["db2"], expected_db2)

    print("Backward_propagation passes basic sanity checks.")


def test_model():
  
# ---- Minimal working example ----

# 1) Create a tiny toy dataset (XOR‑like)
  X = np.array([[0, 0, 1, 1],
                [0, 1, 0, 1]])   # shape (2, 4)
  Y = np.array([[0, 1, 1, 0]])     # shape (1, 4)
  
  # 2) Define a 2‑layer network: 2 inputs → 4 hidden units → 1 output
  layer_dims = [2, 4, 1]

  # 3) Train with a few epochs
  # parameters, costs = model(
  #   X, Y, layer_dims,
  #   optimizer="gd",         # full‑batch gradient descent
  #   learning_rate=0.1,
  #   num_epochs=1000,
  #   batch_size=None,        # None means full batch
  #   print_cost=False,
  #   print_every=200
  #   )
  
  parameters, costs = model(
  X, Y, layer_dims,
  optimizer="adam",
  learning_rate=0.01,      # you can try 0.01 or even 0.1 here
  num_epochs=2000,
  batch_size=None,
  print_cost=True,
  print_every=200,
  beta1=0.9,
  beta2=0.999,
  epsilon=1e-8
  )  
    
  # 4) Inspect training curve & final parameters
  print("Costs logged every 200 epochs:", costs)
  print("W1 shape:", parameters["W1"].shape)
  print("b1 shape:", parameters["b1"].shape)
  print("W2 shape:", parameters["W2"].shape)
  print("b2 shape:", parameters["b2"].shape)

  # 5) Quick prediction check
  def predict(X, parameters):
      # forward only
      weights, biases = split_parameters(parameters)
      AL, _ = forward_propagation(X, weights, biases)
      return (AL > 0.5).astype(int)

  preds = predict(X, parameters)
  print("Predictions on training set:", preds)
  print("True labels:            ", Y)
  accuracy = np.mean(preds == Y)
  print(f"Training accuracy: {accuracy * 100:.1f}%")
