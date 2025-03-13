 INSTRUCTIONS
 
In this assignment, you will implement a training loop for a deep learning model using the chain rule to compute gradients manually. You must not use torch.backward() or torch.optim. Instead, you will derive gradients using manual differentiation and update the model parameters accordingly.


Model Architecture

The model has:
    Two inputs: x subscript 1​ and x subscript 2
    Two layers:
        First layer: 3 neurons
        Second layer: 2 neurons
        Output layer: 1 neuron