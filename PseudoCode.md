BEGIN PROGRAM

1. Setup Neural Network Structure
   - Input: a list with neuron counts for each layer
   - Initialize:
       For every pair of consecutive layers:
           Create a weight matrix (random values) including bias weights
   - Define helper functions:
       * Activation (sigmoid)
       * Forward pass (predict)
       * Training (forward + backpropagation + weight updates)

2. Load and Prepare Data
   - Import MNIST dataset
   - Scale pixel values to [0, 1]
   - Convert labels to one-hot encoded format

3. Build Model
   - Example architecture: [784, 1250, 10]
     (input, hidden, output)

4. Training Phase
   LOOP for a fixed number of iterations (e.g., 30,000):
       - Select a random training image and its label
       - Run forward pass to get output
       - Compute prediction error
       - Perform backpropagation to calculate gradients
       - Adjust weights using learning rate
       - Store error for monitoring
       IF iteration is multiple of 1000:
           - Calculate moving average error
           - Print current step and error value

5. Visualize Training Progress
   - Plot average error vs. iteration steps
   - Save the figure in high resolution for reporting

6. Model Evaluation
   - Load MNIST test set
   - Normalize inputs
   - Predict labels for test images
   - Count number of correct predictions
   - Calculate accuracy percentage
   - Display final accuracy score

END PROGRAM
