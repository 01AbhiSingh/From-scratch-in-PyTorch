import numpy as np

class RNN:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Initialize weights with random values
        # Wxh: weights from input to hidden
        self.Wxh = np.random.randn(hidden_size, input_size) * 0.01
        # Whh: weights from hidden to hidden
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01
        # Why: weights from hidden to output
        self.Why = np.random.randn(output_size, hidden_size) * 0.01
        
        # Initialize biases
        self.bh = np.zeros((hidden_size, 1))  # hidden bias
        self.by = np.zeros((output_size, 1))  # output bias
        
    def forward(self, inputs):
        
        # Initialize hidden state as zeros
        h = np.zeros((self.hidden_size, 1))
        
        # Store all hidden states and outputs
        hidden_states = [h]
        outputs = []
        
        # Forward pass for each time step
        for x in inputs:
            # Convert input to column vector if needed
            if x.ndim == 1:
                x = x.reshape(-1, 1)
            
            # Update hidden state: h_t = tanh(W_xh * x_t + W_hh * h_{t-1} + b_h)
            h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh)
            
            # Compute output: y_t = W_hy * h_t + b_y
            y = np.dot(self.Why, h) + self.by
            
            # Store current hidden state and output
            hidden_states.append(h)
            outputs.append(y)
        
        return hidden_states, outputs
    
    def backward(self, inputs, targets, hidden_states, outputs):
       
        # Initialize gradients
        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dWhy = np.zeros_like(self.Why)
        dbh = np.zeros_like(self.bh)
        dby = np.zeros_like(self.by)
        
        # Initialize total loss
        loss = 0
        
        # Initialize gradient of hidden state at the end
        dh_next = np.zeros_like(hidden_states[0])
        
        # Backward pass, starting from the end
        for t in reversed(range(len(inputs))):
            # Current target and output
            target = targets[t]
            if target.ndim == 1:
                target = target.reshape(-1, 1)
            
            y = outputs[t]
            
            # Compute loss (e.g., mean squared error)
            loss += np.sum((y - target) ** 2) / 2
            
            # Gradient of output weights
            dy = y - target
            dWhy += np.dot(dy, hidden_states[t+1].T)
            dby += dy
            
            # Gradient of hidden state
            dh = np.dot(self.Why.T, dy) + dh_next
            
            # Gradient through tanh
            dhraw = (1 - hidden_states[t+1] ** 2) * dh
            
            # Gradient of biases
            dbh += dhraw
            
            # Gradient of weights
            dWxh += np.dot(dhraw, inputs[t].reshape(1, -1))
            dWhh += np.dot(dhraw, hidden_states[t].T)
            
            # Gradient for next iteration
            dh_next = np.dot(self.Whh.T, dhraw)
        
        # Clip gradients to prevent exploding gradients
        for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(dparam, -5, 5, out=dparam)
        
        # Update weights
        self.Wxh -= self.learning_rate * dWxh
        self.Whh -= self.learning_rate * dWhh
        self.Why -= self.learning_rate * dWhy
        self.bh -= self.learning_rate * dbh
        self.by -= self.learning_rate * dby
        
        return loss
    
    def train(self, inputs, targets, epochs=100):
      
        losses = []
        
        for epoch in range(epochs):
            total_loss = 0
            
            for i in range(len(inputs)):
                # Forward pass
                hidden_states, outputs = self.forward(inputs[i])
                
                # Backward pass
                loss = self.backward(inputs[i], targets[i], hidden_states, outputs)
                total_loss += loss
            
            # Average loss for this epoch
            avg_loss = total_loss / len(inputs)
            losses.append(avg_loss)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {avg_loss}")
        
        return losses
    
    def predict(self, x_sequence):
        
        hidden_states, outputs = self.forward(x_sequence)
        return outputs

# Example usage
if __name__ == "__main__":
    # Create a simple time series dataset: y[t] = sin(x[t-3])
    # This demonstrates the RNN's ability to learn temporal dependencies
    
    # Generate data
    x = np.linspace(0, 20, 100)
    y = np.sin(x)
    
    # Create sequences
    seq_length = 10
    x_seqs = []
    y_seqs = []
    
    for i in range(len(x) - seq_length):
        x_seqs.append(x[i:i+seq_length])
        y_seqs.append(y[i:i+seq_length])
    
    # Convert to numpy arrays
    x_data = [np.array([xi]) for xi in x_seqs]
    y_data = [np.array([yi]) for yi in y_seqs]
    
    # Create and train RNN
    rnn = RNN(input_size=1, hidden_size=16, output_size=1)
    losses = rnn.train(x_data[:70], y_data[:70], epochs=100)
    
    # Test RNN
    test_seq = x_data[80]
    predictions = rnn.predict(test_seq)
    
    print("Test sequence:", [x[0] for x in test_seq])
    print("Predictions:", [p[0][0] for p in predictions])
    print("Actual values:", [y[0] for y in y_data[80]])
