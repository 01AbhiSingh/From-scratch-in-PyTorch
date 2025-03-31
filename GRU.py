import torch
import torch.nn as nn
import torch.nn.functional as F

class GRUCell(nn.Module):
    """
    Gated Recurrent Unit (GRU) Cell Implementation
    
    The GRU is a sophisticated neural network architecture designed to solve key challenges 
    in traditional Recurrent Neural Networks (RNNs):
    1. Vanishing Gradient Problem: By using gates, GRUs can learn long-term dependencies
    2. Information Control: Gates decide what information to keep, update, or discard
    3. Computational Efficiency: Lighter design compared to LSTM networks
    
    Core Components:
    - Update Gate (z): Controls information flow from previous hidden state
    - Reset Gate (r): Determines how much of previous hidden state to forget
    - Candidate Hidden State: Proposes a new potential hidden state
    """
    def __init__(self, input_size, hidden_size):
        """
        Initialize GRU Cell with specific architectural parameters
        
        Args:
            input_size (int): Dimension of the input features 
                             - Represents the number of features in each input vector
                             - Crucial for defining the input layer's weight matrix dimensionality
            
            hidden_size (int): Dimension of the hidden state
                              - Determines the neural network's capacity to capture complex patterns
                              - Larger values can represent more sophisticated representations
        """
        super(GRUCell, self).__init__()
        
        # WEIGHT MATRICES FOR UPDATE GATE
        # Wz: Weight matrix that transforms input and previous hidden state 
        # into update gate values
        # Learns how much of the previous hidden state to keep
        self.Wz = nn.Parameter(torch.Tensor(hidden_size, input_size + hidden_size))
        
        # Bias term for update gate
        # Allows the network to shift the activation function's threshold
        self.bz = nn.Parameter(torch.Tensor(hidden_size))
        
        # WEIGHT MATRICES FOR RESET GATE
        # Wr: Weight matrix that determines how much of the previous 
        # hidden state should be reset/forgotten
        self.Wr = nn.Parameter(torch.Tensor(hidden_size, input_size + hidden_size))
        
        # Bias term for reset gate
        self.br = nn.Parameter(torch.Tensor(hidden_size))
        
        # WEIGHT MATRICES FOR CANDIDATE HIDDEN STATE
        # Wh: Weight matrix for computing the candidate hidden state
        # Proposes a potential new hidden state representation
        self.Wh = nn.Parameter(torch.Tensor(hidden_size, input_size + hidden_size))
        
        # Bias term for candidate hidden state computation
        self.bh = nn.Parameter(torch.Tensor(hidden_size))
        
        # Initialize all parameters using a smart initialization strategy
        self.reset_parameters()
    
    def reset_parameters(self):
        """
        Parameter Initialization Strategy
        
        Why Xavier/Glorot Initialization?
        - Prevents vanishing/exploding gradients during training
        - Maintains variance of activations across layers
        - Helps stabilize deep neural network training
        
        Initialization Process:
        1. Weights are drawn from a uniform distribution
        2. Distribution's range is carefully calculated to maintain signal variance
        3. Biases are initialized to zero to start with a neutral baseline
        """
        # Xavier Uniform initialization for weight matrices
        # Ensures weights are not too large or too small
        nn.init.xavier_uniform_(self.Wz)
        nn.init.xavier_uniform_(self.Wr)
        nn.init.xavier_uniform_(self.Wh)
        
        # Zero initialization for bias terms
        # Starting point is a neutral state before learning
        nn.init.zeros_(self.bz)
        nn.init.zeros_(self.br)
        nn.init.zeros_(self.bh)
    
    def forward(self, x, h_prev):
        """
        Forward Propagation: The Heart of GRU Computation
        
        Args:
            x (Tensor): Current time step's input
                       - Represents the current input vector
                       - Contains features/information for current moment
            
            h_prev (Tensor): Previous hidden state
                             - Captures historical context and learned representations
                             - Serves as memory from previous computations
        
        Returns:
            Tensor: Newly computed hidden state
                   - Integrates current input with historical context
                   - Represents the network's updated understanding
        """
        # STEP 1: Concatenate Current Input and Previous Hidden State
        # Why concatenate? 
        # - Combines current moment's information with historical context
        # - Allows neural network to make informed, context-aware decisions
        combined = torch.cat([x, h_prev], dim=1)
        
        # STEP 2: Compute Update Gate (z)
        # Purpose: Determine how much of previous hidden state to keep
        # Sigmoid Activation: Squashes values between 0 and 1
        # - 0: Completely discard previous hidden state
        # - 1: Fully retain previous hidden state
        # F.linear(): Applies linear transformation (Wx + b)
        z = torch.sigmoid(F.linear(combined, self.Wz, self.bz))
        
        # STEP 3: Compute Reset Gate (r)
        # Purpose: Decide how much of previous hidden state to reset/forget
        # Similar sigmoid activation, but with different learned weights
        # Helps network adaptively forget irrelevant historical information
        r = torch.sigmoid(F.linear(combined, self.Wr, self.br))
        
        # STEP 4: Prepare Input for Candidate Hidden State
        # Elementwise multiplication: r * h_prev
        # Allows reset gate to selectively suppress parts of previous hidden state
        combined_reset = torch.cat([x, r * h_prev], dim=1)
        
        # STEP 5: Compute Candidate Hidden State
        # Propose a potential new hidden state representation
        # Tanh activation bounds values between -1 and 1
        # Helps stabilize gradient flow and representation
        h_candidate = torch.tanh(F.linear(combined_reset, self.Wh, self.bh))
        
        # STEP 6: Compute Final Hidden State
        # Intelligent blending of previous state and candidate state
        # (1 - z) * h_prev: Portion of previous state to keep
        # z * h_candidate: Portion of new candidate state to incorporate
        h_new = (1 - z) * h_prev + z * h_candidate
        
        return h_new

class GRU(nn.Module):
    """
    Full GRU Layer for Processing Sequential Data
    
    Extends single GRUCell to handle:
    - Multiple time steps
    - Multiple layers
    - Batch processing
    """
    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=True):
        """
        Configurable GRU Layer Initialization
        
        Args:
            input_size (int): Dimension of input features
            hidden_size (int): Dimension of hidden state representation
            num_layers (int): Number of stacked GRU layers
                             - Deeper networks can capture more complex patterns
                             - Each layer learns increasingly abstract representations
            batch_first (bool): Input tensor dimension order
                               - True: (batch, sequence_length, features)
                               - False: (sequence_length, batch, features)
        """
        super(GRU, self).__init__()
        
        # Store architectural hyperparameters
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        
        # Create Stack of GRU Cells
        # Each layer takes output of previous layer as input
        # First layer uses original input_size, subsequent layers use hidden_size
        self.gru_cells = nn.ModuleList([
            GRUCell(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])
    
    def forward(self, x, h0=None):
        """
        Process Sequential Input Through Multiple GRU Layers
        
        Args:
            x (Tensor): Input sequence tensor
            h0 (Tensor, optional): Initial hidden state for each layer
                                   - If None, starts with zero hidden state
        
        Returns:
            output (Tensor): Processed sequence
            hn (Tensor): Final hidden state of the last layer
        """
        # STEP 1: Ensure Correct Input Tensor Dimension
        # Transpose if batch_first is True to standardize processing
        if self.batch_first:
            x = x.transpose(0, 1)  # (seq_len, batch, input_size)
        
        # Extract crucial tensor dimensions
        seq_len, batch_size, input_size = x.size()
        
        # STEP 2: Initialize Hidden State
        # If no initial hidden state provided, create zero tensor
        if h0 is None:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, 
                             device=x.device)
        
        # Convert initial hidden state to list for layer-wise processing
        hidden = list(h0)
        
        # Output list to store hidden states at each time step
        outputs = []
        
        # STEP 3: Sequentially Process Each Time Step
        for t in range(seq_len):
            layer_input = x[t]  # Current time step's input
            
            # Process through each GRU layer
            for layer in range(self.num_layers):
                # Update hidden state for current layer
                # Each layer takes input from previous layer
                hidden[layer] = self.gru_cells[layer](layer_input, hidden[layer])
                layer_input = hidden[layer]
            
            # Store output of last layer for this time step
            outputs.append(hidden[-1])
        
        # Convert outputs to tensor
        outputs = torch.stack(outputs)
        
        # Restore original tensor dimension if batch_first was True
        if self.batch_first:
            outputs = outputs.transpose(0, 1)
        
        return outputs, hidden[-1]

# Demonstration of GRU usage
if __name__ == "__main__":
    # Define Realistic Hyperparameters
    input_size = 10   # Number of input features
    hidden_size = 20  # Neural network's representational capacity
    num_layers = 2    # Depth of network
    batch_size = 32   # Number of sequences processed simultaneously
    seq_len = 50      # Length of each sequence

    # Generate Random Sequence Data
    # Simulates real-world sequential input like time series or text
    x = torch.randn(batch_size, seq_len, input_size)
    
    # Instantiate Custom GRU
    gru = GRU(input_size, hidden_size, num_layers)
    
    # Perform Forward Pass
    outputs, final_hidden = gru(x)
    
    # Demonstrate Output Shapes
    print("Output shape:", outputs.shape)
    print("Final hidden state shape:", final_hidden.shape)
