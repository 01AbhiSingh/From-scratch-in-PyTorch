import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMCell(nn.Module):
    """
    A single LSTM cell implementation from scratch.
    
    LSTM cells contain gates that regulate the flow of information:
    - Forget gate: decides what information to throw away from the cell state
    - Input gate: decides what new information to store in the cell state
    - Output gate: decides what parts of the cell state to output
    
    Each gate uses a sigmoid activation function to output values between 0 and 1,
    determining how much information should pass through.
    """
    
    def __init__(self, input_size, hidden_size, bias=True):
        """
        Initialize the LSTM cell.
        
        Args:
            input_size (int): Size of the input features
            hidden_size (int): Size of the hidden state and cell state
            bias (bool): Whether to use bias terms in the linear transformations
        """
        super(LSTMCell, self).__init__()
        
        # Store the hidden size for later use
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        
        # Create a single linear transformation for all gates to optimize computation
        # This combines the transformations for the forget gate, input gate, cell candidate, and output gate
        # The input to this layer is [input, hidden_state] concatenated together
        self.lstm_units = nn.Linear(
            input_size + hidden_size,  # Concatenate input and previous hidden state
            hidden_size * 4,  # Multiply by 4 because we compute 4 gates: forget, input, cell, output
            bias=bias
        )
        
        # Initialize parameters with Glorot/Xavier initialization
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize weights and biases using Xavier uniform initialization."""
        nn.init.xavier_uniform_(self.lstm_units.weight)
        
        if self.bias:
            # Initialize biases to zero except for the forget gate bias
            # which is initialized to 1.0 to help with learning long-term dependencies
            nn.init.zeros_(self.lstm_units.bias)
            # Set forget gate bias to 1.0 (this is the first quarter of the bias)
            nn.init.ones_(self.lstm_units.bias[0:self.hidden_size])
    
    def forward(self, x, hidden_state=None):
        """
        Forward pass of the LSTM cell.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, input_size]
            hidden_state (tuple): Tuple containing (hidden_state, cell_state) from previous step,
                                  both of shape [batch_size, hidden_size]
                                  
        Returns:
            tuple: New (hidden_state, cell_state)
        """
        # Get batch size from input
        batch_size = x.size(0)
        
        # Initialize hidden state and cell state if not provided
        if hidden_state is None:
            # Create zero tensors for initial hidden state and cell state
            h_0 = torch.zeros(batch_size, self.hidden_size, device=x.device)
            c_0 = torch.zeros(batch_size, self.hidden_size, device=x.device)
            hidden_state = (h_0, c_0)
        
        # Unpack the hidden state
        h_prev, c_prev = hidden_state
        
        # Concatenate input and previous hidden state
        # This is done to perform a single matrix multiplication for all gates
        combined = torch.cat([x, h_prev], dim=1)
        
        # Transform the combined input through a single linear layer to calculate all gates at once
        # This is more efficient than having separate linear layers for each gate
        lstm_out = self.lstm_units(combined)
        
        # Split the output into the 4 gates
        # Each gate has size [batch_size, hidden_size]
        gates = torch.split(lstm_out, self.hidden_size, dim=1)
        
        # Extract individual gates
        forget_gate = torch.sigmoid(gates[0])  # Forget gate - decides what to keep/forget from previous cell state
        input_gate = torch.sigmoid(gates[1])   # Input gate - decides what new info to add to cell state
        cell_candidate = torch.tanh(gates[2])  # Candidate values - potential info to add to cell state
        output_gate = torch.sigmoid(gates[3])  # Output gate - decides what to output from cell state
        
        # Update cell state: forget old info + add new info
        # forget_gate * c_prev: selectively forget parts of the old cell state
        # input_gate * cell_candidate: selectively add new candidate values
        c_next = forget_gate * c_prev + input_gate * cell_candidate
        
        # Compute new hidden state
        # output_gate determines how much of the cell state to expose
        # tanh(c_next) squashes cell values to [-1, 1] range before output gate filtering
        h_next = output_gate * torch.tanh(c_next)
        
        return (h_next, c_next)


class LSTM(nn.Module):
    """
    Full LSTM layer that processes sequences using multiple LSTM cells.
    
    This implements a multi-layer LSTM network that can process entire sequences.
    """
    
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=True, dropout=0):
        """
        Initialize the LSTM layer.
        
        Args:
            input_size (int): Size of input features
            hidden_size (int): Size of hidden state and cell state
            num_layers (int): Number of stacked LSTM layers
            bias (bool): Whether to use bias in linear transformations
            batch_first (bool): If True, input shape is [batch_size, seq_len, features]
                                If False, input shape is [seq_len, batch_size, features]
            dropout (float): Dropout probability between LSTM layers (not after the last layer)
        """
        super(LSTM, self).__init__()
        
        # Store configuration
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        
        # Create a ModuleList to store multiple LSTM cells
        self.lstm_cells = nn.ModuleList()
        
        # Add the first LSTM cell (input_size -> hidden_size)
        self.lstm_cells.append(LSTMCell(input_size, hidden_size, bias))
        
        # Add remaining LSTM cells if num_layers > 1 (hidden_size -> hidden_size)
        for _ in range(1, num_layers):
            self.lstm_cells.append(LSTMCell(hidden_size, hidden_size, bias))
        
        # Create dropout layer to use between LSTM layers
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(self, x, hidden_state=None):
        """
        Forward pass for the full LSTM.
        
        Args:
            x (torch.Tensor): Input sequences
                             If batch_first=True, shape is [batch_size, seq_len, input_size]
                             If batch_first=False, shape is [seq_len, batch_size, input_size]
            hidden_state (tuple): Initial (hidden_state, cell_state) for each layer
                                  Each has shape [num_layers, batch_size, hidden_size]
        
        Returns:
            tuple: (
                output: Tensor containing the output features from the last layer for each time step,
                (h_n, c_n): Tuple containing the final hidden state and cell state for each layer
            )
        """
        # Determine batch size and sequence length based on batch_first flag
        if self.batch_first:
            batch_size, seq_len, _ = x.size()
            # Permute to have sequence length as the first dimension for easier iteration
            x = x.permute(1, 0, 2)
        else:
            seq_len, batch_size, _ = x.size()
        
        # Initialize hidden states if not provided
        if hidden_state is None:
            # Create list to hold h_0 and c_0 for each layer
            h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
            c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
            
            # Create list of layer states, each containing (h_0[layer], c_0[layer])
            hidden_state = [(h_0[i], c_0[i]) for i in range(self.num_layers)]
        else:
            # Unpack provided hidden states
            h_n, c_n = hidden_state
            hidden_state = [(h_n[i], c_n[i]) for i in range(self.num_layers)]
        
        # Container for output sequence (last layer's hidden states)
        output_sequence = []
        
        # Process each time step
        for t in range(seq_len):
            # Get input for current time step across all batches
            x_t = x[t]
            
            # Container for hidden states for this time step
            layer_output = x_t
            
            # Process each layer
            new_hidden_state = []
            for layer_idx, lstm_cell in enumerate(self.lstm_cells):
                # Forward through LSTM cell
                h_t, c_t = lstm_cell(layer_output, hidden_state[layer_idx])
                
                # Store new hidden state for this layer
                new_hidden_state.append((h_t, c_t))
                
                # Set output from this layer as input to next layer
                layer_output = h_t
                
                # Apply dropout except after the last layer
                if layer_idx < self.num_layers - 1 and self.dropout > 0:
                    layer_output = self.dropout_layer(layer_output)
            
            # Update hidden states for all layers
            hidden_state = new_hidden_state
            
            # Collect output from the last layer (used for output sequence)
            output_sequence.append(layer_output)
        
        # Stack outputs from all time steps
        outputs = torch.stack(output_sequence, dim=0)
        
        # Prepare final hidden and cell states for return
        h_n = torch.stack([h for h, _ in hidden_state], dim=0)
        c_n = torch.stack([c for _, c in hidden_state], dim=0)
        
        # Reshape output to match expected format
        if self.batch_first:
            # Change from [seq_len, batch, hidden] to [batch, seq_len, hidden]
            outputs = outputs.permute(1, 0, 2)
        
        return outputs, (h_n, c_n)


# Example usage
def lstm_example():
    """Example showing how to use the custom LSTM implementation."""
    # Define LSTM parameters
    batch_size = 3
    seq_len = 5
    input_size = 10
    hidden_size = 20
    num_layers = 2
    
    # Create random input data
    # Shape: [batch_size, seq_len, input_size] for batch_first=True
    x = torch.randn(batch_size, seq_len, input_size)
    
    # Create LSTM instance
    lstm = LSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        batch_first=True,
        dropout=0.2
    )
    
    # Forward pass
    output, (h_n, c_n) = lstm(x)
    
    # Print shapes
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Final hidden state shape: {h_n.shape}")
    print(f"Final cell state shape: {c_n.shape}")
    
    # Verify shapes are as expected
    assert output.shape == (batch_size, seq_len, hidden_size)
    assert h_n.shape == (num_layers, batch_size, hidden_size)
    assert c_n.shape == (num_layers, batch_size, hidden_size)
    
    print("All shapes match expected dimensions!")
    
    # Compare with PyTorch's built-in LSTM
    pytorch_lstm = nn.LSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        batch_first=True,
        dropout=0.2
    )
    
    # Forward pass with PyTorch LSTM
    pytorch_output, (pytorch_h_n, pytorch_c_n) = pytorch_lstm(x)
    
    # Verify output shapes match
    assert pytorch_output.shape == output.shape
    assert pytorch_h_n.shape == h_n.shape
    assert pytorch_c_n.shape == c_n.shape
    
    print("Custom LSTM implementation produces same output shapes as PyTorch's built-in LSTM!")

if __name__ == "__main__":
    lstm_example()
