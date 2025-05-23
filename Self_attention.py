import torch
import torch.nn as nn
import torch.nn.functional as F
import math

#define vocab

vocab = {"money": 0, "bank": 1, "grows": 2}

vocab_lenght = len(vocab)
embedding_dim = 4

#create embedding layer
#parameters= (no. of unique words, dimension of each embedding)

embedding_layer = nn.Embedding(num_embeddings = vocab_lenght, embedding_dim = embedding_dim)
print("Initial random embeddings stored in the layer:\n", embedding_layer.weight)

X = embedding_layer.weight.clone().detach()
print(X)

#convert words into integer ids
sentence_words = ['money', 'bank', 'grows']
sentence_indices = torch.tensor([vocab[word] for word in sentence_words], dtype = torch.long)
print(f"sentence as integer IDs: {sentence_indices}")

X = embedding_layer(sentence_indices)
print("Input Embeddings (X) generated from nn.Embedding:\n", X)
print("Shape of X:", X.shape)

class SimpleSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
            
    def forward(self, x):
        #x is the i/p  matrix of embeddings (sequence_lenght, embedding_dims)
        
        #1. create similarities(dot product)
        scores = torch.matmul(x,x.transpose(-2,-1))
        print("transpose :",x.transpose)
        print("transpose(-2,-1) :",x.transpose(-2,-1))
        print("scores",scores)
        
        #2. normalize score using softmax
        
        attention_weights = F.softmax(scores, dim =-1)
        print(attention_weights)
        print("Sum of weights for each row (should be ~1):", attention_weights.sum(dim=-1))
         
        #3. compute weighted sum of the original embeddings
        contextual_embeddings = torch.matmul(attention_weights,x)
        print(contextual_embeddings)
        
        return contextual_embeddings
simple_attention_model = SimpleSelfAttention();

y_simple = simple_attention_model(X)
print("Shape of Y (Simple Self-Attention):", y_simple.shape)

class SelfAttentionWithParameters(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        self.query_transform = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.key_transform = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.value_transform = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        
        self.scale = math.sqrt(self.embedding_dim)
        
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        query = self.query_transform(x)
        key = self.key_transform(x)
        value = self.value_transform(x)

        print("Query (Q) matrix:\n", query)
        print("Key (K) matrix:\n", key)
        print("Value (V) matrix:\n", value)
        
        scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale
        print("Scores (Q @ K.T / sqrt(d_k)):\n", scores)
        
        attention_weights = F.softmax(scores, dim=-1)
        print("Attention Weights (SoftMax applied):\n", attention_weights)
        print("Sum of weights for each row (should be ~1):", attention_weights.sum(dim=-1))
        
        contextual_embeddings = torch.matmul(attention_weights, value)
        print("Contextual Embeddings (Y_task_specific):\n", contextual_embeddings)

        return contextual_embeddings

    
print("--- Running Self-Attention with Learnable Parameters ---")
embedding_dimension = embedding_dim # Use the defined embedding_dim
attention_with_params_model = SelfAttentionWithParameters(embedding_dimension)
y_task_specific = attention_with_params_model(X) # Pass the X generated from embedding_layer
print("Shape of Y (Self-Attention with Parameters):", y_task_specific.shape)
print("\n" + "=" * 50 + "\n")

print("Conceptual Flow Recap:")
print("1. Input words are converted to integer IDs based on a vocabulary.")
print("2. An `nn.Embedding` layer maps these integer IDs to learnable word embeddings (X).")
print("3. Simple Self-Attention transforms X directly for general context.")
print("4. Self-Attention with Learnable Parameters (Query, Key, Value) transforms X into task-specific context.")

print("\nLearnable Parameters in SelfAttentionWithParameters:")
for name, param in attention_with_params_model.named_parameters():
    if param.requires_grad:
        print(f"  {name}: {param.shape}")

# Also, the embedding_layer itself has learnable parameters
print("\nLearnable Parameters in the nn.Embedding layer:")
for name, param in embedding_layer.named_parameters():
    if param.requires_grad:
        print(f"  {name}: {param.shape}")
