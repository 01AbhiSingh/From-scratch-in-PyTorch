import math
import numpy as np

class Tensor:
    def __init__(self, data, _children=(), _op=''):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data, dtype=np.float64)
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None

    def __repr__(self):
        return f"Tensor(data shape: {self.data.shape})"

    @staticmethod
    def _handle_broadcast_grad(tensor_shape, grad_output):
        if grad_output.shape == tensor_shape:
            return grad_output
        
        n_dims_added = grad_output.ndim - len(tensor_shape)
        if n_dims_added > 0:
            grad_output = np.sum(grad_output, axis=tuple(range(n_dims_added)))
            
        aligned_shape = (1,) * n_dims_added + tensor_shape
        axes_to_sum = [i for i, (s, g) in enumerate(zip(aligned_shape, grad_output.shape)) if s == 1 and g > 1]
        
        if axes_to_sum:
            grad_output = np.sum(grad_output, axis=tuple(axes_to_sum), keepdims=True)
            
        return grad_output.reshape(tensor_shape)

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out_data = self.data + other.data
        out = Tensor(out_data, (self, other), '+')

        def _backward():
            self.grad += self._handle_broadcast_grad(self.data.shape, out.grad)
            other.grad += self._handle_broadcast_grad(other.data.shape, out.grad)
            
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out_data = self.data * other.data
        out = Tensor(out_data, (self, other), '*')
        
        def _backward():
            self.grad += self._handle_broadcast_grad(self.data.shape, out.grad * other.data)
            other.grad += self._handle_broadcast_grad(other.data.shape, out.grad * self.data)
            
        out._backward = _backward
        return out

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out_data = self.data @ other.data
        out = Tensor(out_data, (self, other), '@')
        
        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad
            
        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "pow only supports scalar (int/float) exponents"
        out_data = self.data ** other
        out = Tensor(out_data, (self,), f'**{other}')
        
        def _backward():
            self.grad += self._handle_broadcast_grad(self.data.shape, (other * (self.data ** (other - 1))) * out.grad)
            
        out._backward = _backward
        return out
    
    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * (other ** -1.0)

    def tanh(self):
        out_data = np.tanh(self.data)
        out = Tensor(out_data, (self,), 'tanh')
        
        def _backward():
            self.grad += (1 - out_data**2) * out.grad
            
        out._backward = _backward
        return out

    def relu(self):
        out_data = np.maximum(0, self.data)
        out = Tensor(out_data, (self,), 'ReLU')
        
        def _backward():
            self.grad += (out_data > 0) * out.grad
            
        out._backward = _backward
        return out

    def sum(self, axis=None, keepdims=False):
        out_data = np.sum(self.data, axis=axis, keepdims=keepdims)
        out = Tensor(out_data, (self,), 'sum')
        
        def _backward():
            self.grad += np.ones_like(self.data) * out.grad
            
        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        if axis is None:
            n_elements = self.data.size
        else:
            axes = axis if isinstance(axis, (tuple, list)) else (axis,)
            n_elements = np.prod([self.data.shape[i] for i in axes])
            
        out_data = np.mean(self.data, axis=axis, keepdims=keepdims)
        out = Tensor(out_data, (self,), 'mean')
        
        def _backward():
            self.grad += (np.ones_like(self.data) * out.grad) / n_elements
            
        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        
        self.grad = np.ones_like(self.data)
        
        for node in reversed(topo):
            node._backward()


batch_size = 2
input_features = 3
hidden_neurons = 4
output_classes = 1

np.random.seed(42)

inputs = Tensor(np.random.randn(batch_size, input_features))
w1 = Tensor(np.random.randn(input_features, hidden_neurons))
b1 = Tensor(np.random.randn(1, hidden_neurons))

w2 = Tensor(np.random.randn(hidden_neurons, output_classes))
b2 = Tensor(np.random.randn(1, output_classes))

hid_pre_act = (inputs @ w1) + b1
hid = hid_pre_act.relu()
output = (hid @ w2) + b2

loss = output.mean()

print("Loss:", loss.data)

loss.backward()

print("\n--- Gradients ---")
print("Gradient for W1 (shape:", w1.grad.shape, ")\n", w1.grad)
print("\nGradient for B1 (shape:", b1.grad.shape, ")\n", b1.grad)
print("\nGradient for W2 (shape:", w2.grad.shape, ")\n", w2.grad)
print("\nGradient for B2 (shape:", b2.grad.shape, ")\n", b2.grad)
