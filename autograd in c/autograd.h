#ifndef AUTOGRAD_H
#define AUTOGRAD_H

//forward-declare so fucntion pointer can reference it 
struct Tensor;

typedef struct Tensor
{
    //data and gradient
    double* data; //ptr for acutal data array(using malloc)
    double* grad;//ptr for gradient array(using malloc)

    //shape info
    int* shape; //ptr to array of integers
    int ndim; //no of dims
    int n_elements; //no. of dims
    int visited;

    //autograd graph info 
    struct Tensor* _prev[2]; //ptrs to childrem
    const char* _op; //operation that created this tensor(for debug)

    void(*_backward)(struct Tensor* self);
} Tensor;


Tensor* tensor_create(double* data, int* shape, int ndim);
void tensor_free(Tensor* t);

Tensor* tensor_add(Tensor* a, Tensor* b);
Tensor* tensor_mul(Tensor* a, Tensor* b);
Tensor* tensor_matmul(Tensor* a, Tensor* b);
void tensor_backward(Tensor* t);

#endif // AUTOGRAD_H
