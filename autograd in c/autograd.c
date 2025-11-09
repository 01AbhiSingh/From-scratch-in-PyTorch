#include "autograd.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

Tensor* tensor_create(double*data, int* shape, int ndim) {
    //allocate memory for tensor struct
    Tensor* t = (Tensor*)malloc(sizeof(Tensor));
    if (t == NULL) return NULL; //check for allocation failure

    //calc total elemts
    int n_elements = 1;
    for (int i=0;i<ndim,i++;){
        n_elements *= shape[i];
    }
    t->n_elements = n_elements;
    t->ndim = ndim;

    //alloc mem for shape array and copy it 
    t->shape = (int*)malloc(ndim*sizeof(int));
    if(t->shape ==NULL) {
        free(t); 
        return NULL;
    }

    //alloc mem for data and grad arrays
    //use calloc for grad to init it to all zeros

    t->data = (double*)malloc(n_elements*sizeof(double));
    t->grad = (double*)calloc(n_elements, sizeof(double));

    if(t->data == NULL || t->grad == NULL) {
        free(t->shape);
        free(t);
        return NULL;
    }
    
    //copy initial data
    if (data!=NULL){
        memcpy(t->data, data , n_elements * sizeof(double));

    }
    else{
        memset(t->data, 0 ,n_elements * sizeof(double));
    }
    // 6. Initialize graph properties
    t->_prev[0] = NULL;
    t->_prev[1] = NULL;
    t->_op = "";
    t->_backward = NULL; // NULL means leaf node

    return t; 
}

void tensor_free(Tensor* t) {
    if (t == NULL) return;
    free(t->data);
    free(t->grad);
    free(t->shape);
    //free the sturct container at end
    free(t); 
}

static void _backward_add (Tensor* self){
    //self here is the out from add oper
    Tensor* a = self -> _prev[0];
    Tensor* b = self -> _prev[1];

    //assumes no broadcasting
    for(int i=0;i<self->n_elements;i++){
        a->grad[i] += self->grad[i];
        b->grad[i] += self->grad[i];

    }
}

Tensor* tensor_add(Tensor* a, Tensor*b){
    if(a->n_elements != b->n_elements){
        printf("Error: Shapes must match for simple add.\n");
        return NULL;
    }
    //create o/p tensor(data null for now)
    Tensor* out = tensor_create(NULL, a->shape, a->ndim);

    //perform the f/w pass(elt-wise addition)
    for(int i = 0; i < a->n_elements; i++){
        out->data[i] = a->data[i] + b->data[i];
    }

    out->_prev[0] = a;
    out->_prev[1] = b;
    out->_op = "+";
    out->_backward = _backward_add;

    return out;
}
typedef struct {
    Tensor** nodes;
    int size;
    int capacity;
}TopoList;

void list_init(TopoList* list){
    list->capacity = 16;
    list->size = 0;
    list->nodes = (Tensor**)malloc(list->capacity * sizeof(Tensor*));
}

void list_append(TopoList* list, Tensor* node) {
    if(list -> size >= list ->capacity){
        list ->capacity *= 2;
        list -> nodes = (Tensor**)realloc(list->nodes, list ->capacity * sizeof(Tensor*));

    }
    list -> nodes[list -> size++] =node;
}

void list_free(TopoList* list) {
    free(list->nodes);
}

static void build_topo_recursive(Tensor* v, TopoList* topo){
    if(v->visited){
        return;
    }
    v->visited = 1;

    if(v->_prev[0]) build_topo_recursive(v->_prev[0], topo);
    if(v->_prev[1]) build_topo_recursive(v->_prev[1], topo);

    list_append(topo,v);
}

void tensor_backward(Tensor* t){
    TopoList topo;
    list_init(&topo);

    //topological sort
    build_topo_recursive(t,&topo);

    // set the init. grad. (d_out / d_out = 1)
    for(int i = 0;i< t->n_elements; i++) {
        t->grad[i] = 1.0;
    }

    for (int i = topo.size - 1; i >= 0;i--) {
        Tensor* node = topo.nodes[i];

        if(node ->_backward != NULL){
            node->_backward(node);
        }
        node ->visited = 0;
    }
    list_free(&topo);
}
