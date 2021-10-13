#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "uwnet.h"


// Run an activation layer on input
// layer l: pointer to layer to run
// matrix x: input to layer
// returns: the result of running the layer y = f(x)
matrix forward_activation_layer(layer l, matrix x)
{
    // Saving our input
    // Probably don't change this
    free_matrix(*l.x);
    *l.x = copy_matrix(x);

    ACTIVATION a = l.activation;
    matrix y = copy_matrix(x);

    // TODO: 2.1
    // apply the activation function to matrix y
    // logistic(x) = 1/(1+e^(-x))
    // relu(x)     = x if x > 0 else 0
    // lrelu(x)    = x if x > 0 else .01 * x
    // softmax(x)  = e^{x_i} / sum(e^{x_j}) for all x_j in the same row 
    int row, col;
    for (row = 0; row < y.rows; row++) {
        double sum = 0.0;
        for (col = 0; col < y.cols; col++) {
            double temp = y.data[row * y.cols + col];
            if (a == LOGISTIC) {
                y.data[row * y.cols + col] = 1.0 / (1.0 + exp(-temp));
            } else if (a == RELU) {
                y.data[row * y.cols + col] = (temp > 0.0) ? temp : 0.0;
            } else if (a == LRELU) {
                y.data[row * y.cols + col] = (temp > 0.0) ? temp : 0.01 * temp;
            } else if (a == SOFTMAX) {
                y.data[row * y.cols + col] = exp(-temp);
                sum += y.data[row * y.cols + col];
            }
            if (a == SOFTMAX) {
                for (int k = 0; k < y.cols; k++) {
                    y.data[row * y.cols + k] /= sum;
                }
            }
        }
    }

    return y;
}

// Run an activation layer on input
// layer l: pointer to layer to run
// matrix dy: derivative of loss wrt output, dL/dy
// returns: derivative of loss wrt input, dL/dx
matrix backward_activation_layer(layer l, matrix dy)
{
    matrix x = *l.x;
    matrix dx = copy_matrix(dy);
    ACTIVATION a = l.activation;

    // TODO: 2.2
    // calculate dL/dx = f'(x) * dL/dy
    // assume for this part that f'(x) = 1 for softmax because we will only use
    // it with cross-entropy loss for classification and include it in the loss
    // calculations
    // d/dx logistic(x) = logistic(x) * (1 - logistic(x))
    // d/dx relu(x)     = 1 if x > 0 else 0
    // d/dx lrelu(x)    = 1 if x > 0 else 0.01
    // d/dx softmax(x)  = 1
    int row, col;
    for (row = 0; row < dx.rows; row++) {
        for (col = 0; col < dx.cols; col++) {
            double x_ = x.data[row * dx.cols + col];
            double grad = 1.0;
            if (a == LOGISTIC) {
                grad = x_ * (1.0 - x_);
            } else if (a == RELU) {
                grad = (x_ > 0.0) ? 1.0 : 0.0;
            } else if (a == LRELU) {
                grad = (x_ > 0.0) ? 1.0 : 0.01;
            } else if (a == SOFTMAX) {
                grad = 1.0;
            }
            dx.data[row * dx.cols + col] *= grad;
        }
    }

    return dx;
}

// Update activation layer..... nothing happens tho
// layer l: layer to update
// float rate: SGD learning rate
// float momentum: SGD momentum term
// float decay: l2 normalization term
void update_activation_layer(layer l, float rate, float momentum, float decay){}

layer make_activation_layer(ACTIVATION a)
{
    layer l = {0};
    l.activation = a;
    l.x = calloc(1, sizeof(matrix));
    l.forward = forward_activation_layer;
    l.backward = backward_activation_layer;
    l.update = update_activation_layer;
    return l;
}
