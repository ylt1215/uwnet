#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <float.h>
#include "uwnet.h"


// Run a maxpool layer on input
// layer l: pointer to layer to run
// matrix in: input to layer
// returns: the result of running the layer
matrix forward_maxpool_layer(layer l, matrix in)
{
    // Saving our input
    // Probably don't change this
    free_matrix(*l.x);
    *l.x = copy_matrix(in);

    int outw = (l.width-1)/l.stride + 1;
    int outh = (l.height-1)/l.stride + 1;
    matrix out = make_matrix(in.rows, outw*outh*l.channels);

    printf("in: %d, %d => %d\n", in.rows, in.cols, in.rows * in.cols);
    printf("out: %d, %d => %d\n", out.rows, out.cols, out.rows * out.cols);
    printf("l.width: %d, l.height: %d, l.c: %d\n", l.width, l.height, l.channels);
    printf("size: %d, stride: %d\n", l.size, l.stride);

    // TODO: 6.1 - iterate over the input and fill in the output with max values
    int i, j;
    int out_idx = 0;
    for(i = 0; i < in.rows; i+=l.stride*l.stride){
        for(j = 0; j < in.cols; j+=l.stride){

            // int a_start = i - (l.size - 1) / 2;
            // int b_start = j - (l.size - 1) / 2;
            float max = in.data[i * in.cols + j]; //[a_start * in.cols + b_start];
            // for(int a = a_start; a < a_start + l.size; ++a) {
            //     for(int b = b_start; b < b_start + l.size; ++b) {
            for(int a = i; a < i + l.size; ++a) {
                for(int b = j; b < j + l.size; ++b) {
                    if (a >= 0 && b >= 0 && a < in.rows && b < in.cols) {
                        float val = in.data[a * in.cols + b];
                        if (val > max) {
                            max = val;
                        }
                    }
                }
            }
            if (out_idx == 0) {
                printf("...setting out.data[out_idx] = %d\n", max);
            }
            out.data[out_idx] = max;
            printf("out_idx: %d\n ", out_idx);
            out_idx++;
        }
    }



    return out;
}

// Run a maxpool layer backward
// layer l: layer to run
// matrix dy: error term for the previous layer
matrix backward_maxpool_layer(layer l, matrix dy)
{
    matrix in    = *l.x;
    matrix dx = make_matrix(dy.rows, l.width*l.height*l.channels);

    int outw = (l.width-1)/l.stride + 1;
    int outh = (l.height-1)/l.stride + 1;
    // TODO: 6.2 - find the max values in the input again and fill in the
    // corresponding delta with the delta from the output. This should be
    // similar to the forward method in structure.



    return dx;
}

// Update maxpool layer
// Leave this blank since maxpool layers have no update
void update_maxpool_layer(layer l, float rate, float momentum, float decay){}

// Make a new maxpool layer
// int w: width of input image
// int h: height of input image
// int c: number of channels
// int size: size of maxpool filter to apply
// int stride: stride of operation
layer make_maxpool_layer(int w, int h, int c, int size, int stride)
{
    layer l = {0};
    l.width = w;
    l.height = h;
    l.channels = c;
    l.size = size;
    l.stride = stride;
    l.x = calloc(1, sizeof(matrix));
    l.forward  = forward_maxpool_layer;
    l.backward = backward_maxpool_layer;
    l.update   = update_maxpool_layer;
    return l;
}

