#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <string.h>
#include "uwnet.h"

// Add bias terms to a matrix
// matrix xw: partially computed output of layer
// matrix b: bias to add in (should only be one row!)
// returns: y = wx + b
matrix forward_convolutional_bias(matrix xw, matrix b)
{
    assert(b.rows == 1);
    assert(xw.cols % b.cols == 0);

    matrix y = copy_matrix(xw);
    int spatial = xw.cols / b.cols;
    int i,j;
    for(i = 0; i < y.rows; ++i){
        for(j = 0; j < y.cols; ++j){
            y.data[i*y.cols + j] += b.data[j/spatial];
        }
    }
    return y;
}

// Calculate dL/db from a dL/dy
// matrix dy: derivative of loss wrt xw+b, dL/d(xw+b)
// returns: derivative of loss wrt b, dL/db
matrix backward_convolutional_bias(matrix dy, int n)
{
    assert(dy.cols % n == 0);
    matrix db = make_matrix(1, n);
    int spatial = dy.cols / n;
    int i,j;
    for(i = 0; i < dy.rows; ++i){
        for(j = 0; j < dy.cols; ++j){
            db.data[j/spatial] += dy.data[i*dy.cols + j];
        }
    }
    return db;
}

// helper function for im2col, similar to get_pixel in image.c
float get_impixel(image im, int x, int y, int c) {
    if(x >= im.w) return 0;
    if(y >= im.h) return 0;
    if(x < 0) return 0;
    if(y < 0) return 0;

    if(x >= im.w) x = im.w - 1;
    if(y >= im.h) y = im.h - 1;
    if(x < 0) x = 0;
    if(y < 0) y = 0;
    assert(c >= 0);
    assert(c < im.c);
    return im.data[x + im.w*(y + im.h*c)];
}

// Make a column matrix out of an image
// image im: image to process
// int size: kernel size for convolution operation
// int stride: stride for convolution
// returns: column matrix
matrix im2col(image im, int size, int stride)
{
    int c, imx, imy;
    int outw = (im.w-1)/stride + 1;
    int outh = (im.h-1)/stride + 1;
    int rows = im.c*size*size;
    int cols = outw * outh;
    matrix col = make_matrix(rows, cols);

    // TODO: 5.1
    // Fill in the column matrix with patches from the image
    int col_row, col_col;
    for (c = 0; c < im.c; c++) {
        col_col = 0;
        for (imy = 0; imy < im.h; imy += stride) {
            for (imx = 0; imx < im.w; imx += stride) {
                // for size * size window
                int y, x;
                col_row = 0;
                for (int i = 0; i < size; i++) {
                    
                    for (int j = 0; j < size; j++) {
                        y = imy - (size-1)/2 + i;
                        x = imx - (size-1)/2 + j;
                        col.data[c*size*size*cols + col_row*cols + col_col] = get_impixel(im, x, y, c);
                        col_row++;
                    }
                }
                col_col++;
            }
        }
    }

    return col;
}

// The reverse of im2col, add elements back into image
// matrix col: column matrix to put back into image
// int size: kernel size
// int stride: convolution stride
// image im: image to add elements back into
image col2im(int width, int height, int channels, matrix col, int size, int stride)
{
    int c, imx, imy;

    image im = make_image(width, height, channels);

    // TODO: 5.2
    // Add values into image im from the column matrix
    int col_row, col_col;
    for (c = 0; c < im.c; c++) {
        col_col = 0;
        for (imy = 0; imy < im.h; imy += stride) {
            for (imx = 0; imx < im.w; imx += stride) {
                // for size * size window
                int y, x;
                col_row = 0;
                for (int i = 0; i < size; i++) {
                    
                    for (int j = 0; j < size; j++) {
                        y = imy - (size-1)/2 + i;
                        x = imx - (size-1)/2 + j;
                        float val = get_impixel(im, x, y, c);
                        set_pixel(im, x, y, c, val + col.data[c*size*size*col.cols + col_row*col.cols + col_col]);
                        col_row++;
                    }
                }
                col_col++;
            }
        }
    }
    return im;
}

// Run a convolutional layer on input
// layer l: pointer to layer to run
// matrix in: input to layer
// returns: the result of running the layer
matrix forward_convolutional_layer(layer l, matrix in)
{
    assert(in.cols == l.width*l.height*l.channels);
    // Saving our input
    // Probably don't change this
    free_matrix(*l.x);
    *l.x = copy_matrix(in);

    int i, j;
    int outw = (l.width-1)/l.stride + 1;
    int outh = (l.height-1)/l.stride + 1;
    matrix out = make_matrix(in.rows, outw*outh*l.filters);
    for(i = 0; i < in.rows; ++i){
        image example = float_to_image(in.data + i*in.cols, l.width, l.height, l.channels);
        matrix x = im2col(example, l.size, l.stride);
        matrix wx = matmul(l.w, x);
        for(j = 0; j < wx.rows*wx.cols; ++j){
            out.data[i*out.cols + j] = wx.data[j];
        }
        free_matrix(x);
        free_matrix(wx);
    }
    matrix y = forward_convolutional_bias(out, l.b);
    free_matrix(out);

    return y;
}

// Run a convolutional layer backward
// layer l: layer to run
// matrix dy: dL/dy for this layer
// returns: dL/dx for this layer
matrix backward_convolutional_layer(layer l, matrix dy)
{
    matrix in = *l.x;
    assert(in.cols == l.width*l.height*l.channels);

    int i;
    int outw = (l.width-1)/l.stride + 1;
    int outh = (l.height-1)/l.stride + 1;


    matrix db = backward_convolutional_bias(dy, l.db.cols);
    axpy_matrix(1, db, l.db);
    free_matrix(db);


    matrix dx = make_matrix(dy.rows, l.width*l.height*l.channels);
    matrix wt = transpose_matrix(l.w);

    for(i = 0; i < in.rows; ++i){
        image example = float_to_image(in.data + i*in.cols, l.width, l.height, l.channels);

        dy.rows = l.filters;
        dy.cols = outw*outh;

        matrix x = im2col(example, l.size, l.stride);
        matrix xt = transpose_matrix(x);
        matrix dw = matmul(dy, xt);
        axpy_matrix(1, dw, l.dw);

        matrix col = matmul(wt, dy);
        image dxi = col2im(l.width, l.height, l.channels, col, l.size, l.stride);
        memcpy(dx.data + i*dx.cols, dxi.data, dx.cols * sizeof(float));
        free_matrix(col);

        free_matrix(x);
        free_matrix(xt);
        free_matrix(dw);
        free_image(dxi);

        dy.data = dy.data + dy.rows*dy.cols;
    }
    free_matrix(wt);
    return dx;

}

// Update convolutional layer
// layer l: layer to update
// float rate: learning rate
// float momentum: momentum term
// float decay: l2 regularization term
void update_convolutional_layer(layer l, float rate, float momentum, float decay)
{
    // TODO: 5.3
    axpy_matrix(decay, l.w, l.dw);
    axpy_matrix(-rate, l.dw, l.w);
    scal_matrix(momentum, l.dw);
    axpy_matrix(-rate, l.db, l.b);
    scal_matrix(momentum, l.db);
}

// Make a new convolutional layer
// int w: width of input image
// int h: height of input image
// int c: number of channels
// int size: size of convolutional filter to apply
// int stride: stride of operation
layer make_convolutional_layer(int w, int h, int c, int filters, int size, int stride)
{
    layer l = {0};
    l.width = w;
    l.height = h;
    l.channels = c;
    l.filters = filters;
    l.size = size;
    l.stride = stride;
    l.w  = random_matrix(filters, size*size*c, sqrtf(2.f/(size*size*c)));
    l.dw = make_matrix(filters, size*size*c);
    l.b  = make_matrix(1, filters);
    l.db = make_matrix(1, filters);
    l.x = calloc(1, sizeof(matrix));
    l.forward  = forward_convolutional_layer;
    l.backward = backward_convolutional_layer;
    l.update   = update_convolutional_layer;
    return l;
}

