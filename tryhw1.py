from uwnet import *

def conv_net():
    l = [   make_convolutional_layer(32, 32, 3, 8, 3, 1),
            make_activation_layer(RELU),
            make_maxpool_layer(32, 32, 8, 3, 2),
            make_convolutional_layer(16, 16, 8, 16, 3, 1),
            make_activation_layer(RELU),
            make_maxpool_layer(16, 16, 16, 3, 2),
            make_convolutional_layer(8, 8, 16, 32, 3, 1),
            make_activation_layer(RELU),
            make_maxpool_layer(8, 8, 32, 3, 2),
            make_convolutional_layer(4, 4, 32, 64, 3, 1),
            make_activation_layer(RELU),
            make_maxpool_layer(4, 4, 64, 3, 2),
            make_connected_layer(256, 10),
            make_activation_layer(SOFTMAX)]
    return make_net(l)

def neural_net():
    l = [   make_connected_layer(3*32*32, 256),
            make_activation_layer(RELU),
            make_connected_layer(256, 512),
            make_activation_layer(RELU),
            make_connected_layer(512, 1024),
            make_activation_layer(RELU),
            make_connected_layer(1024, 512),
            make_activation_layer(RELU),
            make_connected_layer(512, 10),
            make_activation_layer(SOFTMAX)]
    return make_net(l)

print("loading data...")
train = load_image_classification_data("cifar/cifar.train", "cifar/cifar.labels")
test  = load_image_classification_data("cifar/cifar.test",  "cifar/cifar.labels")
print("done")
print

print("making model...")
batch = 128
iters = 5000
rate = .01
momentum = .9
decay = .005

# m = conv_net()
m = neural_net()
print("training...")
train_image_classifier(m, train, batch, iters, rate, momentum, decay)
print("done")
print

print("evaluating model...")
print("training accuracy: %f", accuracy_net(m, train))
print("test accuracy:     %f", accuracy_net(m, test))

# How accurate is the fully connected network vs the convnet when they use similar number of operations?

# The convnet uses an estimated 141,885,440 operations (summing up operations from matmul in the convolutional
# and connected layers). To compare, we designed a fully-connected network that uses 161,087,488 operations 
# across five layers. Our results are shown below. We found that the convnet performed better with a 65.62%
# test accuracy compared to 53.16% test accuracy from the fully-connected network.

# CNN
# training accuracy: 0.7069600224494934
# test accuracy:     0.6561999917030334

# Fully-connected NN
# training accuracy: 0.5998200178146362
# test accuracy:     0.5315999984741211

# Why are you seeing these results? Speculate based on the information you've gathered and what you know about DL and ML.
# Your answer:
# With smaller convolutions, convnet is able to leverage spatial locality within the images for more specialized 
# feature extraction, which helps it achieve a greater accuracy than the fully-connected network. Also, because 
# convnet is a sparsely connected network with fewer trainable weights per layer, it requires a lower training 
# time to optimize those weights. The fully-connected network might be able to achieve a comparable accuracy 
# if trained over more iterations or with adjusted parameters, but because its structure has so many more weights 
# to be updated, it will have a much slower ability to learn compared to the convnet.
