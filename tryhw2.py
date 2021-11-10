from uwnet import *
def conv_net():
    l = [   make_convolutional_layer(32, 32, 3, 8, 3, 2),
            # make_batchnorm_layer(32),
            make_activation_layer(RELU),
            make_maxpool_layer(16, 16, 8, 3, 2),
            make_convolutional_layer(8, 8, 8, 16, 3, 1),
            # make_batchnorm_layer(8),
            make_activation_layer(RELU),
            make_maxpool_layer(8, 8, 16, 3, 2),
            make_convolutional_layer(4, 4, 16, 32, 3, 1),
            # make_batchnorm_layer(4),
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
iters = 1000
rate = .1
momentum = .9
decay = .005

m = conv_net()
print("training...")
train_image_classifier(m, train, batch, iters, rate, momentum, decay)
rate = 0.01
train_image_classifier(m, train, batch, iters, rate, momentum, decay)
rate = 0.001
train_image_classifier(m, train, batch, iters, rate, momentum, decay)
rate = 0.0001
train_image_classifier(m, train, batch, iters, rate, momentum, decay)
print("done")
print

print("evaluating model...")
print("training accuracy: %f", accuracy_net(m, train))
print("test accuracy:     %f", accuracy_net(m, test))

# 7.6 Question: What do you notice about training the convnet with/without batch normalization? How does it affect convergence? How does it affect what magnitude of learning rate you can use? Write down any observations from your experiments:

# Training the convnet with batch normalization increased the resulting test accuracy from 39.8% to 52.8%,
# a significant improvement. Increasing the learning rate with annealing (steps: 0.1, 0.01, 0.001, 0.0001)
# further raised the test accuracy by enabling better convergence; giving the network more time to train
# by doubling the iters (from 500 to 1000) increased the test accuracy to 58%.

# The fluctuation range of the loss also decreased along with each experiment, as shown below.
# Toward the end of the training iterations, the original convnet had losses between 1.5 and 1.8. However,
# each update to the model lowered this range, with the final convnet (with batch normalization, annealing
# the learning rate, and using 1000 iters) reaching a loss between 0.9 and 1.2 by the end of training. This
# suggests that an increasingly smaller learning rate with normalized weights helps the model more closely 
# converge around the minimum loss. Running the same model again but without the batch normalization resulted
# in a clearly lower test accuracy (~46%) which demonstrates that adjusting the magnitude of the learning 
# rate alone doesn't improve performance nearly as much as it does when the training is stabilized by  
# normalizing the outputs at each layer before activation.

# convnet as usual:                    training accuracy: 0.3983800113201141, test accuracy: 0.398499995470047
# convnet with batchnorm:              training accuracy: 0.5360000133514404, test accuracy: 0.5288000106811523
# convnet with batchnorm + annealing:  training accuracy: 0.5566800236701965, test accuracy: 0.5485000014305115
# same as above but with 1000 iters:   training accuracy: 0.5982999801635742, test accuracy: 0.5803999900817871
# same as above but WITHOUT batchnorm: training accuracy: 0.4814200103282928, test accuracy: 0.4668999910354614

# convnet as usual:                    loss was 1.5-1.8 towards the end
# convnet with batchnorm:              loss was 1.1-1.4 towards the end
# convnet with batchnorm + annealing:  loss was 1.0-1.3 towards the end
# same as above but with 1000 iters:   loss was 0.9-1.2 towards the end
# same as above but WITHOUT batchnorm: loss was 1.1-1.4 towards the end
