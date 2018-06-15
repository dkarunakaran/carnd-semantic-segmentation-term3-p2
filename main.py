import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()
    input_layer = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3 = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4 = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7 = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
    
    return input_layer, keep_prob, layer3, layer4, layer7
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    
    # 1X1 connvolution of the layer 7
    conv_1x1_7th_layer = tf.layers.conv2d(vgg_layer7_out,num_classes, 1,padding = 'same',
                                     kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3),
                                     name='conv_1x1_7th_layer')
    # Upsampling x 4
    upsampling1 = tf.layers.conv2d_transpose(conv_1x1_7th_layer,
                                                num_classes,
                                                4,
                                                strides= (2, 2),
                                                padding= 'same',
                                                kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3),
                                                name='upsampling1')
    # 1X1 convolution of the layer 4
    conv_1x1_4th_layer = tf.layers.conv2d(vgg_layer4_out,
                                     num_classes,
                                     1,
                                     padding = 'same',
                                     kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3),
                                     name='conv_1x1_4th_layer')
    skip1 = tf.add(conv_1x1_4th_layer, upsampling1, name="skip1")

    # Upsampling x 4
    upsampling2 = tf.layers.conv2d_transpose(skip1,
                                    num_classes,
                                    4,
                                    strides= (2, 2),
                                    padding= 'same',
                                    kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3),
                                    name='upsampling2')

    # 1X1 convolution of the layer 3
    conv_1x1_3th_layer = tf.layers.conv2d(vgg_layer3_out,
                                     num_classes,
                                     1,
                                     padding = 'same',
                                     kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3),
                                     name='conv_1x1_3th_layer')
    skip2 = tf.add(conv_1x1_3th_layer, upsampling2, name="skip2")

    # Upsampling x 8.
    upsampling3 = tf.layers.conv2d_transpose(skip2, num_classes, 16,
                                                  strides= (8, 8),
                                                  padding= 'same',
                                                  kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3),
                                                  name='upsampling3')


    return upsampling3
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function

    # Reshape the label same as logits 
    label_reshaped = tf.reshape(correct_label, (-1,num_classes))

    # Converting the 4D tensor to 2D tensor. logits is now a 2D tensor where each row represents a pixel and each column a class
    logits = tf.reshape(nn_last_layer, (-1, num_classes))

    # Name logits Tensor, so that is can be loaded from disk after training
    logits = tf.identity(logits, name='logits')

    # Loss and Optimizer
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label_reshaped))

    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    reg_constant = 1e-3
    loss = cross_entropy_loss + reg_constant * sum(reg_losses)

    train_op = tf.train.AdamOptimizer(learning_rate= learning_rate).minimize(loss)    
    
    return logits, train_op, loss
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function
    print('Checking the Training on a Single Batch...')

    # Initializing the variables
    sess.run(tf.global_variables_initializer())
    
    # Training cycle
    for epoch in range(epochs):
        print("Epoch {}".format(epoch + 1))
        training_loss = 0
        training_samples_length = 0
        for image, label in get_batches_fn(batch_size):
            training_samples_length += len(image)
            _, loss = sess.run([train_op, cross_entropy_loss], feed_dict={
                input_image: image,
                correct_label: label,
                keep_prob: 0.5,
                learning_rate: 0.0001
            })
            training_loss += loss
            print(loss)
        
        # Total training loss
        training_loss /= training_samples_length
        print("********************Total loss***********************")
        print(training_loss)

    pass
tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    save_model_path = './saver/model'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function
        input_layer, keep_prob_tensor, layer3, layer4, layer7 = load_vgg(sess, vgg_path)
        nn_last_layer = layers(layer3, layer4, layer7, num_classes)

        # Placeholders
        correct_label_tensor = tf.placeholder(tf.int32, [None, None, None, num_classes], name='correct_label')
        learning_rate_tensor = tf.placeholder(tf.float32, name='learning_rate')

        logits, train_op, cross_entropy_loss = optimize(nn_last_layer, correct_label_tensor, learning_rate_tensor, num_classes)     
        
        epochs = 50
        batch_size = 5
        saver = tf.train.Saver()
        
        # TODO: Train NN using the train_nn function
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_layer,
             correct_label_tensor, keep_prob_tensor, learning_rate_tensor)

        save_path = saver.save(sess, save_model_path)

        print("Saved to "+save_path)

        # TODO: Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob_tensor, input_layer)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
