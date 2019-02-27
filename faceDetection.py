# -*- coding: utf-8 -*-
"""
Created on Tue Jan  1 18:36:45 2019

@author: Arthur

Implementation of a simplified version of "A convolutional neural network
cascade for face detection, Li H. et al.". In particular I haven't implemented
the recalibration step as of now.
"""
import numpy as np
import tensorflow as tf
import tensorflow.contrib.predictor as predictor
from tensorflow.python.estimator.export.export_output import *
import cv2
from sklearn.datasets import fetch_lfw_people
from matplotlib import pyplot as plt
import os
from onlinePrediction import OnlinePredictor
import load_IMDB_WIKI
from sklearn.model_selection import train_test_split

class LiFaceRecognizer(object):
    def __init__(self, w, h, window_nb_points = 12, window_min_size = 48,
                 window_stride = 4,
                 blackwhite = True):
        """Args:
            window_nb_points    size of the window in terms of points
            window_min_size     minimum size of the scaled windows in pixels
            window_stride       number of points we shift the detection window.
            blackwhite          bool, True if black and white images False if
                                RGB.
        """
        params_12 = {'win_size' : window_nb_points, 'learning_rate' : 1e-5}
        params_24 = {'win_size' : 2 * window_nb_points, 'learning_rate' : 1e-5}
        params_48 = {'win_size' : 4 * window_nb_points,'learning_rate' : 1e-1}

        self.net12 = tf.estimator.Estimator(self._model_fn_net12_24,
                                            model_dir = 'tfModelDir12',
                                            params = params_12)
        self.net24 = tf.estimator.Estimator(self._model_fn_net12_24,
                                            model_dir = 'tfModelDir24',
                                            params = params_24)
        self.net48 = tf.estimator.Estimator(self._model_fn_net48,
                                            model_dir = 'tfModelDir48',
                                            params = params_48)
        self.training_data_set_negative = None
        self.training_data_set_positive = None
        self.window_min_size = window_min_size
        self.window_stride = window_stride
        self.window_nb_points = window_nb_points
        #Only one channel if black and white input images, otherwise 3 
        #channels for RGB.
        self.nb_channels = 1 if blackwhite else 3
        self.batch_size = 200
        #The function theresholds for net12 and net24
        self.t_net12 = 0.03610440343618393
        self.t_net24 =  0.3929968476295471
    
    def set_training_data_set_positive(self, data):
        """Sets the training data for positive labels (faces). Must be an array
        of images of the same size.
        Args:
            data            4-d numpy array, where the first dimension 
                            corresponds to different images, and the last 
                            dimension corresponds to different channels.
        """
        assert(data.ndim == 4)
        assert(data.shape[3] == self.nb_channels)
#        data = (data - np.mean(data, 0)) / np.std(data, 0)
        self.training_data_set_positive = data
    
    def set_training_data_set_negative(self, data):
        """Sets the training data for negative labels. data here is a list of
        images, of possibly various sizes as the training data is obtained by
        subsampling random subimages from this data.
        Args:
            data            a list of 3-d numpy arrays, where the last 
                            dimension corresponds to the different channels.
        """
        assert(data[0].ndim == 3)
        assert(data[0].shape[2] == self.nb_channels)
        extended_data = self._random_sub_images(data, self.window_min_size,
                                       self.window_min_size, 25)
#        data = (data - np.mean(data, 0)) / np.std(data, 0)
        self.training_data_set_negative = extended_data
    
    def _resize_data(self, data, w, h):
        """Resizes all images in the array data.
        Args:
            data            4-D numpy array with channel as last dimension
            w               integer, the first dimension of the resized images
            h               integer, the second dimension of the resized images
        Returns:
            numpy array with shape (N, w, h, self.nb_channels) where N is the
            size of the first dimension of data.
        """
        nb_images = data.shape[0]
        data_new = np.zeros((nb_images, w, h, self.nb_channels))
        for i in range(nb_images):
            resized = cv2.resize(data[i,:,:,:], (w,h))
            resized = np.reshape(resized, (w, h, self.nb_channels))
            data_new[i,:,:,:] = resized
        return data_new
    
    def _ensure_dimensions(self, image):
        """Makes sure the array is three dimensional, even for black and white 
        images, where the last dimension, the channel, has size one. This is
        for example necessary after calling cv2.resize on a black and white 
        image which changes the shape of the array to dimension 2"""
        return np.reshape(image, (image.shape[0], image.shape[1],
                                  self.nb_channels))
    
    def _random_sub_images(self, data, w, h, nb_per_images):
        """Returns randomly sampled subimages based on the images contained in
        data. For each image we sample nb_per_images sub-images, with
        w x h pixels. We also allow for random scales."""
        nb_images = len(data)
        data_new = np.zeros((nb_images * nb_per_images, w, h,
                             self.nb_channels))
        for i in range(nb_images * nb_per_images):
            #Image we sample from
            image_from = int(i // nb_per_images)
            image_w = data[image_from].shape[0]
            image_h = data[image_from].shape[1]
            #Random size
            random_size = np.random.randint(self.window_min_size,
                                            min(image_w, image_h))
            rdm_corner_0 = np.random.randint(0, image_w - random_size)
            rdm_corner_1 = np.random.randint(0, image_h - random_size)
            rdm_sub = data[image_from][rdm_corner_0 :
                rdm_corner_0 + random_size, rdm_corner_1 : 
                    rdm_corner_1 + random_size, :]
            rdm_sub = cv2.resize(rdm_sub, (w, h))
            #This is necessary for black and white images.
            rdm_sub = self._ensure_dimensions(rdm_sub)
            data_new[i,:,:,:] = rdm_sub
        return data_new
    
    def _input_fn_12_24(self, nb_points_per_window, negative, positive = None,
                        shuffle = True):
        """Input function for the training of net12 and net24NN"""
        if positive is not None:
            resized_positive = self._resize_data(positive,
                                             nb_points_per_window,
                                             nb_points_per_window)
            nb_images = positive.shape[0]
            labels = np.ones((nb_images, 1), dtype = np.int32)
            dataset_positive = tf.data.Dataset.from_tensor_slices(
                ({'x' : resized_positive}, labels))
        resized_negative = self._resize_data(negative,
                                             nb_points_per_window,
                                             nb_points_per_window)
        nb_images = negative.shape[0]
        labels = np.zeros((nb_images, 1), dtype = np.int32)
        dataset_negative = tf.data.Dataset.from_tensor_slices(
                ({'x' : resized_negative}, labels))
        if positive is not None:
            nb_pos = positive.shape[0]
            nb_neg = negative.shape[0]
            if nb_pos > nb_neg:
                dataset_negative = dataset_negative.repeat(int(nb_pos / nb_neg))
            else:
                dataset_positive = dataset_positive.repeat(int(nb_neg / nb_pos))
            dataset = dataset_positive.concatenate(dataset_negative)
        else:
            dataset = dataset_negative
        if shuffle:
            dataset = dataset.shuffle(100000)
            dataset = dataset.repeat()
        dataset = dataset.batch(self.batch_size)
        return dataset
    
    def _find_threshold(self, estimator, positive_data, recall = 0.99):
        """Returns the value of the threshold that leads to a given recall 
        rate for the passed data.
        Args:
            estimator       a tf estimator
            positive_data   a numpy array, that contains the samples for which 
                            we require a certain recall rate. 
            recall          numeric, between 0 and 1, the value of the recall 
                            rate
        Returns:
            float between 0 and 1 corresponding to the calculated threshold.
        """
        #First we need to compute the scores for the data
        def input_fun():
            dataset = tf.data.Dataset.from_tensor_slices(
                    {'x' : positive_data})
            dataset = dataset.batch(self.batch_size)
            return dataset
        nb_samples = positive_data.shape[0]
        scores = np.zeros(nb_samples)
        pred_generator = estimator.predict(input_fun)
        for (k, prediction) in enumerate(pred_generator):
            scores[k] = prediction['probas'][1]
        sorted_scores = np.sort(scores)
        index = int((1-recall) * nb_samples)
        print('MAX PREDICTED : ' + str(np.max(scores)))
        return sorted_scores[index]
    
    def train(self, steps12 = 1, steps24 = 1, steps48 = 1):
        assert(self.training_data_set_positive is not None)
        assert(self.training_data_set_negative is not None)
        #We split the positive dataset in two. One bit will be used to train 
        #the nets in the cascade, the second bit will be used to determine 
        #their threshold.
        positive_train, positive_test = train_test_split(
                self.training_data_set_positive, train_size = 0.8)
        negative_train, negative_test = train_test_split(
                self.training_data_set_negative, train_size = 0.25)
        
        #TRAINING OF NET12-----------------------------------------------------
        #Input function for the training of net12
        input_fn = lambda : self._input_fn_12_24(12, negative_train,
                                                 positive_train)
        print('Training of net12...')
        self.net12.train(input_fn, steps = steps12)
        print('Training of net12 completed...')
        #Now we determine the threshold value that provides a 99% recall for 
        #net12
        resized_positive = self._resize_data(positive_test,
                                             12, 12)
        self.t_net12 = self._find_threshold(self.net12,
                             resized_positive, 0.99)
        print('Threshold for net12 99% recall: ' + str(self.t_net12))
        #The negative training set for the second net is obtained by selecting
        #the false-positive from the negative test data.
        input_fn_negative = lambda : self._input_fn_12_24(12,
                                 negative_test,
                                 shuffle = False)
        net12_predictions = self.net12.predict(input_fn_negative)
        false_positive = np.zeros(negative_test.shape[0], dtype = np.bool)
        for (k,pred) in enumerate(net12_predictions):
            false_positive[k] = pred['probas'][1] >= self.t_net12
        negative_filtered = negative_test[false_positive,:,:,:]
        percentage = negative_filtered.shape[0] / negative_test.shape[0]
        print('Percentge of false positive: {}'.format(percentage))
        print('Number of false positive: {}'.format(negative_filtered.shape[0]))
#        
        #TRAINING of NET24-----------------------------------------------------
        if steps24 == 0:
            return
        #We split the filtered negative data in two, one bit used for training
        #and the other to compute the threshold.
        negative_train, negative_test = train_test_split(negative_filtered,
                                                         train_size = 0.25)
        input_fn = lambda : self._input_fn_12_24(24, negative_train,
                                                 positive_train)
        #We train net24
        print('Training of net24...')
        self.net24.train(input_fn, steps = steps24)
        print('Training of net24 completed.')
        resized_positive = self._resize_data(positive_test,
                                             24, 24)
        #We determine the threshold value that gives 97% recall for net24
        self.t_net24 = self._find_threshold(self.net24, resized_positive,
                                            0.97)
        print('Threshold for net24 97% recall: ' + str(self.t_net24))
        #We get the false positives as negative data for the next net
        input_fn_negative = lambda : self._input_fn_12_24(24, negative_test,
                                 shuffle = False)
        net24_predictions = self.net24.predict(input_fn_negative)
        false_positive = np.zeros(negative_test.shape[0],
                                  dtype = np.bool)
        for (k,pred) in enumerate(net24_predictions):
            false_positive[k] = pred['probas'][1] > self.t_net24
        negative_filtered = negative_test[false_positive, :, :, :]
        percentage = negative_filtered.shape[0] / negative_test.shape[0]
        print('Percentge of false positive: {}'.format(percentage))
        print('Number of false positive: {}'.format(negative_filtered.shape[0]))
        
        #TRAINING of NET48-----------------------------------------------------
        if steps48 == 0:
            return
        negative_train = negative_filtered
        print('Training net48')
        input_fn = lambda : self._input_fn_12_24(48, negative_train,
                                                 positive_train)
        self.net48.train(input_fn, steps = steps48)
    
    def initialize_predictors(self):
        """Initializes predictors based on the NN of the model. This allows to 
        achieve online prediction without loading an estimator repeatedly,
        which would largely slow down the process."""
        def serving_input_fn():
            x = tf.placeholder(dtype=tf.float32, name='x')
            inputs = {'x': x }
            return tf.estimator.export.ServingInputReceiver(inputs, inputs)
        self.pred12 = predictor.from_estimator(self.net12, serving_input_fn,
                                              graph=tf.Graph())
        self.pred24 = predictor.from_estimator(self.net24, serving_input_fn,
                                              graph = tf.Graph())
        self.pred48 = predictor.from_estimator(self.net48, serving_input_fn,
                                              graph = tf.Graph())
    
    def _get_pyramid(self, image, size):
        """Returns the pyramid image, where we resize according to size"""
        pyramid = {1: image}
        p = 1.2
        factor = 1
        while True:
            current_img = pyramid[factor]
            current_w, current_h = current_img.shape[:2]
            if min(current_img.shape[:2]) >= 48 * 2:
#                pyramid[factor * p] = cv2.pyrDown(current_img)
                pyramid[factor * p] = cv2.resize(current_img, None,
                       fx = 1 / p, fy = 1 / p)
                factor *= p
            else:
                break
        for factor in pyramid:
            temp = pyramid[factor]
            temp = cv2.resize(temp, None,
                       fx = size / self.window_min_size,
                       fy = size / self.window_min_size)
            pyramid[factor] = np.reshape(temp, (temp.shape[0],
                       temp.shape[1], self.nb_channels))
        return pyramid
    
    def _get_windows(self, image, corners, size):
        windows = np.zeros((corners.shape[0], size, size, self.nb_channels))
        for i, corner in enumerate(corners):
            x, y = corner
            windows[i,:,:,:] = np.reshape(image[x : x + size, y : y + size],
                   (size, size, self.nb_channels))
        return windows
    
    def _filter(self, windows, predictor, threshold):
        predictions = predictor({'x' : windows})
        probas = predictions['output'][:, 1]
        return probas >= threshold
    
    def identify_faces(self, image):
        """Returns the provided image with identified faces shown as black
        boxes.
        """
        #TODO add the deletion of overlapping windows.
        result_image = np.copy(image)
        #We get the pyramid of images, which have been resized by 
        #12/self.window_min_size
        pyramid12 = self._get_pyramid(image, 12)
        pyramid24 = self._get_pyramid(image, 24)
        pyramid48 = self._get_pyramid(image, 48)
        for factor in pyramid12:
            level_img_12 = pyramid12[factor]
            level_img_24 = pyramid24[factor]
            level_img_48 = pyramid48[factor]
            #We get all the corners
            corners = self._get_all_corners(level_img_12, 12)
            #We filter through net12
            sub_images = self._get_windows(level_img_12, corners, 12)
            corners = corners[self._filter(sub_images, self.pred12, 
                                           self.t_net12)]
            #We filter through net24
            if len(corners) == 0:
                continue
            sub_images = self._get_windows(level_img_24, corners * 2, 24)
            corners = corners[self._filter(sub_images, self.pred24, 
                                           self.t_net24)]
            #We filter through net48
            if len(corners) == 0:
                continue
            sub_images = self._get_windows(level_img_48, corners * 4, 48)
            corners = corners[self._filter(sub_images, self.pred48, 0.5), :]
            #Add to the output image the selected rectangles.
            corners = corners * factor * self.window_min_size / 12
            corners = corners.astype(np.int64)
            current_size = int(self.window_min_size * factor)
            self._rectangles_on_image(result_image, corners, current_size, 
                                      (0,0,0), 3)
        return result_image
    
    def _get_all_corners(self, image, size):
        """Returns an array of all the corners locations that are admissible
        given the passed size of rectangles."""
        w = image.shape[0]
        h = image.shape[1]
        xs = np.array(range(0, w - size, self.window_stride))
        ys = np.array(range(0, h - size, self.window_stride))
        x, y = np.meshgrid(xs, ys)
        x, y = np.ravel(x), np.ravel(y)
        corners = np.vstack((x,y))
        corners = np.transpose(corners)
        return corners
    
    def _rectangles_on_image(self, image, corners, size, colour, 
                             line_width):
        """Returns the passed image with rectangles shown.
        Args:
            image: 3-d array, the image to transform
            corners: a list of two-element tuples defining the positions of
            top-left corners.
            sizes: a list of two-element tuples defining the sizes of the
            rectangles
            colours: a list of three-element tuples defining the RGB colour of
            the rectangles
            line-width: int
        """
        for corner in corners:
            x, y = corner
            w, h = size, size
            r, g, b = colour
            assert(x + w < image.shape[0])
            assert(y + h < image.shape[1])
            cv2.rectangle(image, (y, x), (y+h, x+w), colour, line_width)
    
    def _model_fn_net12_24(self, features, labels, mode, params):
        """Defines the shallow nets used as early-step classifiers."""
        win_size = params['win_size']
        learning_rate = params['learning_rate']
        
        features = tf.cast(features['x'], tf.float32)
        #The input image is flat, we reshape it before applying convolution
        input_layer = tf.reshape(features, (-1, win_size, win_size,
                                            self.nb_channels))
        
        #A first convolutional layer
        conv_layer1 = tf.layers.conv2d(input_layer, 16, (3,3), strides = 1, 
                                   padding = 'same', 
                                   data_format = 'channels_last',
                                   activation = tf.nn.relu)
        #Followed by a max-pool layer. This divides by 2 each dimension of the
        #image.
        pool_layer = tf.layers.max_pooling2d(conv_layer1, (3,3), strides = 2,
                                               padding = 'same')
        
        #Followed by a fully-connected layer after falttening
        pool_layer_flat = tf.reshape(pool_layer, 
                                     (-1, 16 * int(win_size ** 2/ 4)))
        full_layer1 = tf.layers.dense(pool_layer_flat, units = 16,
                                  activation = tf.nn.relu)
        
        #Followed by a final fully connected layer
        output_layer = tf.layers.dense(full_layer1, units = 2,
                                  activation = tf.nn.softmax)
        #Predicted classes
        predicted_classes = tf.argmax(output_layer, 1)
        
        #Depending on the mode we return different specs
        if mode == tf.estimator.ModeKeys.PREDICT:
            #Prediction mode
            predictions = {'classes' : predicted_classes,
                           'probas' : output_layer}
            return tf.estimator.EstimatorSpec(mode = mode,
                                              predictions = predictions,
                                              export_outputs = 
                                              {'probability': PredictOutput(output_layer)})
        
        #If we train or evaluate, we need a loss and the labels. The loss needs
        #to account for possibly unbalanced classes
        labels = tf.cast(labels, tf.int32)
#        weight1 = tf.matmul(labels, tf.ones_like(labels), transpose_a=True)
#        weight0 = tf.matmul(1-labels, tf.ones_like(labels), transpose_a=True)
#        weights = 
        loss = tf.losses.sparse_softmax_cross_entropy(labels, output_layer)
        
        if mode == tf.estimator.ModeKeys.TRAIN:
            #Training mode
            #TODO make the training rate a parameter
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, 
                                      global_step = tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode = mode, loss = loss,
                                          train_op = train_op)
        
        if mode == tf.estimator.ModeKeys.EVAL:
            eval_metric_op = {'accuracy' : tf.metrics.accuracy(labels, 
                                                           predictions['classes'])}
            return tf.estimator.EstimatorSpec(mode = mode,
                                          eval_metric_ops = eval_metric_op,
                                          loss = loss)
        
    
    def _model_fn_net48(self, features, labels, mode, params):
        """Defines the deeper net used for windows that have gone through the
        first steps of the cascade"""
        win_size = params['win_size']
        learning_rate = params['learning_rate']
        
        features = tf.cast(features['x'], tf.float32)
        input_layer = tf.reshape(features, (-1, win_size, win_size,
                                            self.nb_channels))
        
        #A first convolutional layer
        conv_layer1 = tf.layers.conv2d(input_layer, 64, (5, 5), strides = 1, 
                                       padding = 'same', 
                                       data_format = 'channels_last',
                                       activation = tf.nn.relu)
        #Followed by a max-pool layer. This divides by 2 each dimension of the
        #image.
        pool_layer = tf.layers.max_pooling2d(conv_layer1, (3,3), strides = 2,
                                               padding = 'same')

        normalization_layer = tf.layers.batch_normalization(pool_layer)
        
        #Followed by a second convolutional layer
        conv_layer2 = tf.layers.conv2d(normalization_layer, 64, (5, 5), 
                                       strides = 1, 
                                       padding = 'same', 
                                       data_format = 'channels_last',
                                       activation = tf.nn.relu)
        
        #TODO add a normalization layer
        normalization_layer2 = tf.layers.batch_normalization(conv_layer2)
        
        #Followed by a max-pool layer. This divides by 2 each dimension of the
        #image.
        pool_layer2 = tf.layers.max_pooling2d(normalization_layer2, (3,3),
                                              strides = 2,
                                              padding = 'same')
        
        #Followed by a fully-connected layer after falttening
        pool_layer_flat = tf.reshape(pool_layer2, 
                                     (-1, 64 * int(win_size ** 2/ 16)))
        full_layer1 = tf.layers.dense(pool_layer_flat, units = 256,
                                  activation = tf.nn.relu)
        
        #Followed by the final layer
        output_layer = tf.layers.dense(full_layer1, units = 2,
                                       activation = tf.nn.softmax)
        
        #Predicted classes
        predicted_classes = tf.argmax(output_layer, 1)
        
        #Depending on the mode we return different specs
        if mode == tf.estimator.ModeKeys.PREDICT:
            #Prediction mode
            predictions = {'classes' : predicted_classes,
                           'probas' : output_layer}
            return tf.estimator.EstimatorSpec(mode = mode,
                                              predictions = predictions,
                                              export_outputs = 
                                              {'probability': PredictOutput(output_layer)})
        
        #If we train or evaluate, we need a loss
        loss = tf.losses.sparse_softmax_cross_entropy(labels, output_layer)
        labels = tf.cast(labels, tf.int32)
        
        if mode == tf.estimator.ModeKeys.TRAIN:
            #Training mode
            #TODO make the training rate a parameter
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, 
                                      global_step = tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode = mode, loss = loss,
                                          train_op = train_op)
        
        if mode == tf.estimator.ModeKeys.EVAL:
            eval_metric_op = {'accuracy' : tf.metrics.accuracy(labels, 
                                                           predictions['classes'])}
            return tf.estimator.EstimatorSpec(mode = mode,
                                          eval_metric_ops = eval_metric_op,
                                          loss = loss)
            
    def _model_fn_cal_12_24(self, features, labels, mode, params):
        """Defines the calibration net after the 12x12 detection net, or the
        calibration net after the 24x24 detection net. The function determines
        between the two depending on the value of the parameter x_calibration
        passed through the dictionary params. This value should be 12 or 24."""
        learning_rate = params['learning_rate']
        win_size = params['win_size']
        x_calibration = params['x_calibration']
        
        features = tf.cast(features['x'], tf.float32)
        input_layer = tf.reshape(features, (-1, win_size, win_size,
                                            self.nb_channels))
        
        if x_calibration == 12:
            #A first convolutional layer (16 3x3 filters stride 1)
            conv_layer1 = tf.layers.conv2d(input_layer, 16, (3, 3), strides = 1, 
                                           padding = 'same', 
                                           data_format = 'channels_last',
                                           activation = tf.nn.relu)
        elif x_calibration == 24:
            conv_layer1 = tf.layers.conv2d(input_layer, 32, (5,5), strides = 1,
                                           padding = 'same',
                                           data_format = 'channels_last',
                                           activation = tf.nn.relu)
        #Followed by a max-pool layer. This divides by 2 each dimension of the
        #image.
        pool_layer = tf.layers.max_pooling2d(conv_layer1, (3,3), strides = 2,
                                               padding = 'same')
        #Followed by a fully-connected layer after falttening
        if x_calibration == 12:
            pool_layer_flat = tf.reshape(pool_layer, 
                                         (-1, 16 * int(win_size ** 2/ 4)))
            full_layer1 = tf.layers.dense(pool_layer_flat, units = 128,
                                      activation = tf.nn.relu)
        elif x_calibration == 24:
            pool_layer_flat = tf.reshape(pool_layer, 
                                         (-1, 32 * int(win_size ** 2/ 4)))
            full_layer1 = tf.layers.dense(pool_layer_flat, units = 64,
                                      activation = tf.nn.relu)
        #And finally the output layer
        output_layer = tf.layers.dense(full_layer1, units = 45, 
                                       activation = tf.nn.softmax)
        
         #Depending on the mode we return different specs
        if mode == tf.estimator.ModeKeys.PREDICT:
            #Prediction mode
            predictions = {'probas' : output_layer}
            return tf.estimator.EstimatorSpec(mode = mode,
                                              predictions = predictions,
                                              export_outputs = 
                                              {'probability': PredictOutput(output_layer)})
        
        #If we train or evaluate, we need a loss
        loss = tf.losses.sparse_softmax_cross_entropy(labels, output_layer)
        labels = tf.cast(labels, tf.int32)
        
        if mode == tf.estimator.ModeKeys.TRAIN:
            #Training mode
            #TODO make the training rate a parameter
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, 
                                      global_step = tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode = mode, loss = loss,
                                          train_op = train_op)
        
        if mode == tf.estimator.ModeKeys.EVAL:
            eval_metric_op = {'accuracy' : tf.metrics.accuracy(labels, 
                                                           predictions['classes'])}
            return tf.estimator.EstimatorSpec(mode = mode,
                                          eval_metric_ops = eval_metric_op,
                                          loss = loss)
        


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.DEBUG)
#    tf.logging.set_verbosity(tf.logging.ERROR)
    
    #We load the faces dataset
#    lfw_people = fetch_lfw_people(min_faces_per_person=1, resize=1)
#    positive_data = lfw_people.images
#    w, h = positive_data.shape[1], positive_data.shape[2]
#    positive_data = np.reshape(positive_data, (-1, w, h, 1))
#    positive_data = positive_data / 255
#    #negative_data
#    negative_data = []
#    for file_name in os.listdir('images/car_side'):
#        print('Loading file %s' % file_name)
#        img = cv2.imread('images/car_side/' + file_name)
#        #The images are black and white
#        negative_data.append(img[:,:, 0 : 1] / 255)
#    face_recognizer = LiFaceRecognizer(800, 600)
#    face_recognizer.set_training_data_set_positive(positive_data)
#    face_recognizer.set_training_data_set_negative(negative_data)
#    face_recognizer.train(1, 1, 200)
    
    #Pandas dataframe that will contain part of the data from the WIKI database
    w, h = 96, 96
    nb_images = 5000
    f = 'D:\Data sets\Faces\wiki.tar\wiki\wiki\wiki.mat'
    df = load_IMDB_WIKI.load_IMDB_WIKI(f, nb = nb_images)
    #We eliminate images where no face was detected
    df = df[df.face_score > -np.inf]
    df = df[df.face_location != None]
    #We eliminate images where more than one face was detected
    df = df[np.isnan(df.second_face_score)]
    #We update the number of images, as some have been removed by the two 
    #former operations
    nb_images = len(df)
    positive_data = np.zeros(shape = (nb_images, w, h, 3))
    negative_data = []
    positive_count = 0
#    for i, v  in enumerate(zip(df.image, df.face_location)):
#        image_path = v[0]
#        face_location = v[1]
#        image_data = cv2.imread(image_path)
#        if image_data is None:
#            continue
#        positive_count += 1
#        image_face = image_data[int(face_location[1]) : int(face_location[3]),
#                                int(face_location[0]) : int(face_location[2]),
#                                :]
#        image_face = cv2.resize(image_face, (w,h)).copy()
#        positive_data[i, :, :, :] = image_face
#        try:
#            temp = image_data[:,
#                int(0.1*h + face_location[2]):, :]
#            if temp.shape[1] > h:
#                negative_data.append(temp / 255)
#        except:
#            pass
#    positive_data = positive_data[:positive_count, :, :, :]
#    
#    #Standardization of the data
#    positive_data = positive_data / 255
#    
    face_recognizer = LiFaceRecognizer(800, 600, blackwhite = False)
#    face_recognizer.set_training_data_set_positive(positive_data)
#    face_recognizer.set_training_data_set_negative(negative_data)
#    face_recognizer.train(200, 200, 1500)
    
    
    #Detection of faces from the webcam
    cap = cv2.VideoCapture(0)
    face_recognizer.initialize_predictors()
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        frame = frame[58:415, :, :]
    
        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
        # Display the resulting frame
        shape0 = frame.shape[0]
        frame = frame.reshape((shape0, 640, 3))
        frame = frame / 255
        frame = face_recognizer.identify_faces(frame)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    