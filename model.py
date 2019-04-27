import copy
import numpy as np
import os

from typing import Optional

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, UpSampling2D, Conv2D, BatchNormalization, Activation, Layer, Lambda, Input, \
    Concatenate
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from keras.utils.generic_utils import get_custom_objects
import tensorflow as tf
from tensorflow.contrib import tensorrt as tftrt

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class MobileDetectNetFrozenGraph(object):
    def __init__(self, model, shape):
        shape = (None, shape[0], shape[1], shape[2])
        x_name = 'image_tensor_x'
        with K.get_session() as sess:
            x_tensor = tf.placeholder(tf.float32, shape, x_name)
            K.set_learning_phase(0)
            y_tensor = model(x_tensor)
            y_name = [y_tensor[-1].name[:-2], y_tensor[-2].name[:-2]]
            graph = sess.graph.as_graph_def()
            graph0 = tf.graph_util.convert_variables_to_constants(sess, graph, y_name)
            graph1 = tf.graph_util.remove_training_nodes(graph0)

        self.x_name = [x_name]
        self.y_name = y_name
        self.frozen = graph1
        self.model = model


class MobileDetectNetTFEngine(object):
    def __init__(self, graph):
        g = tf.Graph()
        with g.as_default():
            x_op, y_op1, y_op2 = tf.import_graph_def(
                graph_def=graph.frozen, return_elements=graph.x_name + graph.y_name)
            self.x_tensor = x_op.outputs[0]

            self.y_tensor1 = y_op1.outputs[0]
            self.y_tensor2 = y_op2.outputs[0]

        config = tf.ConfigProto(gpu_options=
                                tf.GPUOptions(per_process_gpu_memory_fraction=0.5,
                                              allow_growth=True))

        self.sess = tf.Session(graph=g, config=config)

    def infer(self, x):
        y1, y2 = self.sess.run([self.y_tensor1, self.y_tensor2],
                               feed_dict={self.x_tensor: x})
        return y2, y1


class BBoxMultiply(Layer):
    def __init__(self, **kwargs):
        super(BBoxMultiply, self).__init__(**kwargs)

    def build(self, input_shape):

        self.kernel = self.add_weight(
            name='kernel',
            shape=(1, 14, 14, 4),
            initializer='zero',
            dtype='float32',
            trainable=False,
        )

        weights = np.zeros((1, input_shape[1], input_shape[2], 4), dtype=np.float32)
        weights[0, :, :, 0] = np.mgrid[0:14, 0:14][1]
        weights[0, :, :, 1] = np.mgrid[0:14, 0:14][0]
        weights[0, :, :, 2] = np.mgrid[0:14, 0:14][1]
        weights[0, :, :, 3] = np.mgrid[0:14, 0:14][0]
        weights = weights / 14

        tf.keras.backend.set_value(self.kernel, weights)

        super(BBoxMultiply, self).build(input_shape)

    def call(self, x):
        return tf.math.multiply(self.kernel, x)

    def compute_output_shape(self, input_shape):
        return input_shape


class MobileDetectnetTFTRTEngine(MobileDetectNetTFEngine):
    def __init__(self, graph, batch_size, precision):
        tftrt_graph = tftrt.create_inference_graph(
            graph.frozen,
            outputs=graph.y_name,
            max_batch_size=batch_size,
            max_workspace_size_bytes=1 << 30,
            precision_mode=precision,
            minimum_segment_size=2)

        self.tftrt_graph = tftrt_graph
        self.graph = graph

        opt_graph = copy.deepcopy(graph)
        opt_graph.frozen = tftrt_graph
        super(MobileDetectnetTFTRTEngine, self).__init__(opt_graph)
        self.batch_size = batch_size

    def infer(self, x):
        num_tests = x.shape[0]

        bboxes_height = int(self.graph.model.get_layer('bboxes').output.shape[1])
        bboxes_width = int(self.graph.model.get_layer('bboxes').output.shape[2])

        coverage_height = int(self.graph.model.get_layer('coverage').output.shape[1])
        coverage_width = int(self.graph.model.get_layer('coverage').output.shape[2])

        y1 = np.zeros((num_tests, bboxes_height, bboxes_width, 4), np.float32)
        y2 = np.zeros((num_tests, coverage_height, coverage_width, 1), np.float32)
        batch_size = self.batch_size

        for i in range(0, num_tests, batch_size):
            x_part = x[i: i + batch_size]
            y_part1, y_part2 = self.sess.run([self.y_tensor1, self.y_tensor2],
                                             feed_dict={self.x_tensor: x_part})
            y1[i: i + batch_size] = y_part1
            y2[i: i + batch_size] = y_part2

        return y2, y1


class MobileDetectNetModel(Model):
    CNN_OUT_DIMS = (7, 7, 256)

    @staticmethod
    def cnn(input_width: int = 224,
                  input_height: int = 224,
                  transfer_weights: Optional[str] = "imagenet"):

        return keras.applications.mobilenet.MobileNet(include_top=False,
                                                      input_shape=(input_height, input_width, 3),
                                                      weights=transfer_weights,
                                                      alpha=0.25)

    @staticmethod
    def coverage(coverage_input=None):

        if coverage_input is None:
            # Input is the MobileNet feature map
            coverage_input = Input(shape=(MobileDetectNetModel.CNN_OUT_DIMS[0],
                                          MobileDetectNetModel.CNN_OUT_DIMS[1],
                                          256), name='coverage_input')

        # Force the network to compress
        coverage_conv2d_1 = Conv2D(4, kernel_size=3, padding='same', name='coverage_conv2d_1')(coverage_input)
        coverage_batchnorm_1 = BatchNormalization(name='coverage_batchnorm_1')(coverage_conv2d_1)
        coverage_activation_1 = Activation('relu', name='coverage_activation_1')(coverage_batchnorm_1)

        # We upsample to allow for a more fine grained coverage map
        coverage_upsample_1 = UpSampling2D(2, 'channels_last', name='coverage_upsample_1')(coverage_activation_1)

        coverage = Conv2D(1, 1, activation='sigmoid', name='coverage')(coverage_upsample_1)

        return coverage, coverage_input

    @staticmethod
    def region(region_input=None):

        # The input is the coverage map
        if region_input is None:
            region_input = Input(shape=(MobileDetectNetModel.CNN_OUT_DIMS[0] * 2,
                                        MobileDetectNetModel.CNN_OUT_DIMS[1] * 2,
                                        1), name='coverage')

        region_conv2d_1 = Conv2D(4, kernel_size=3, padding='same', name='region_conv2d_1')(region_input)
        region_batchnorm_1 = BatchNormalization(name='region_batchnorm_1')(region_conv2d_1)
        region_activation_1 = Activation('relu', name='region_activation_1')(region_batchnorm_1)

        # Multiply the entire previous coverage map with a linear activation
        bboxes = BBoxMultiply(name='bboxes')(region_activation_1)

        return bboxes, region_input

    @staticmethod
    def pooling(coverage_input=None, region_input=None):

        if coverage_input is None:
            coverage_input = Input(shape=(MobileDetectNetModel.CNN_OUT_DIMS[0] * 2,
                                          MobileDetectNetModel.CNN_OUT_DIMS[1] * 2,
                                          1), name='coverage')

        if region_input is None:
            region_input = Input(shape=(MobileDetectNetModel.CNN_OUT_DIMS[0] * 2,
                                        MobileDetectNetModel.CNN_OUT_DIMS[1] * 2,
                                        1), name='bboxes')

        pooling_concatenate = Concatenate(axis=-1)([region_input, coverage_input])

        pooling_conv2d_1 = Conv2D(4, kernel_size=3, name='pooling_conv2d_1', padding='same')(pooling_concatenate)
        pooling_batchnorm_1 = BatchNormalization(name='pooling_batchnorm_1')(pooling_conv2d_1)
        pooling_activation_1 = Activation('relu', name='pooling_activation_1')(pooling_batchnorm_1)

        # Multiply the entire previous coverage map with a linear activation
        bboxes_pooled = BBoxMultiply(name='bboxes_pooled')(pooling_activation_1)

        return bboxes_pooled, coverage_input, region_input

    @staticmethod
    def complete_model():

        cnn = MobileDetectNetModel.cnn()
        coverage, _ = MobileDetectNetModel.coverage(cnn.output)
        region, _ = MobileDetectNetModel.region(coverage)
        pooling, _, _ = MobileDetectNetModel.pooling(coverage, region)

        return MobileDetectNetModel(inputs=cnn.input, outputs=[coverage, region, pooling])

    @staticmethod
    def coverage_model():
        cnn = MobileDetectNetModel.cnn()
        coverage, _ = MobileDetectNetModel.coverage(cnn.output)

        return Model(inputs=cnn.input, outputs=coverage)

    @staticmethod
    def region_model():
        coverage, coverage_input = MobileDetectNetModel.coverage()
        region, _ = MobileDetectNetModel.region(coverage)

        return Model(inputs=coverage_input, outputs=region)

    @staticmethod
    def pooling_model():
        region, region_input = MobileDetectNetModel.region()
        pooling, coverage_input, _ = MobileDetectNetModel.pooling(None, region)

        return Model(inputs=[region_input, coverage_input], outputs=pooling)

    def plot(self, path: str = "mobiledetectnet_plot.png"):
        from tensorflow.keras.utils import plot_model
        plot_model(self, to_file=path, show_shapes=True)

    def freeze(self):
        return MobileDetectNetFrozenGraph(self, (self.input.shape[1], self.input.shape[2], self.input.shape[3]))

    def tf_engine(self):
        return MobileDetectNetTFEngine(self.freeze())

    def tftrt_engine(self, batch_size: int = 1, precision: str = 'FP16'):
        return MobileDetectnetTFTRTEngine(self.freeze(), batch_size=batch_size, precision=precision)


if __name__ == '__main__':
    mobiledetectnet = MobileDetectNetModel.coverage_model()
    mobiledetectnet.summary()
    #mobiledetectnet.plot()
