import copy
import numpy as np
import os

from typing import Optional

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Reshape, UpSampling2D, Conv2D, Dropout, BatchNormalization, Activation, Layer
import tensorflow.keras as keras
from tensorflow.keras import backend as K
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


class TranslationalVariantConv2D(Layer):
    def __init__(self, **kwargs):

        self.output_dim = None

        super(TranslationalVariantConv2D, self).__init__(**kwargs)

    def build(self, input_shape):

        self.output_dim = tf.TensorShape((input_shape[0], tf.Dimension(7), tf.Dimension(7), tf.Dimension(4)))
        self.kernel = self.add_weight(name='kernel',
                                      shape=tf.TensorShape((self.output_dim[1], self.output_dim[2], tf.Dimension(2), tf.Dimension(2), tf.Dimension(1), tf.Dimension(4))),
                                      initializer='uniform',
                                      trainable=True)

        super(TranslationalVariantConv2D, self).build(input_shape)

    def call(self, x):
        rows = []
        cols = []
        t_0_0 = K.conv2d(x[:, 0 * 2:0 * 2 + 2, 0 * 2:0 * 2 + 2], kernel=self.kernel[0, 0], strides=(2, 2))
        cols.append(t_0_0)
        t_0_1 = K.conv2d(x[:, 0 * 2:0 * 2 + 2, 1 * 2:1 * 2 + 2], kernel=self.kernel[0, 1], strides=(2, 2))
        cols.append(t_0_1)
        t_0_2 = K.conv2d(x[:, 0 * 2:0 * 2 + 2, 2 * 2:2 * 2 + 2], kernel=self.kernel[0, 2], strides=(2, 2))
        cols.append(t_0_2)
        t_0_3 = K.conv2d(x[:, 0 * 2:0 * 2 + 2, 3 * 2:3 * 2 + 2], kernel=self.kernel[0, 3], strides=(2, 2))
        cols.append(t_0_3)
        t_0_4 = K.conv2d(x[:, 0 * 2:0 * 2 + 2, 4 * 2:4 * 2 + 2], kernel=self.kernel[0, 4], strides=(2, 2))
        cols.append(t_0_4)
        t_0_5 = K.conv2d(x[:, 0 * 2:0 * 2 + 2, 5 * 2:5 * 2 + 2], kernel=self.kernel[0, 5], strides=(2, 2))
        cols.append(t_0_5)
        t_0_6 = K.conv2d(x[:, 0 * 2:0 * 2 + 2, 6 * 2:6 * 2 + 2], kernel=self.kernel[0, 6], strides=(2, 2))
        cols.append(t_0_6)
        rows.append(K.concatenate(cols, axis=2))
        cols = []
        t_1_0 = K.conv2d(x[:, 1 * 2:1 * 2 + 2, 0 * 2:0 * 2 + 2], kernel=self.kernel[1, 0], strides=(2, 2))
        cols.append(t_1_0)
        t_1_1 = K.conv2d(x[:, 1 * 2:1 * 2 + 2, 1 * 2:1 * 2 + 2], kernel=self.kernel[1, 1], strides=(2, 2))
        cols.append(t_1_1)
        t_1_2 = K.conv2d(x[:, 1 * 2:1 * 2 + 2, 2 * 2:2 * 2 + 2], kernel=self.kernel[1, 2], strides=(2, 2))
        cols.append(t_1_2)
        t_1_3 = K.conv2d(x[:, 1 * 2:1 * 2 + 2, 3 * 2:3 * 2 + 2], kernel=self.kernel[1, 3], strides=(2, 2))
        cols.append(t_1_3)
        t_1_4 = K.conv2d(x[:, 1 * 2:1 * 2 + 2, 4 * 2:4 * 2 + 2], kernel=self.kernel[1, 4], strides=(2, 2))
        cols.append(t_1_4)
        t_1_5 = K.conv2d(x[:, 1 * 2:1 * 2 + 2, 5 * 2:5 * 2 + 2], kernel=self.kernel[1, 5], strides=(2, 2))
        cols.append(t_1_5)
        t_1_6 = K.conv2d(x[:, 1 * 2:1 * 2 + 2, 6 * 2:6 * 2 + 2], kernel=self.kernel[1, 6], strides=(2, 2))
        cols.append(t_1_6)
        rows.append(K.concatenate(cols, axis=2))
        cols = []
        t_2_0 = K.conv2d(x[:, 2 * 2:2 * 2 + 2, 0 * 2:0 * 2 + 2], kernel=self.kernel[2, 0], strides=(2, 2))
        cols.append(t_2_0)
        t_2_1 = K.conv2d(x[:, 2 * 2:2 * 2 + 2, 1 * 2:1 * 2 + 2], kernel=self.kernel[2, 1], strides=(2, 2))
        cols.append(t_2_1)
        t_2_2 = K.conv2d(x[:, 2 * 2:2 * 2 + 2, 2 * 2:2 * 2 + 2], kernel=self.kernel[2, 2], strides=(2, 2))
        cols.append(t_2_2)
        t_2_3 = K.conv2d(x[:, 2 * 2:2 * 2 + 2, 3 * 2:3 * 2 + 2], kernel=self.kernel[2, 3], strides=(2, 2))
        cols.append(t_2_3)
        t_2_4 = K.conv2d(x[:, 2 * 2:2 * 2 + 2, 4 * 2:4 * 2 + 2], kernel=self.kernel[2, 4], strides=(2, 2))
        cols.append(t_2_4)
        t_2_5 = K.conv2d(x[:, 2 * 2:2 * 2 + 2, 5 * 2:5 * 2 + 2], kernel=self.kernel[2, 5], strides=(2, 2))
        cols.append(t_2_5)
        t_2_6 = K.conv2d(x[:, 2 * 2:2 * 2 + 2, 6 * 2:6 * 2 + 2], kernel=self.kernel[2, 6], strides=(2, 2))
        cols.append(t_2_6)
        rows.append(K.concatenate(cols, axis=2))
        cols = []
        t_3_0 = K.conv2d(x[:, 3 * 2:3 * 2 + 2, 0 * 2:0 * 2 + 2], kernel=self.kernel[3, 0], strides=(2, 2))
        cols.append(t_3_0)
        t_3_1 = K.conv2d(x[:, 3 * 2:3 * 2 + 2, 1 * 2:1 * 2 + 2], kernel=self.kernel[3, 1], strides=(2, 2))
        cols.append(t_3_1)
        t_3_2 = K.conv2d(x[:, 3 * 2:3 * 2 + 2, 2 * 2:2 * 2 + 2], kernel=self.kernel[3, 2], strides=(2, 2))
        cols.append(t_3_2)
        t_3_3 = K.conv2d(x[:, 3 * 2:3 * 2 + 2, 3 * 2:3 * 2 + 2], kernel=self.kernel[3, 3], strides=(2, 2))
        cols.append(t_3_3)
        t_3_4 = K.conv2d(x[:, 3 * 2:3 * 2 + 2, 4 * 2:4 * 2 + 2], kernel=self.kernel[3, 4], strides=(2, 2))
        cols.append(t_3_4)
        t_3_5 = K.conv2d(x[:, 3 * 2:3 * 2 + 2, 5 * 2:5 * 2 + 2], kernel=self.kernel[3, 5], strides=(2, 2))
        cols.append(t_3_5)
        t_3_6 = K.conv2d(x[:, 3 * 2:3 * 2 + 2, 6 * 2:6 * 2 + 2], kernel=self.kernel[3, 6], strides=(2, 2))
        cols.append(t_3_6)
        rows.append(K.concatenate(cols, axis=2))
        cols = []
        t_4_0 = K.conv2d(x[:, 4 * 2:4 * 2 + 2, 0 * 2:0 * 2 + 2], kernel=self.kernel[4, 0], strides=(2, 2))
        cols.append(t_4_0)
        t_4_1 = K.conv2d(x[:, 4 * 2:4 * 2 + 2, 1 * 2:1 * 2 + 2], kernel=self.kernel[4, 1], strides=(2, 2))
        cols.append(t_4_1)
        t_4_2 = K.conv2d(x[:, 4 * 2:4 * 2 + 2, 2 * 2:2 * 2 + 2], kernel=self.kernel[4, 2], strides=(2, 2))
        cols.append(t_4_2)
        t_4_3 = K.conv2d(x[:, 4 * 2:4 * 2 + 2, 3 * 2:3 * 2 + 2], kernel=self.kernel[4, 3], strides=(2, 2))
        cols.append(t_4_3)
        t_4_4 = K.conv2d(x[:, 4 * 2:4 * 2 + 2, 4 * 2:4 * 2 + 2], kernel=self.kernel[4, 4], strides=(2, 2))
        cols.append(t_4_4)
        t_4_5 = K.conv2d(x[:, 4 * 2:4 * 2 + 2, 5 * 2:5 * 2 + 2], kernel=self.kernel[4, 5], strides=(2, 2))
        cols.append(t_4_5)
        t_4_6 = K.conv2d(x[:, 4 * 2:4 * 2 + 2, 6 * 2:6 * 2 + 2], kernel=self.kernel[4, 6], strides=(2, 2))
        cols.append(t_4_6)
        rows.append(K.concatenate(cols, axis=2))
        cols = []
        t_5_0 = K.conv2d(x[:, 5 * 2:5 * 2 + 2, 0 * 2:0 * 2 + 2], kernel=self.kernel[5, 0], strides=(2, 2))
        cols.append(t_5_0)
        t_5_1 = K.conv2d(x[:, 5 * 2:5 * 2 + 2, 1 * 2:1 * 2 + 2], kernel=self.kernel[5, 1], strides=(2, 2))
        cols.append(t_5_1)
        t_5_2 = K.conv2d(x[:, 5 * 2:5 * 2 + 2, 2 * 2:2 * 2 + 2], kernel=self.kernel[5, 2], strides=(2, 2))
        cols.append(t_5_2)
        t_5_3 = K.conv2d(x[:, 5 * 2:5 * 2 + 2, 3 * 2:3 * 2 + 2], kernel=self.kernel[5, 3], strides=(2, 2))
        cols.append(t_5_3)
        t_5_4 = K.conv2d(x[:, 5 * 2:5 * 2 + 2, 4 * 2:4 * 2 + 2], kernel=self.kernel[5, 4], strides=(2, 2))
        cols.append(t_5_4)
        t_5_5 = K.conv2d(x[:, 5 * 2:5 * 2 + 2, 5 * 2:5 * 2 + 2], kernel=self.kernel[5, 5], strides=(2, 2))
        cols.append(t_5_5)
        t_5_6 = K.conv2d(x[:, 5 * 2:5 * 2 + 2, 6 * 2:6 * 2 + 2], kernel=self.kernel[5, 6], strides=(2, 2))
        cols.append(t_5_6)
        rows.append(K.concatenate(cols, axis=2))
        cols = []
        t_6_0 = K.conv2d(x[:, 6 * 2:6 * 2 + 2, 0 * 2:0 * 2 + 2], kernel=self.kernel[6, 0], strides=(2, 2))
        cols.append(t_6_0)
        t_6_1 = K.conv2d(x[:, 6 * 2:6 * 2 + 2, 1 * 2:1 * 2 + 2], kernel=self.kernel[6, 1], strides=(2, 2))
        cols.append(t_6_1)
        t_6_2 = K.conv2d(x[:, 6 * 2:6 * 2 + 2, 2 * 2:2 * 2 + 2], kernel=self.kernel[6, 2], strides=(2, 2))
        cols.append(t_6_2)
        t_6_3 = K.conv2d(x[:, 6 * 2:6 * 2 + 2, 3 * 2:3 * 2 + 2], kernel=self.kernel[6, 3], strides=(2, 2))
        cols.append(t_6_3)
        t_6_4 = K.conv2d(x[:, 6 * 2:6 * 2 + 2, 4 * 2:4 * 2 + 2], kernel=self.kernel[6, 4], strides=(2, 2))
        cols.append(t_6_4)
        t_6_5 = K.conv2d(x[:, 6 * 2:6 * 2 + 2, 5 * 2:5 * 2 + 2], kernel=self.kernel[6, 5], strides=(2, 2))
        cols.append(t_6_5)
        t_6_6 = K.conv2d(x[:, 6 * 2:6 * 2 + 2, 6 * 2:6 * 2 + 2], kernel=self.kernel[6, 6], strides=(2, 2))
        cols.append(t_6_6)
        rows.append(K.concatenate(cols, axis=2))

        return K.concatenate(rows, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.output_dim


class MobileDetectNetModel(Model):
    @staticmethod
    def create(input_width: int = 224,
               input_height: int = 224,
               feature_upsample: int = 1,
               weights: Optional[str] = "imagenet"):

        mobilenet = keras.applications.mobilenet.MobileNet(include_top=False,
                                                           input_shape=(input_height, input_width, 3),
                                                           weights=weights,
                                                           alpha=0.25)

        new_output = mobilenet.get_layer('conv_pw_13_relu').output

        if feature_upsample == 1:
            coverage = Conv2D(1, 3, activation='sigmoid', padding='same', name='coverage')(new_output)
        else:
            choke = Conv2D(4, 3, padding='same', name='choke')(new_output)
            batchnorm_choke = BatchNormalization(name='batchnorm_choke')(choke)
            batchnorm_choke_relu = Activation('relu', name='batchnorm_choke_relu')(batchnorm_choke)

            upsample = UpSampling2D(feature_upsample, 'channels_last', name='up_sampling2d')(batchnorm_choke_relu)
            coverage = Conv2D(1, 1, activation='sigmoid', name='coverage')(upsample)

        coverage_height = int(coverage.shape[1])
        coverage_width = int(coverage.shape[2])

        if feature_upsample == 1:
            bboxes = Conv2D(4, 2, strides=feature_upsample, activation='linear', name='bboxes')(coverage)
        else:
            bboxes = TranslationalVariantConv2D(name='bboxes')(coverage)

        return (MobileDetectNetModel(inputs=mobilenet.input,
                                     outputs=[coverage, bboxes]),
                                    (coverage_height, coverage_width))

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
    mobiledetectnet, coverage_shape = MobileDetectNetModel.create(feature_upsample=2)
    mobiledetectnet.summary()
    mobiledetectnet.plot()
