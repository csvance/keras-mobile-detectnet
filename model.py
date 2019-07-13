import numpy as np
import os

from typing import Optional

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Activation, Layer, Input, \
    Flatten, Reshape
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

        # deep copy causes issues with the latest graph (apparently it contains an RLock
        # passing this by reference seems to work, but more investigation is needed.
        # opt_graph = copy.deepcopy(graph)

        opt_graph = graph
        opt_graph.frozen = tftrt_graph
        super(MobileDetectnetTFTRTEngine, self).__init__(opt_graph)
        self.batch_size = batch_size

    def infer(self, x):
        num_tests = x.shape[0]

        bboxes_height = int(self.graph.model.get_layer('bboxes').output.shape[1])
        bboxes_width = int(self.graph.model.get_layer('bboxes').output.shape[2])

        classes_height = int(self.graph.model.get_layer('classes').output.shape[1])
        classes_width = int(self.graph.model.get_layer('classes').output.shape[2])
        classes_nb = int(self.graph.model.get_layer('classes').output.shape[3])

        y1 = np.zeros((num_tests, bboxes_height, bboxes_width, 4), np.float32)
        y2 = np.zeros((num_tests, classes_height, classes_width, classes_nb), np.float32)

        batch_size = self.batch_size

        for i in range(0, num_tests, batch_size):
            x_part = x[i: i + batch_size]
            y_part1, y_part2 = self.sess.run([self.y_tensor1, self.y_tensor2],
                                             feed_dict={self.x_tensor: x_part})
            y2[i: i + batch_size] = y_part1
            y1[i: i + batch_size] = y_part2

        return y2, y1


class MobileDetectNetModel(Model):

    @staticmethod
    def cnn(input_width: int = 224,
            input_height: int = 224,
            transfer_weights: Optional[str] = "imagenet"):

        return keras.applications.MobileNet(include_top=False,
                                            input_shape=(input_height, input_width, 3),
                                            weights=transfer_weights,
                                            alpha=0.25)

    @staticmethod
    def region(region_input=None):

        # The input is the coverage map
        if region_input is None:
            region_input = Input(shape=(7, 7, 1), name='region_input')

        x = Conv2D(9, 3, padding='same', name='region_conv2d_1')(region_input)
        x = BatchNormalization(name='region_batchnorm_1')(x)
        x = Activation('relu', name='region_activation_1')(x)

        # Multiply the entire previous coverage map with a linear activation
        region = Conv2D(9, 1, activation='sigmoid', name='region')(x)

        return region, region_input

    @staticmethod
    def bboxes(region_input=None):

        if region_input is None:
            region_input = Input(shape=(7, 7, 9), name='region_input')

        x = Flatten(name='bboxes_flatten')(region_input)
        x = Dense(7 * 7 * 4, name='bboxes_dense')(x)

        bboxes = Reshape((7, 7, 4), name='bboxes')(x)

        return bboxes, region_input

    @staticmethod
    def classes(cnn_input=None):

        if cnn_input is None:
            cnn_input = Input(shape=(7, 7, 256), name='cnn_input')

        x = Conv2D(4, 3, padding='same', name='classes_conv2d')(cnn_input)
        x = BatchNormalization(name='classes_batchnorm')(x)
        x = Activation('relu', name='classes_activation')(x)
        x = Flatten(name='classes_flatten')(x)
        x = Dense(7 * 7 * 1, name='classes_dense', activation='sigmoid')(x)

        classes = Reshape((7, 7, 1), name='classes')(x)

        return classes, cnn_input

    @staticmethod
    def complete_model(extra_inputs: Optional[list] = None):

        cnn = MobileDetectNetModel.cnn()
        region, _ = MobileDetectNetModel.region(cnn.output)
        bboxes, _ = MobileDetectNetModel.bboxes(region)
        classes, _ = MobileDetectNetModel.classes(cnn.output)

        model_inputs = [cnn.input]
        model_outputs = [region, bboxes, classes]

        if extra_inputs is not None:
            model_inputs.extend(extra_inputs)

        return MobileDetectNetModel(inputs=model_inputs, outputs=model_outputs)

    @staticmethod
    def region_model():
        region, region_input = MobileDetectNetModel.region()

        return Model(inputs=region_input, outputs=region)

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
    mobiledetectnet = MobileDetectNetModel.complete_model()
    mobiledetectnet.summary()
    mobiledetectnet.plot()
