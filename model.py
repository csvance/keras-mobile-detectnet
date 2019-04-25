import copy
import numpy as np
import os

from typing import Optional

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Reshape, UpSampling2D, Conv2D
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


class MobileDetectNetModel(Model):
    @staticmethod
    def create(input_width: int = 224,
               input_height: int = 224,
               feature_upsample: int = 2,
               weights: Optional[str] = "imagenet"):

        mobilenet = keras.applications.mobilenet.MobileNet(include_top=False,
                                                           input_shape=(input_height, input_width, 3),
                                                           weights=weights,
                                                           alpha=0.25)

        new_output = mobilenet.get_layer('conv_pw_13_relu').output

        if feature_upsample == 1:
            coverage = Dense(1, activation='sigmoid', name='sigmoid')(new_output)
        else:
            choke = Dense(2*feature_upsample, activation='relu', name='choke')(new_output)
            upsample = UpSampling2D(feature_upsample, 'channels_last', name='up_sampling2d')(choke)
            coverage = Dense(1, activation='sigmoid', name='coverage')(upsample)

        coverage_height = int(coverage.shape[1])
        coverage_width = int(coverage.shape[2])

        bbox_width = int(coverage_width / feature_upsample)
        bbox_height = int(coverage_height / feature_upsample)

        flatten = Flatten()(coverage)
        bboxes_preshape = Dense(bbox_height * bbox_width * 4, activation='linear', name='bboxes_preshape')(flatten)
        bboxes = Reshape((bbox_height, bbox_width, 4), name='bboxes')(bboxes_preshape)

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
    mobiledetectnet, coverage_shape = MobileDetectNetModel.create()
    mobiledetectnet.summary()
    mobiledetectnet.plot()
