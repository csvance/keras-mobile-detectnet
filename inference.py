from model import MobileDetectNetModel
from train import MobileDetectNetSequence
import cv2
import matplotlib.pyplot as plt
import tensorflow.keras as keras
import numpy as np
import time
import plac
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


@plac.annotations(
    inference_type=("Type of inference to test (TF, FP32, FP16, INT8)", 'option', 'T', str),
    batch_size=("Size of the TensorRT batch", 'option', 'B', int),
    weights=("Model weights", 'option', 'W', str),
    multi_gpu_weights=("Multi GPU model weights", 'option', 'G', str),
    test_path=("Test images path", 'option', 'I', str),
    merge=("Test images only: Merge detected regions", 'flag', 'm', bool),
    stage=("Test images only: Augmentation training stage", 'option', 's', str),
    limit=("Test images only: Max number of images to run inference on", 'option', 'l', int),
    confidence=("Test images only: Minimum confidence in coverage to draw bbox", "option", "c", float),
    model=("Which model to load (complete, coverage, regions, pooling)", "option", 'M', str)
)
def main(inference_type: str = "K",
         batch_size: int = 1,
         test_path: str = None,
         weights: str = None,
         multi_gpu_weights: str = None,
         merge: bool = False,
         stage: str = "test",
         limit: int = 20,
         confidence: float = 0.1,
         model="complete"):

    if model is None or model == 'complete':
        keras_model = MobileDetectNetModel.complete_model()
    elif model == 'coverage':
        keras_model = MobileDetectNetModel.coverage_model()
    elif model == 'region':
        cnn = MobileDetectNetModel.cnn()
        coverage, _ = MobileDetectNetModel.coverage(cnn.output)
        regions, _ = MobileDetectNetModel.region(coverage)

        keras_model = keras.models.Model(inputs=cnn.input, outputs=[coverage, regions])
        keras_model.summary()

        keras_model.load_weights('region.hdf5', by_name=True)
        keras_model.load_weights('coverage.hdf5', by_name=True)

    elif model == 'pooling':
        raise Exception("Not implemented")
    else:
        raise Exception("Invalid model")

    if multi_gpu_weights is not None:
        keras_model = keras.utils.multi_gpu_model(keras_model, gpus=[0, 1], cpu_merge=True, cpu_relocation=False)
        keras_model.load_weights(multi_gpu_weights)
    elif weights is not None:
        keras_model.load_weights(weights)

    images_done = 0

    seq = MobileDetectNetSequence.create_augmenter(stage)

    images_full = []
    images_input = []

    for r, d, f in os.walk(test_path):
        for file in f:
            image_full = cv2.imread(os.path.join(r, file))
            image_input = cv2.resize(image_full, (224, 224))

            seq_det = seq.to_deterministic()
            image_aug = (seq_det.augment_image(image_input).astype(np.float32) / 127.5) - 1.

            images_full.append(image_full)
            images_input.append(image_aug)

            images_done += 1

            if images_done == limit:
                break

        if images_done == limit:
            break

    x_test = np.array(images_input)

    if inference_type == 'K':
        t0 = time.time()
        model_outputs = keras_model.predict(x_test)
        t1 = time.time()
    elif inference_type == 'TF':
        tf_engine = keras_model.tf_engine()
        t0 = time.time()
        model_outputs = tf_engine.infer(x_test)
        t1 = time.time()
    elif inference_type == 'FP32':
        tftrt_engine = keras_model.tftrt_engine(precision='FP32', batch_size=batch_size)
        t0 = time.time()
        model_outputs = tftrt_engine.infer(x_test)
        t1 = time.time()
    elif inference_type == 'FP16':
        tftrt_engine = keras_model.tftrt_engine(precision='FP16', batch_size=batch_size)
        t0 = time.time()
        model_outputs = tftrt_engine.infer(x_test)
        t1 = time.time()
    elif inference_type == 'INT8':
        tftrt_engine = keras_model.tftrt_engine(precision='INT8', batch_size=batch_size)
        t0 = time.time()
        model_outputs = tftrt_engine.infer(x_test)
        t1 = time.time()
    else:
        raise Exception("Invalid inference type")

    print('Time', t1 - t0)

    if model == 'complete':

        coverage, bboxes, bboxes_center = model_outputs

        if images_full is not None:
            for idx in range(0, len(images_full)):

                rectangles = []
                for y in range(0, 14):
                    for x in range(0, 14):

                        if coverage[idx, y, x] > confidence:
                            rect = [
                                int(bboxes_center[idx, int(y), int(x), 0] * 224),
                                int(bboxes_center[idx, int(y), int(x), 1] * 224),
                                int(bboxes_center[idx, int(y), int(x), 2] * 224),
                                int(bboxes_center[idx, int(y), int(x), 3] * 224)]

                            rectangles.append(rect)

                if merge:
                    rectangles, merges = cv2.groupRectangles(rectangles, 1, eps=0.75)

                for rect in rectangles:
                    cv2.rectangle(images_input[idx],
                                  (rect[0], rect[1]),
                                  (rect[2], rect[3]),
                                  (0, 1, 0), 3)

                plt.imshow((images_input[idx] + 1) / 2, alpha=1.0)
                plt.imshow(
                    cv2.resize(coverage[idx].reshape((14, 14)),
                               (x_test.shape[1], x_test.shape[2])),
                    interpolation='nearest', alpha=0.5)
                plt.show()

    elif model == 'coverage':
        coverage = model_outputs

        if images_full is not None:
            for idx in range(0, len(images_full)):
                plt.imshow((images_input[idx] + 1) / 2, alpha=1.0)
                plt.imshow(
                    cv2.resize(coverage[idx].reshape((14, 14)),
                               (x_test.shape[1], x_test.shape[2])),
                    interpolation='nearest', alpha=0.5)
                plt.show()

    elif model == 'region':
        coverage, bboxes = model_outputs

        if images_full is not None:
            for idx in range(0, len(images_full)):

                rectangles = []
                for y in range(0, 14):
                    for x in range(0, 14):

                        if coverage[idx, y, x] > confidence:
                            rect = [
                                int(bboxes[idx, int(y), int(x), 0] * 224),
                                int(bboxes[idx, int(y), int(x), 1] * 224),
                                int(bboxes[idx, int(y), int(x), 2] * 224),
                                int(bboxes[idx, int(y), int(x), 3] * 224)]

                            rectangles.append(rect)

                if merge:
                    rectangles, merges = cv2.groupRectangles(rectangles, 1, eps=0.75)

                for rect in rectangles:
                    cv2.rectangle(images_input[idx],
                                  (rect[0], rect[1]),
                                  (rect[2], rect[3]),
                                  (0, 1, 0), 3)

                plt.imshow((images_input[idx] + 1) / 2, alpha=1.0)
                plt.imshow(
                    cv2.resize(coverage[idx].reshape((14, 14)),
                               (x_test.shape[1], x_test.shape[2])),
                    interpolation='nearest', alpha=0.5)
                plt.show()


if __name__ == '__main__':
    plac.call(main)
