from model import MobileDetectNetModel

import numpy as np
import time
import plac
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


@plac.annotations(
    inference_type=("Type of inference to test (TF, FP32, FP16, INT8)", 'option', 'T', str),
    batch_size=("Size of the TensorRT batch", 'option', 'B', int),
    test_size=("Number of samples run the inference on", 'option', 'S', int),
    input_dims=("Comma seperate input dimensions ie 224, 224, 3", 'option', 'D', str),
    weights_path=("Model weights", 'positional', None, str),
    test_path=("Test images path", 'option', 'I', str),
    merge=("Test images only: Merge detected regions", 'flag', 'M', bool),
    stage=("Test images only: Augmentation training stage", 'option', 's', str),
    limit=("Test images only: Max number of images to run inference on", 'option', 'l', int),
    confidence=("Test images only: Minimum confidence in coverage to draw bbox", "option", "c", float),
    feature_upsample=("", "option", "u", int)
)
def main(inference_type: str = "K",
         batch_size: int = 1,
         test_size: int = 1,
         test_path: str = None,
         input_dims: str = "224, 224, 3",
         weights_path: str = "mobiledetectnet.h5",
         merge: bool = False,
         stage: str = "test",
         limit: int = 20,
         confidence: float = 0.1,
         feature_upsample: int = 1):

    model, coverage_shape = MobileDetectNetModel.create(weights=None, feature_upsample=feature_upsample)
    model.load_weights(weights_path)

    test_dims = [int(d) for d in input_dims.split(",")]

    images_full = None
    images_input = None

    images_done = 0

    if test_path is None:
        test_dims.insert(0, test_size)

        x_test = np.random.random(test_dims)
    else:

        from train import MobileDetectNetSequence
        import cv2
        import matplotlib.pyplot as plt

        seq = MobileDetectNetSequence.create_augmenter(stage)

        images_full = []
        images_input = []

        for r, d, f in os.walk(test_path):
            for file in f:

                #for i in range(0, limit):
                if True:

                    image_full = cv2.imread(os.path.join(r, file))
                    image_input = cv2.resize(image_full, (test_dims[0], test_dims[1]))

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
        coverage, bboxes = model.predict(x_test)
        t1 = time.time()
    elif inference_type == 'TF':
        tf_engine = model.tf_engine()
        t0 = time.time()
        coverage, bboxes = tf_engine.infer(x_test)
        t1 = time.time()
    elif inference_type == 'FP32':
        tftrt_engine = model.tftrt_engine(precision='FP32', batch_size=batch_size)
        t0 = time.time()
        coverage, bboxes = tftrt_engine.infer(x_test)
        t1 = time.time()
    elif inference_type == 'FP16':
        tftrt_engine = model.tftrt_engine(precision='FP16', batch_size=batch_size)
        t0 = time.time()
        coverage, bboxes = tftrt_engine.infer(x_test)
        t1 = time.time()
    elif inference_type == 'INT8':
        tftrt_engine = model.tftrt_engine(precision='INT8', batch_size=batch_size)
        t0 = time.time()
        coverage, bboxes = tftrt_engine.infer(x_test)
        t1 = time.time()
    else:
        raise Exception("Invalid inference_type")

    print('Time', t1 - t0)

    if images_full is not None:
        for idx in range(0, len(images_full)):

            rectangles = []
            for y in range(0, coverage_shape[0]):
                for x in range(0, coverage_shape[1]):

                    if coverage[idx, y, x] > confidence:
                        rect = [
                            int(bboxes[idx, int(y / feature_upsample), int(x / feature_upsample), 0] * test_dims[1]),
                            int(bboxes[idx, int(y / feature_upsample), int(x / feature_upsample), 1] * test_dims[0]),
                            int(bboxes[idx, int(y / feature_upsample), int(x / feature_upsample), 2] * test_dims[1]),
                            int(bboxes[idx, int(y / feature_upsample), int(x / feature_upsample), 3] * test_dims[0])]

                        rectangles.append(rect)

            if merge:
                rectangles, merges = cv2.groupRectangles(rectangles, 1, eps=0.3)

            for rect in rectangles:
                cv2.rectangle(images_input[idx],
                              (rect[0], rect[1]),
                              (rect[2], rect[3]),
                              (0, 1, 0), 3)

            plt.imshow((images_input[idx] + 1) / 2, alpha=1.0)
            plt.imshow(
                cv2.resize(coverage[idx].reshape((coverage_shape[0], coverage_shape[1])), (x_test.shape[1], x_test.shape[2])),
                interpolation='nearest', alpha=0.5)
            plt.show()


if __name__ == '__main__':
    plac.call(main)
