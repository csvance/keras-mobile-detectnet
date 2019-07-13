import numpy as np
import time
import plac
import os

from model import MobileDetectNetModel

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


@plac.annotations(
    inference_type=("Type of inference to test (TF, FP32, FP16, INT8)", 'option', 'T', str),
    batch_size=("Size of the TensorRT batch", 'option', 'B', int),
    weights=("Model weights", 'option', 'W', str),
    test_path=("Test images path", 'option', 'I', str),
    merge=("Test images only: Merge detected regions", 'flag', 'm', bool),
    stage=("Test images only: Augmentation training stage", 'option', 's', str),
    limit=("Test images only: Max number of images to run inference on", 'option', 'l', int),
    confidence=("Test images only: Minimum confidence in coverage to draw bbox", "option", "c", float),
    visualize=("Visualize the inference", "option", "V", bool)
)
def main(inference_type: str = "K",
         batch_size: int = 1,
         test_path: str = None,
         weights: str = None,
         merge: bool = False,
         stage: str = "test",
         limit: int = 20,
         confidence: float = 0.1,
         visualize: bool = True):

    keras_model = MobileDetectNetModel.complete_model()

    if weights is not None:
        keras_model.load_weights(weights, by_name=True)

    images_done = 0

    if test_path is not None:
        import cv2

        if stage != 'test':
            from generator import MobileDetectNetSequence
            seq = MobileDetectNetSequence.create_augmenter(stage)
        else:
            seq = None

        images_full = []
        images_input = []
        images_scale = []

        for r, d, f in os.walk(test_path):
            for file in f:
                image_full = cv2.imread(os.path.join(r, file))
                image_input = cv2.resize(image_full, (224, 224))

                scale_width = image_full.shape[1] / 224
                scale_height = image_full.shape[0] / 224
                images_scale.append((scale_width, scale_height))

                if stage != 'test':
                    seq_det = seq.to_deterministic()
                    image_aug = (seq_det.augment_image(image_input).astype(np.float32) / 127.5) - 1.
                else:
                    image_aug = image_input.astype(np.float32) / 127.5 - 1.

                images_full.append(image_full)
                images_input.append(image_aug)

                images_done += 1

                if images_done == limit:
                    break

            if images_done == limit:
                break

        x_test = np.array(images_input)
    else:
        x_test = np.random.random((limit, 224, 224, 3))

    x_cold = np.random.random((batch_size, 224, 224, 3))

    if inference_type == 'K':
        keras_model.predict(x_cold)
        t0 = time.time()
        model_outputs = keras_model.predict(x_test)
        t1 = time.time()
    elif inference_type == 'TF':
        tf_engine = keras_model.tf_engine()
        tf_engine.infer(x_cold)
        t0 = time.time()
        model_outputs = tf_engine.infer(x_test)
        t1 = time.time()
    elif inference_type == 'FP32':
        tftrt_engine = keras_model.tftrt_engine(precision='FP32', batch_size=batch_size)
        tftrt_engine.infer(x_cold)
        t0 = time.time()
        model_outputs = tftrt_engine.infer(x_test)
        t1 = time.time()
    elif inference_type == 'FP16':
        tftrt_engine = keras_model.tftrt_engine(precision='FP16', batch_size=batch_size)
        tftrt_engine.infer(x_cold)
        t0 = time.time()
        model_outputs = tftrt_engine.infer(x_test)
        t1 = time.time()
    elif inference_type == 'INT8':
        tftrt_engine = keras_model.tftrt_engine(precision='INT8', batch_size=batch_size)
        tftrt_engine.infer(x_cold)
        t0 = time.time()
        model_outputs = tftrt_engine.infer(x_test)
        t1 = time.time()
    else:
        raise ValueError("Invalid inference type")

    print('Time: ', t1 - t0)
    print('FPS: ', x_test.shape[0]/(t1 - t0))

    if not visualize:
        return

    if len(model_outputs) == 2:
        classes, bboxes = model_outputs

    # TF / TensorRT models won't output regions (not useful for production)
    elif len(model_outputs) == 3:
        regions, bboxes, classes = model_outputs
    else:
        raise ValueError("Invalid model length output")

    if test_path is not None:
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap

        # get colormap
        ncolors = 256
        color_array = plt.get_cmap('viridis')(range(ncolors))

        # change alpha values
        color_array[:, -1] = np.linspace(0.0, 1.0, ncolors)

        # create a colormap object
        map_object = LinearSegmentedColormap.from_list(name='viridis_alpha', colors=color_array)

        # register this new colormap with matplotlib
        plt.register_cmap(cmap=map_object)

        for idx in range(0, len(images_full)):

            rectangles = []
            for y in range(0, 7):
                for x in range(0, 7):

                    if classes[idx, y, x, 0] >= confidence:
                        rect = [
                            int(bboxes[idx, int(y), int(x), 0] * 224),
                            int(bboxes[idx, int(y), int(x), 1] * 224),
                            int(bboxes[idx, int(y), int(x), 2] * 224),
                            int(bboxes[idx, int(y), int(x), 3] * 224)]
                        rectangles.append(rect)

            if merge:
                rectangles, merges = cv2.groupRectangles(rectangles, 1, eps=0.75)

            scale_width, scale_height = images_scale[idx]

            for rect in rectangles:
                cv2.rectangle(images_full[idx],
                              (int(rect[0]*scale_width), int(rect[1]*scale_height)),
                              (int(rect[2]*scale_width), int(rect[3]*scale_height)),
                              (0, 255, 0), 5)

            plt.imshow(cv2.cvtColor(images_full[idx], cv2.COLOR_BGR2RGB), alpha=1.0, aspect='auto')
            plt.imshow(
                cv2.resize(classes[idx].reshape((7, 7)),
                           (images_full[idx].shape[1], images_full[idx].shape[0])),
                interpolation='nearest', alpha=0.5, cmap='viridis_alpha', aspect='auto')
            plt.show()


if __name__ == '__main__':
    plac.call(main)
