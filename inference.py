from model import MobileDetectnetModel

import numpy as np
import time
import plac


@plac.annotations(
    inference_type=("Type of inference to test (TF, FP32, FP16, INT8)", 'option', 'T', str),
    batch_size=("Size of the TensorRT batch", 'option', 'B', int),
    test_size=("Number of samples run the inference on", 'option', 'S', int),
    input_dims=("Comma seperate input dimensions ie 224, 224, 3", 'option', 'D', str),
    weights_path=("Model weights", 'positional', None, str)
)
def main(inference_type: str = "FP16",
         batch_size: int = 1,
         test_size: int = 1,
         input_dims: str = "224, 224, 3",
         weights_path: str = "mobiledetectnet.h5"):

    model = MobileDetectnetModel.create(weights=None)
    model.load_weights(weights_path)

    test_dims = [int(d) for d in input_dims.split(",")]
    test_dims.insert(0, test_size)

    x_test = np.random.random(test_dims)

    if inference_type == 'TF':
        tf_engine = model.tf_engine()
        t0 = time.time()
        y_tf = tf_engine.infer(x_test)
        t1 = time.time()

    elif inference_type == 'FP32':
        tftrt_engine = model.tftrt_engine(precision='FP32', batch_size=batch_size)
        t0 = time.time()
        y_tftrt = tftrt_engine.infer(x_test)
        t1 = time.time()
    elif inference_type == 'FP16':
        tftrt_engine = model.tftrt_engine(precision='FP16', batch_size=batch_size)
        t0 = time.time()
        y_tftrt = tftrt_engine.infer(x_test)
        t1 = time.time()
    elif inference_type == 'INT8':
        tftrt_engine = model.tftrt_engine(precision='INT8', batch_size=batch_size)
        t0 = time.time()
        y_tftrt = tftrt_engine.infer(x_test)
        t1 = time.time()
    else:
        raise Exception("Invalid inference_type")

    print('Time', t1 - t0)


if __name__ == '__main__':
    plac.call(main)
