import plac
from model import MobileDetectNetModel
import tensorflow.keras as keras
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


@plac.annotations(
    input_weights_path=('Path to the weights file', 'positional', None, str),
    output_weights_path=('Path to the single GPU weights file', 'option', 'O', str),
    model_index=('Index of the layer which contains the single GPU model', 'option', 'I', int),
    model=('Which model architecture to train (complete, coverage, regions, pooling)', 'option', 'M', str)

)
def main(input_weights_path: str,
         output_weights_path: str = 'mobiledetectnet.h5',
         model_index: int = -4,
         model="complete"):

    if model is None or model == "complete":
        keras_model = MobileDetectNetModel.complete_model()
        raise Exception("Not implemented yet!")
    elif model == "coverage":
        keras_model = MobileDetectNetModel.coverage_model()
    elif model == "region":
        keras_model = MobileDetectNetModel.region_model()
        raise Exception("Not implemented yet!")
    elif model == "pooling":
        keras_model = MobileDetectNetModel.pooling_model()
        raise Exception("Not implemented yet!")
    else:
        raise Exception("Invalid mode: %s" % model)

    keras_model = keras.utils.multi_gpu_model(keras_model, gpus=[0, 1], cpu_merge=True, cpu_relocation=False)
    keras_model.summary()
    keras_model.load_weights(input_weights_path)

    single_gpu_model = keras_model.layers[model_index]
    single_gpu_model.save_weights(output_weights_path)


if __name__ == '__main__':
    plac.call(main)
