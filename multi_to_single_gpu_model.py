import plac
from model import MobileDetectNetModel
import tensorflow.keras as keras
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


@plac.annotations(
    input_weights_path=('Path to the weights file', 'positional', None, str),
    output_weights_path=('Path to the single GPU weights file', 'option', 'O', str),

)
def main(input_weights_path: str,
         output_weights_path: str = None):

    keras_model = MobileDetectNetModel.complete_model()
    model_index = -4

    keras_model = keras.utils.multi_gpu_model(keras_model, gpus=[0, 1], cpu_merge=True, cpu_relocation=False)
    keras_model.summary()
    keras_model.load_weights(input_weights_path)

    single_gpu_model = keras_model.layers[model_index]

    output_weights_path = "mobiledetectnet.hdf5" if output_weights_path is None else output_weights_path

    single_gpu_model.save_weights(output_weights_path)


if __name__ == '__main__':
    plac.call(main)
