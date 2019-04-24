import plac
from model import MobileDetectNetModel
import tensorflow.keras as keras
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


@plac.annotations(
    input_weights_path=('Path to the weights file', 'positional', None, str),
    output_weights_path=('Path to the single GPU weights file', 'option', 'O', str),
    model_index=('Index of the layer which contains the single GPU model', 'option', 'I', int)
)
def main(input_weights_path: str,
         output_weights_path: str = 'mobiledetectnet.h5',
         model_index: int = -3):

    mobiledetectnet = MobileDetectNetModel.create()
    mobiledetectnet = keras.utils.multi_gpu_model(mobiledetectnet, gpus=[0, 1], cpu_merge=True, cpu_relocation=False)
    mobiledetectnet.summary()
    mobiledetectnet.load_weights(input_weights_path)

    single_gpu_model = mobiledetectnet.layers[model_index]
    single_gpu_model.save_weights(output_weights_path)


if __name__ == '__main__':
    plac.call(main)
