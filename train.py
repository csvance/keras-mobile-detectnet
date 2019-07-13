import numpy as np
import os
import plac
import time

from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.layers import Input
from tensorflow.keras import backend as K

from model import MobileDetectNetModel
from generator import MobileDetectNetSequence


@plac.annotations(
    batch_size=('The training batch size', 'option', 'B', int),
    epochs=('Number of epochs to train', 'option', 'E', int),
    train_path=(
            'Path to the train folder which contains both an images and labels folder with KITTI labels',
            'option', 'T', str),
    val_path=(
            'Path to the validation folder which contains both an images and labels folder with KITTI labels',
            'option', 'V', str),
    weights=('Weights file to start with', 'option', 'W', str),
    workers=('Number of fit_generator workers', 'option', 'w', int)
)
def main(batch_size: int = 24,
         epochs: int = 384,
         train_path: str = 'train',
         val_path: str = 'val',
         weights=None,
         workers: int = 8):

    # We use an extra input during training to discount bounding box loss when a class is not present in an image.
    discount_input = Input(shape=(7, 7), name='discount')

    keras_model = MobileDetectNetModel.complete_model(extra_inputs=[discount_input])
    keras_model.summary()

    if weights is not None:
        keras_model.load_weights(weights, by_name=True)

    train_seq = MobileDetectNetSequence(train_path, stage="train", batch_size=batch_size)
    val_seq = MobileDetectNetSequence(val_path, stage="val", batch_size=batch_size)

    callbacks = []

    def region_loss(classes):
        def loss_fn(y_true, y_pred):
            # Don't penalize bounding box errors when there is no object present
            return 10 * (classes * K.abs(y_pred[:, :, :, 0] - y_true[:, :, :, 0]) +
                         classes * K.abs(y_pred[:, :, :, 1] - y_true[:, :, :, 1]) +
                         classes * K.abs(y_pred[:, :, :, 2] - y_true[:, :, :, 2]) +
                         classes * K.abs(y_pred[:, :, :, 3] - y_true[:, :, :, 3]))

        return loss_fn

    keras_model.compile(optimizer=Nadam(lr=0.001), loss=['mean_absolute_error',
                                                         region_loss(discount_input),
                                                         'binary_crossentropy'])

    filepath = "weights-{epoch:02d}-{val_loss:.4f}-multi-gpu.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks.append(checkpoint)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001, verbose=1)
    callbacks.append(reduce_lr)

    try:
        os.mkdir('logs')
    except FileExistsError:
        pass

    tensorboard = TensorBoard(log_dir='logs/%s' % time.strftime("%Y-%m-%d_%H-%M-%S"))
    callbacks.append(tensorboard)

    keras_model.fit_generator(train_seq,
                              validation_data=val_seq,
                              epochs=epochs,
                              steps_per_epoch=np.ceil(len(train_seq) / batch_size),
                              validation_steps=np.ceil(len(val_seq) / batch_size),
                              callbacks=callbacks,
                              use_multiprocessing=True,
                              workers=workers,
                              shuffle=True)


if __name__ == '__main__':
    plac.call(main)
