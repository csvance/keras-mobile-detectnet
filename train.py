import cv2
import numpy as np
import os
import plac

import imgaug as ia
from imgaug import augmenters as iaa

import tensorflow.keras as keras
from tensorflow.keras.losses import mean_absolute_error, mean_squared_error
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import Sequence

from model import MobileDetectnetModel


class MobileDetectnetSequence(Sequence):
    def __init__(self,
                 path: str,
                 augment: bool = True,
                 batch_size: int = 12,
                 resize_width: int = 224,
                 resize_height: int = 224,
                 coverage_width: int = 7,
                 coverage_height: int = 7,
                 ):

        self.images = []
        self.labels = []

        for r, d, f in os.walk("%s/images" % path):
            for file in f:
                self.images.append(os.path.join(r, file))
                self.labels.append(os.path.join("%s/labels" % path, (file.split(".")[0] + ".txt")))

        self.batch_size = batch_size
        self.augment = augment
        self.resize_width = resize_width
        self.resize_height = resize_height
        self.coverage_width = coverage_width
        self.coverage_height = coverage_height

        if augment:
            self.seq = iaa.Sequential([
                iaa.Sometimes(0.5, iaa.Fliplr(1.0)),
                iaa.SomeOf((0, 4), [
                    iaa.AddToHueAndSaturation((-20, 20)),
                    iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}),
                    iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}),
                    iaa.GaussianBlur(sigma=(0, 3.0))
                ])
            ])
        else:
            self.seq = iaa.Sequential([])

    def __len__(self):
        return int(np.floor(len(self.images) / float(self.batch_size)))

    def __getitem__(self, idx):
        input_image = np.zeros((self.batch_size, self.resize_height, self.resize_width, 3))
        output_coverage_map = np.zeros((self.batch_size, self.coverage_height, self.coverage_width))
        output_bboxes = np.zeros((self.batch_size, self.coverage_height, self.coverage_width, 4))

        for i in range(0, self.batch_size):

            seq_det = self.seq.to_deterministic()

            image = cv2.imread(self.images[idx * self.batch_size + i])
            old_shape = image.shape
            image = cv2.resize(image, (self.resize_height, self.resize_width))

            bboxes, segmap = MobileDetectnetSequence.load_kitti_label(image,
                                                                      scale=(image.shape[0] / old_shape[0],
                                                                             image.shape[1] / old_shape[1]),
                                                                      label=self.labels[idx * self.batch_size + i])

            image_aug = seq_det.augment_image(image)

            bboxes_aug = seq_det.augment_bounding_boxes(bboxes)
            segmap_aug = seq_det.augment_segmentation_maps(segmap)

            output_segmap = segmap_aug.arr.astype(np.float32)
            output_segmap = cv2.resize(output_segmap, (self.coverage_height, self.coverage_width),
                                       interpolation=cv2.INTER_AREA).astype(np.float32)[:, :, 1]

            # Work on building a batch
            input_image[i] = image_aug.astype(np.float32) / 255
            output_coverage_map[i] = output_segmap

            # Put each predicted bbox in its center
            for bbox in bboxes_aug.bounding_boxes:
                center_x = min(int(np.floor(bbox.center_x / self.resize_width * self.coverage_width)),
                               self.coverage_width - 1)
                center_y = min(int(np.floor(bbox.center_y / self.resize_height * self.coverage_height)),
                               self.coverage_height - 1)

                output_bboxes[i, center_y, center_x, 0] = bbox.x1 / self.resize_width
                output_bboxes[i, center_y, center_x, 1] = bbox.y1 / self.resize_height
                output_bboxes[i, center_y, center_x, 2] = bbox.x2 / self.resize_width
                output_bboxes[i, center_y, center_x, 3] = bbox.y2 / self.resize_height

        return input_image, [
            output_coverage_map.reshape((self.batch_size, self.coverage_height, self.coverage_width, 1)),
            output_bboxes]

    @staticmethod
    # KITTI Format Labels
    def load_kitti_label(image: np.ndarray, scale, label: str):
        label = open(label, 'r').read()

        bboxes = []

        segmap = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

        for row in label.split('\n'):
            fields = row.split(' ')

            bbox_class = fields[0]
            bbox_truncated = float(fields[1])
            bbox_occluded = int(fields[2])
            bbox_alpha = float(fields[3])
            bbox_x1 = float(fields[4]) * scale[1]
            bbox_y1 = float(fields[5]) * scale[0]
            bbox_x2 = float(fields[6]) * scale[1]
            bbox_y2 = float(fields[7]) * scale[0]

            polygon = ia.Polygon(
                np.float32([[bbox_x1, bbox_y1], [bbox_x1, bbox_y2], [bbox_x2, bbox_y2], [bbox_x2, bbox_y1]]))
            segmap = polygon.draw_on_image(segmap, alpha=1.0, alpha_perimeter=0.0)

            bbox = ia.BoundingBox(bbox_x1, bbox_y1, bbox_x2, bbox_y2, bbox_class)
            bboxes.append(bbox)

        segmap = np.argmax(segmap, axis=2)

        bboi = ia.BoundingBoxesOnImage(bboxes, shape=image.shape)
        smoi = ia.SegmentationMapOnImage(segmap, shape=image.shape, nb_classes=2)

        return bboi, smoi


@plac.annotations(
    batch_size=('The training batch size', 'option', 'B', int),
    epochs=('Number of epochs to train', 'option', 'E', int),
    train_path=('Path to the train folder which contains both an images and labels folder with KITTI labels', 'option', 'T', str),
    val_path=('Path to the validation folder which contains both an images and labels folder with KITTI labels', 'option', 'V', str),
    metric=('Loss metric to minimize', 'option', 'L', str),
    weights=('Weights file to start with', 'option', 'W', str)
)
def main(batch_size: int = 24,
         epochs: int = 500,
         train_path: str = 'train',
         val_path: str = 'val',
         metric='val_bboxes_loss',
         weights=None):

    mobiledetectnet = MobileDetectnetModel.create()

    if weights is not None:
        mobiledetectnet.load_weights(weights)

    mobiledetectnet = keras.utils.multi_gpu_model(mobiledetectnet, gpus=[0, 1], cpu_merge=True, cpu_relocation=False)
    mobiledetectnet.compile(optimizer=Adam(lr=0.0001, decay=0.000001),
                            loss=[mean_squared_error, mean_absolute_error])

    train_seq = MobileDetectnetSequence(train_path, augment=True, batch_size=batch_size)
    val_seq = MobileDetectnetSequence(val_path, augment=False, batch_size=batch_size)

    filepath = "weights-improvement-{epoch:02d}-{%s:.4f}.hdf5" % metric
    checkpoint = ModelCheckpoint(filepath, monitor=metric, verbose=1, save_best_only=True, mode='min')

    mobiledetectnet.fit_generator(train_seq,
                                            validation_data=val_seq,
                                            epochs=epochs,
                                            steps_per_epoch=len(train_seq),
                                            validation_steps=len(val_seq),
                                            callbacks=[checkpoint],
                                            use_multiprocessing=True, workers=8)


if __name__ == '__main__':
    plac.call(main)
