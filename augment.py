import cv2
import plac
import matplotlib.pyplot as pyplot

from train import MobileDetectNetSequence
from model import MobileDetectNetModel


@plac.annotations(
    test_path=("Path to images and labels folder", "positional", None, str),
    stage=("Augmentation stage", "option", "S", str),
    num_images=("Number of images to test", "option", "I", str),
    feature_upsample=("", "option", "U", int)
)
def main(test_path: str = "test",
         stage: str = "train",
         num_images: int = 100,
         feature_upsample: int = 2):

    mobiledetectnet, coverage_shape = MobileDetectNetModel.create(feature_upsample=feature_upsample)
    bboxes_shape = [int(d/feature_upsample) for d in coverage_shape]

    generator = MobileDetectNetSequence(test_path, stage=stage, batch_size=num_images,
                                        coverage_width=coverage_shape[1], coverage_height=coverage_shape[0],
                                        bboxes_width=bboxes_shape[1], bboxes_height=bboxes_shape[0],
                                        feature_upsample=feature_upsample)

    images, labels = generator.__getitem__(0)

    coverage = labels[0]
    bboxes = labels[1]

    for i in range(0, num_images):

        # Normalize an image scaled between [-1, 1] to [0, 1]
        image = (images[i] + 1) / 2

        for y in range(0, coverage_shape[0]):
            for x in range(0, coverage_shape[1]):
                if coverage[i, y, x] > 0:
                    cv2.rectangle(image,
                                  (int(bboxes[i, int(y/feature_upsample), int(x/feature_upsample), 0]*224), int(bboxes[i, int(y/feature_upsample), int(x/feature_upsample), 1]*224)),
                                  (int(bboxes[i, int(y/feature_upsample), int(x/feature_upsample), 2]*224), int(bboxes[i, int(y/feature_upsample), int(x/feature_upsample), 3]*224)),
                                  (0, 1, 0), 2)

        pyplot.imshow(image, alpha=1.0)
        pyplot.imshow(cv2.resize(coverage[i], (224, 224), interpolation=cv2.INTER_AREA), alpha=0.5)
        pyplot.show()


if __name__ == '__main__':
    plac.call(main)
