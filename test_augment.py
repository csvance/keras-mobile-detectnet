import cv2
import plac
import matplotlib.pyplot as plt
import numpy as np

from train import MobileDetectNetSequence


@plac.annotations(
    test_path=("Path to images and labels folder", "positional", None, str),
    stage=("Augmentation stage", "option", "S", str),
    num_images=("Number of images to test", "option", "I", str),
)
def main(test_path: str = "test",
         stage: str = "train",
         num_images: int = 100):

    generator = MobileDetectNetSequence(test_path, stage=stage, batch_size=num_images)

    images, labels = generator.__getitem__(0)

    regions = labels[0]
    bboxes = labels[1]

    for i in range(0, num_images):

        bbox_count = 0

        # Normalize an image scaled between [-1, 1] to [0, 1]
        image = (images[i] + 1) / 2

        for y in range(0, 7):
            for x in range(0, 7):
                if np.max(regions[i, y, x, :]) > 0.3:
                    cv2.rectangle(image,
                              (int(bboxes[i, int(y), int(x), 0]*224), int(bboxes[i, int(y), int(x), 1]*224)),
                              (int(bboxes[i, int(y), int(x), 2]*224), int(bboxes[i, int(y), int(x), 3]*224)),
                              (0, 1, 0), 2)
                    bbox_count += 1

        plt.imshow(image, alpha=1.0)
        plt.imshow(
            cv2.resize(np.max(regions[i], axis=-1).reshape((7, 7)),
                       (224, 224)),
            interpolation='nearest', alpha=0.5)

        # plt.imshow(cv2.resize(coverage[i], (224, 224), interpolation=cv2.INTER_AREA), alpha=0.5)
        plt.show()


if __name__ == '__main__':
    plac.call(main)
