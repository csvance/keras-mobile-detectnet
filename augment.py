import cv2
import plac
import matplotlib.pyplot as pyplot

from train import MobileDetectnetSequence

@plac.annotations(
    test_path=("Path to images and labels folder", "positional", None, str),
    stage=("Augmentation stage", "option", "S", str),
    num_images=("Number of images to test", "option", "I", str)
)
def main(test_path: str = "test",
         stage: str = "train",
         num_images: int = 10):

    generator = MobileDetectnetSequence(test_path, stage=stage, batch_size=num_images)
    images, labels = generator.__getitem__(0)

    coverage = labels[0]
    bboxes = labels[1]

    for i in range(0, num_images):

        image = (images[i] + 1) / 2

        for y in range(0, 7):
            for x in range(0, 7):
                if coverage[i, y, x] > 0:
                    cv2.rectangle(image,
                                  (int(bboxes[i, y, x, 0]*224), int(bboxes[i, y, x, 1]*224)),
                                  (int(bboxes[i, y, x, 2]*224), int(bboxes[i, y, x, 3]*224)),
                                  (0, 1, 0), 2)

        pyplot.imshow(image, alpha=1.0)
        pyplot.imshow(cv2.resize(coverage[i], (224, 224), interpolation=cv2.INTER_AREA), alpha=0.5)
        pyplot.show()


if __name__ == '__main__':
    plac.call(main)
