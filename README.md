# Keras MobileDetectNet

MobileDetectNet is an object detector which uses [MobileNet][mobilenet] feature extractor to predict a coverage map and bounding boxes. It was designed to be computationally efficient for deployment on embedded systems and easy to train with limited data.

### Network Arcitecture
```
        # Input: scaled from -1, to 1 to take advantage of transfer learning
        mobilenet = keras.applications.mobilenet.MobileNet(include_top=False,
                                                           input_shape=(224, 224, 3),
                                                           weights='imagenet',
                                                           alpha=0.25)

        # Last layer of MobileNet
        new_output = mobilenet.get_layer('conv_pw_13_relu').output

        # Add detection network
        coverage = Conv2D(1, 1, activation='sigmoid', name='coverage')(new_output)
        flatten = Flatten()(coverage)
        bboxes_preshape = Dense(7*7*4, activation='linear', name='bboxes_preshape')(flatten)
        bboxes = Reshape((7, 7, 4), name='bboxes')(bboxes_preshape)
        
        mobiledetectnet = Model(inputs=mobilenet.input, outputs=[coverage, bboxes])
        mobiledetectnet.compile(optimizer=opt, loss=['mean_absolute_error', 'mean_absolute_error'])
```

### Training
Training is done with [imgaug][imgaug] utilizing Keras [Sequences][sequence] for multicore preprocessing and online data augmentation:

```
iaa.Sequential([
                iaa.Fliplr(0.5),
                iaa.CropAndPad(px=(0, 112), sample_independently=False),
                iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}),
                iaa.SomeOf((0, 3), [
                    iaa.AddToHueAndSaturation((-10, 10)),
                    iaa.Affine(scale={"x": (0.9, 1.1), "y": (0.9, 1.1)}),
                    iaa.GaussianBlur(sigma=(0, 1.0)),
                    iaa.AdditiveGaussianNoise(scale=0.1 * 255)
                ])
            ])
```

Data augmentation is also used for validation for the purpose of making sure smaller objects are detected. Validation images are shrunk up to 50% and shifted left/right up to 20%. If a dataset contains many smaller bounding boxes or detecting smaller objects is not a concern, this should be adjusted for both train and validation augmentation.

[SGD with Warm Restarts][sgdr] seems to converge effectively for the application.

### Label Format
MobileDetectNet uses the KITTI label format and directory structure. See [here for more details][kitti]

[mobilenet]: https://arxiv.org/abs/1704.04861
[imgaug]: https://github.com/aleju/imgaug
[sequence]: https://keras.io/utils/
[sgdr]: https://arxiv.org/abs/1608.03983
[kitti]: https://github.com/NVIDIA/DIGITS/tree/master/digits/extensions/data/objectDetection