# Picnic-Image-Classifier

- Clean the dataset by removing noisy images and adding few images for more support in a case where the model was performing worse
- Data Augmentation: Created more set of images by flipping, rotating, flipping and adding Gaussian noise to the image to create more data for the CNN to learn
- Use the pre-trained model trained on Imagenet dataset to fine-tune on the above dataset to produce a better result.
-  Construct a validation set from the training images (about 10%)
- to further increase the accuracy on the test set, use the average ensemble method consisting of InceptionV3, InceptionResnetV2 and Xception
