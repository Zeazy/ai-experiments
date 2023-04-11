"""
Experimenting with MNIST dataset using fastai
Note - Code below not finished yet
"""
from fastai.losses import CrossEntropyLossFlat
from fastai.vision.all import *
import numpy as np  # linear algebra
import struct
from array import array
from os.path import join
import matplotlib.pyplot as plt
import os


input_path = os.getcwd() + "/datasets/MNIST"
training_images_filepath = join(
    input_path, "train-images-idx3-ubyte/train-images-idx3-ubyte"
)
training_labels_filepath = join(
    input_path, "train-labels-idx1-ubyte/train-labels-idx1-ubyte"
)
test_images_filepath = join(input_path, "t10k-images-idx3-ubyte/t10k-images-idx3-ubyte")
test_labels_filepath = join(input_path, "t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte")

class MnistDataloader(object):
    """
    MNIST data loader class
    """

    def __init__(
        self,
        training_images_filepath,
        training_labels_filepath,
        test_images_filepath,
        test_labels_filepath,
    ):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath

    def read_images_labels(self, images_filepath, labels_filepath):
        """
        Read MNIST images and labels from specified filepaths
        images_filepath: path to image data file
        labels_filepath: path to label data file
        """
        labels = []
        with open(labels_filepath, "rb") as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError(
                    "Magic number mismatch, expected 2049, got {}".format(magic)
                )
            labels = array("B", file.read())

        with open(images_filepath, "rb") as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError(
                    "Magic number mismatch, expected 2051, got {}".format(magic)
                )
            image_data = array("B", file.read())
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols : (i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img

        return images, labels

    def load_data(self):
        """
        Load MNIST data
        """
        x_train, y_train = self.read_images_labels(
            self.training_images_filepath, self.training_labels_filepath
        )
        x_test, y_test = self.read_images_labels(
            self.test_images_filepath, self.test_labels_filepath
        )
        return (x_train, y_train), (x_test, y_test)
    
    def transform_data(self, data: tuple):
        """
        Transform MNIST data into pytorch tensors
        data: a tuple of (x, y) where x is a list of images and y is a list of labels
        """
        x, y = data
        x = np.array(x)[:, np.newaxis, :, :] / 255.0  # Convert list of ndarrays to a single ndarray and normalize
        x = np.repeat(x, 3, axis=1)        
        x = torch.tensor(x, dtype=torch.float32)     # Convert the ndarray to a tensor
        y = torch.tensor(y, dtype=torch.long)        # Convert the labels to a tensor
        return list(zip(x, y))

def show_images(images, title_texts):
    """
    Show a list of images in a grid
    images: a list of images
    title_texts: a list of title texts
    """
    cols = 5
    rows = int(len(images) / cols) + 1
    plt.figure(figsize=(30, 20))
    index = 1
    for x in zip(images, title_texts):
        image = x[0]
        title_text = x[1]
        plt.subplot(rows, cols, index)
        plt.imshow(image, cmap=plt.cm.gray)
        if title_text != "":
            plt.title(title_text, fontsize=15)
        index += 1
    plt.show()


def get_dls(path, bs=32, size=192):
    """
    Returns a DataLoaders object for the given path
    """
    return DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        splitter=RandomSplitter(valid_pct=0.2, seed=42),
        get_y=parent_label,
        item_tfms=[Resize(size, method="squish")],
    ).dataloaders(path, bs=bs)
    
    
def learn(dls, model_name: str, save: bool = True):
    """
    Returns the pre-trained resnet18 fine-tuned on the given dataset
    """
    model = vision_learner(dls, resnet18, metrics=error_rate, loss_func=CrossEntropyLossFlat())
    model.fine_tune(1)
    # if save:
    #     model.export(f"model/{model_name}.pkl")
    if save:
        torch.save(model.model.state_dict(), f"models/{model_name}.pth")
    print(f"Finished fine-tuning. Error rate: {model.validate()}")
    return model

def get_dls(train_data, test_data, bs=64, n_out=10):
    dls = DataLoaders.from_dsets(train_data, test_data, bs=bs, shuffle=True, drop_last=True)
    dls.c = n_out # set the number of output classes (10 for MNIST)
    return dls

def main():
    """
    1. Instantiate the MNIST data loader
    2. Load the MNIST dataset
    3. Create a DataLoaders object for the MNIST dataset
    4. Fine-tune a pre-trained resnet18 model on the MNIST dataset
    5. Save the fine-tuned model
    """
    mnist_dataloader = MnistDataloader(
    training_images_filepath,
    training_labels_filepath,
    test_images_filepath,
    test_labels_filepath,
    )
    (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()
    train_data = mnist_dataloader.transform_data((x_train, y_train))
    test_data = mnist_dataloader.transform_data((x_test, y_test))
    
    dls = get_dls(train_data, test_data)
    learn(dls, "resnet18-fine-tuned-on-mnist")
    
if __name__ == "__main__":
    main()
