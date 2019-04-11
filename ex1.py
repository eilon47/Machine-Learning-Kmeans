from optparse import OptionParser
import matplotlib.pyplot as plt
from scipy.misc import toimage, imread
from kmeans import train

option_parser = OptionParser()
option_parser.add_option("-p", "--plot", dest="plot", action="store_true", help="save plots", default=False)
option_parser.add_option("-i", "--img", dest="img", action="store_true", help="save images", default=False)


def load_image(path):
    """
    loading image in a specific path
    """
    A = imread(path)
    A = A.astype(float) / 255.
    img_size = A.shape
    X = A.reshape(img_size[0] * img_size[1], img_size[2])
    return X, img_size


def resize_image(image, oldsize):
    """
    resizing image to its old size
    :param image:
    :param oldsize:
    :return:
    """
    return image.reshape(oldsize)


def main():
    options, args = option_parser.parse_args()
    image,old_size = load_image("dog.jpeg")
    for k in [2,4,8,16]:
        new_img, loss = train(image, k)
        plt.plot(range(len(loss)), loss)
        if options.plot:
            plt.savefig("plots/plot-{}.jpeg".format(k))
        if options.img:
            resized = resize_image(new_img, old_size)
            toimage(resized).save("images/img-{}.jpeg".format(k))
        plt.clf()


if __name__ == '__main__':
    main()


