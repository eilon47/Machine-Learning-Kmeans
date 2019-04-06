import sys
import matplotlib.pyplot as plt
from scipy.misc import toimage
from kmeans import train
from load import load_image, resize_image
from optparse import OptionParser

option_parser = OptionParser()
option_parser.add_option("-p", "--plot", dest="plot", action="store_true", help="save plots", default=False)
option_parser.add_option("-i", "--img", dest="img", action="store_true", help="save images", default=False)


def main():
    options, args = option_parser.parse_args()
    image,old_size = load_image()
    for k in [2,4,8,16]:
        new_img, loss = train(image, k)
        plt.plot(range(len(loss)), loss)
        if options.plot:
            plt.savefig("plots/plot-{}.jpeg".format(k))
        if options.img:
            resized = resize_image(new_img, old_size)
            toimage(resized).save("images/img-{}.jpeg".format(k))



if __name__ == '__main__':
    main()