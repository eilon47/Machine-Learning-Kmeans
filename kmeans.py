import numpy as np
from init_centroids import init_centroids

EPOCH = 10
PIX_FORMAT = "[{}, {}, {}]"


def format_centroid(cent):
    """
    formatting the centroid to a string 
    :param cent: 
    :return: 
    """""
    x,y,z = np.floor(cent[0] * 100) / 100, np.floor(cent[1] * 100) / 100, \
            np.floor(cent[2] * 100) / 100
    return PIX_FORMAT.format(str(x),str(y),str(z))


def distance(p1, p2):
    """
    euclidean distance calculator
    :param p1:
    :param p2:
    :return:
    """
    return np.linalg.norm(p1-p2)


def choose_centroid(centroids, pixel):
    """
    choosing centroid from possible centroids that is the closest to the pixel according to a given distance function.
    :param centroids:
    :param pixel:
    :return:
    """
    min_dist = np.inf
    centroid_index = -1
    for i in range(len(centroids)):
        c = centroids[i]
        dist = distance(c, pixel)
        if dist < min_dist:
            min_dist = dist
            centroid_index = i
    return centroids[centroid_index], centroid_index


def train(image, k):
    """
    training the image by finding the best centroids, then creating the new image with those centroids
    :param image: data
    :param k: number of centroids
    :return:
    """
    print("k={}:".format(k))
    data_len = len(image)
    centroids = init_centroids(image, k)
    loss = []
    for epoch in range(EPOCH+1):
        print("iter {}: {}".format(epoch, ", ".join([format_centroid(c) for c in centroids])))
        sums = [np.zeros(3)] * k
        nums = [0] * k
        iter_loss = 0
        for index, pixel in enumerate(image):
            centroid, i = choose_centroid(centroids, pixel)
            nums[i] += 1
            sums[i] = sums[i] + pixel
            iter_loss += np.power(distance(pixel, centroid),2)
        centroids = [s/n for s,n in zip(sums,nums)]
        loss.append(iter_loss/data_len)
    new_img = []
    for i,pix in enumerate(image):
        centroid = list(choose_centroid(centroids, pix))[0]
        new_img.append(centroid)

    return np.asarray(new_img), loss


