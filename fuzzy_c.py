import sys
import os.path
import random
import numpy as np
from matplotlib import pyplot as plt
import scipy.cluster

NUM_CLUSTERS = 7

def fcm(data):
    ## initialization

    #fuzzifier
    m = 2
    #number of clusters
    k = NUM_CLUSTERS 

    #initial coefficients/membership grades
    coeffs = np.zeros((len(data), k))

    #init centroids
    max = np.max(data)
    min = np.min(data)
    c = np.random.uniform(min, max, [k, 2])


    r = 10
    ## main loop
    for _ in range(r):
        #compute coefficients/membership grades/weights
        for j in range(k): #j is a cluster
            # euc dist = sqrt((x1-x2)^2 + (y1-y2)^2)
            numerator = np.sqrt(
                np.power(
                    np.subtract(data[:, 0], c[j][0]), 2) 
                + np.power(
                    np.subtract(data[:, 1], c[j][1]), 2)
            )
            
            totals = np.zeros((len(data)))
            for _k in range(k): # _k is cluster again
                totals += numerator / np.sqrt(
                    np.power(
                        np.subtract(data[:, 0], c[_k][0]), 2) 
                    + np.power(
                        np.subtract(data[:, 1], c[_k][1]), 2)
                )

            coeffs[:, j] = 1 / np.power(totals, (2 / (m - 1)))

        for i in range(k):
        #compute centroids
            # fuzzed weights
            coeff_fuzz = np.power(coeffs[:, i], m)

            # x-vals
            data_x = np.multiply(coeff_fuzz, data[:, 0])
            c[i][0] = np.sum(data_x) / np.sum(coeff_fuzz)

            data_y = np.multiply(coeff_fuzz, data[:, 1])
            c[i][1] = np.sum(data_y) / np.sum(coeff_fuzz)

        clusters = label_data_cmeans(data, coeffs)
        plot_data(clusters, c, _, 'c-means')




def label_data_cmeans(data, coeffs):
    clusters = np.zeros((NUM_CLUSTERS, len(data), 2))
    for i in range(len(data)):
        clusters[np.argmax(coeffs[i])][i] = (data[i])

    return clusters

def label_data_kmeans(data, centroids):
    pass



#plot
def plot_data(clusters, centroids, r, filename):
    colors = list("gbcmky")

    for i in range(NUM_CLUSTERS):
        color = colors[i%len(colors)]
        plt.scatter(clusters[i][:, 0], clusters[i][:, 1], color=color)

    plt.scatter(centroids[:, 0], centroids[:, 1], color='r')

    plt.savefig(filename + str(r) + '.png')
    plt.show()




def read_in_data(file_name):
    file = open(file_name, 'r')
    data = np.zeros((1500, 2))

    line = file.readline()
    i = 0
    while(line):
        x_y = line.split()
        x_y = [float(x_y[0]), float(x_y[1])]
        data[i][0] = float(x_y[0])
        data[i][1] = float(x_y[1])
        line = file.readline()
        i += 1

    file.close()
    return data


def main():
    if len(sys.argv) > 1:
        datafile = sys.argv[1]
        if not os.path.isfile(datafile):
            print("file error")
            sys.exit(1)
    else:
        print("provide a data file")
        sys.exit(1)

    data = read_in_data(datafile)
    fcm(data)




if __name__ == '__main__':
    main()