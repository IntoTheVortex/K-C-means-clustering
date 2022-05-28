import sys
import os.path
import numpy as np
from matplotlib import pyplot as plt

NUM_CLUSTERS = 2
LIMIT = 200
INTERVAL = 10

#TODO sum of squares error
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


    counter = 0
    prev_clusters = np.ones((NUM_CLUSTERS, len(data), 2))
    clusters = np.zeros((NUM_CLUSTERS, len(data), 2))

    ## main loop
    while not np.array_equal(prev_clusters, clusters):
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

        clusters = label_data_cmeans(data, coeffs)

        if counter%INTERVAL == 0:
            plot_data(clusters, c, counter, 'c-means')

        for i in range(k):
        #recompute centroids
            # fuzzed weights
            coeff_fuzz = np.power(coeffs[:, i], m)

            # x-vals
            data_x = np.multiply(coeff_fuzz, data[:, 0])
            c[i][0] = np.sum(data_x) / np.sum(coeff_fuzz)

            data_y = np.multiply(coeff_fuzz, data[:, 1])
            c[i][1] = np.sum(data_y) / np.sum(coeff_fuzz)

        counter += 1
        if counter > LIMIT:
            print("limit reached")
            break



def label_data_cmeans(data, coeffs):
    clusters = np.zeros((NUM_CLUSTERS, len(data), 2))
    for i in range(len(data)):
        clusters[np.argmax(coeffs[i])][i] = (data[i])

    return clusters



#TODO sum of squares error
def kmeans(data):
     #number of clusters
    k = NUM_CLUSTERS 

    #init centroids
    max = np.max(data)
    min = np.min(data)
    c = np.random.uniform(min, max, [k, 2])

    clusters = np.zeros((NUM_CLUSTERS, len(data), 2))


    ## main loop
    counter = 0
    prev_clusters = np.ones((NUM_CLUSTERS, len(data), 2))
    clusters = np.zeros((NUM_CLUSTERS, len(data), 2))

    ## main loop
    while not np.array_equal(prev_clusters, clusters):
        #compute distances
        distances = np.zeros((len(data), k))
        for j in range(k): #j is a cluster
            # euc dist = sqrt((x1-x2)^2 + (y1-y2)^2)
            distances[:, j] = np.sqrt(
                np.power(
                    np.subtract(data[:, 0], c[j][0]), 2) 
                + np.power(
                    np.subtract(data[:, 1], c[j][1]), 2)
            )

        #assign by least distance
        clustered_data = np.zeros((NUM_CLUSTERS, len(data), 2))
        for i in range(len(data)):
            clustered_data[np.argmin(distances[i])][i] = data[i]

        #print(clustered_data)
        if counter%INTERVAL == 0:
            plot_data(clustered_data, c, counter, 'k-means')



        for i in range(k):
        #re-compute centroids
            c[i][0] = np.sum(clustered_data[i][:, 0]) / (np.count_nonzero(clustered_data[i], axis=0))[0]
            #print(clustered_data[i][:,0])
            c[i][1] = np.sum(clustered_data[i][:, 1]) / (np.count_nonzero(clustered_data[i], axis=0))[1]

        counter += 1
        if counter > LIMIT:
            print("limit reached")
            break






#plot
def plot_data(clusters, centroids, r, filename):
    plt.figure()
    colors = list("gbcmy")
    colors = ['mediumblue', 'slateblue', 'rebeccapurple', 'indigo', 
                 'darkviolet', 'cornflowerblue', 'aqua'
                 ]

    for i in range(NUM_CLUSTERS):
        color = colors[i%len(colors)]
        plt.scatter(clusters[i][:, 0], clusters[i][:, 1], color=color)

    plt.scatter(centroids[:, 0], centroids[:, 1], color='r')

    plt.savefig(filename + str(r) + '.png')
    #plt.show()
    plt.close()




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
    kmeans(data)




if __name__ == '__main__':
    main()