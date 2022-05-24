import sys
import os.path
import numpy as np


def fcm(data):
    ## initialization
    m = 2
    #number of clusters
    k = 5

    #initial coefficients/membership grades
    coeffs = np.zeros((len(data), k))

    #init centroids
    max = np.max(data)
    min = np.min(data)

    c = np.random.uniform(min, max, [k, 2])
    #print(c)

    ## main loop

    #compute coefficients/membership grades/weights
    for j in range(k):
        # dist = sqrt((x1-x2)^2 + (y1-y2)^2)
        numerator = np.sqrt(
            np.power(
                np.subtract(data[:, 0], c[j][0]), 2) 
            + np.power(
                np.subtract(data[:, 1], c[j][1]), 2)
        )
        #print(numerator)
        #print(numerator.shape)
        
        '''trouble getting the order right here'''
        totals = np.zeros((len(data), k))
        for _k in range(k):
            #coeffs[:, j] = numerator / np.sqrt(
            totals[_k] += numerator / np.sqrt(
                np.power(
                    np.subtract(data[:, 0], c[_k][0]), 2) 
                + np.power(
                    np.subtract(data[:, 1], c[_k][1]), 2)
            )

        #print("weights shape:", coeffs[:, j].shape)
        coeffs[:, j] = 1 / np.power(totals, (2 / (m - 1)))
        print("weights shape:", coeffs[:, j].shape)

    coeffs = 0
        
        

    for i in range(k):
    #compute centroid
        # fuzzed weights
        coeff_fuzz = np.power(coeffs[:, i], m)

        # x-vals
        data_x = np.multiply(coeff_fuzz, data[:, 0])
        c[i][0] = np.sum(data_x) / np.sum(coeff_fuzz)

        data_y = np.multiply(coeff_fuzz, data[:, 1])
        c[i][1] = np.sum(data_y) / np.sum(coeff_fuzz)
    print("c", c)




def read_in_data(file_name):
    file = open(file_name, 'r')
    data = np.zeros((1500, 2))

    line = file.readline()
    i = 0
    while(line):
        #print(line)
        x_y = line.split()
        x_y = [float(x_y[0]), float(x_y[1])]
        data[i][0] = float(x_y[0])
        data[i][1] = float(x_y[1])
        line = file.readline()
        i += 1

    file.close()
    print("data", data)
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