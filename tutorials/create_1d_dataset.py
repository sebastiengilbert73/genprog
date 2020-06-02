import numpy
import math

def gaussian(x, mu, sigma):
    return numpy.exp( -((x - mu)**2)/(2 * sigma**2))

def function(x):
    f1 = gaussian(x, 0.2, 0.3)
    f2 = gaussian(x, 0.5, 0.1)
    f3 = gaussian(x, 0.9, 0.4)
    return 0.4 * f1 - 0.5 * f2 + 0.1 * f3

def main():
    print ("create_1d_dataset.py main()")
    numberOfSamples = 2000
    noiseAmplitude = 0.02
    outputFilepath = './data/samples_1d.csv'

    rng = numpy.random.default_rng()
    xs = rng.random(numberOfSamples)
    ys = [function(x) for x in xs]
    noises = rng.normal(0, noiseAmplitude, numberOfSamples)
    ys += noises
    xys = list(zip(xs, ys))

    # Write to a file
    with open(outputFilepath, "w") as outputFile:
        outputFile.write('x,y\n')
        for (x, y) in xys:
            outputFile.write("{},{}\n".format(x, y))



if __name__ == '__main__':
    main()