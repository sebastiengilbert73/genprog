import pandas
import argparse
import logging
import ast
from sklearn import preprocessing
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('dataFilepath')
parser.add_argument('--variableNameToTypeDict',
                    help="The dictionary of variable names to their types. Default: {'p0': 'float', 'p1': 'float', 'p2': 'float', 'p3': 'float', 'p4': 'float', 'p5': 'float', 'p6': 'float', 'p7': 'float', 'p8': 'float', 'p9': 'float'}",
                    default="{'p0': 'float', 'p1': 'float', 'p2': 'float', 'p3': 'float', 'p4': 'float', 'p5': 'float', 'p6': 'float', 'p7': 'float', 'p8': 'float', 'p9': 'float'}")

args = parser.parse_args()
variableNameToTypeDict = ast.literal_eval(args.variableNameToTypeDict)

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s %(message)s')

def main():
    logging.info("data_preprocessor.py main()")
    logging.info("Opening {}".format(args.dataFilepath))
    originalDF = pandas.read_csv(args.dataFilepath)
    print(originalDF.head())

    featuresDF = originalDF[list(variableNameToTypeDict)] # Will exclude the target value (for example 'reward')

    # Normalize each float value
    """logging.info("Normalizing float values")
    for (name, type) in variableNameToTypeDict.items():
        if type == 'float':
            print (name)
    """

    min_max_scaler = preprocessing.MinMaxScaler()
    min_max_scaler.fit_transform(featuresDF)

    logging.info("After normalization, features array is:")
    normalizedFeaturesArr = min_max_scaler.transform(featuresDF)
    print (normalizedFeaturesArr)

    saved_scaler_filepath = args.dataFilepath + '_features_MinMaxScaler.pkl'
    logging.info("Saving the features MinMaxScaler as {}".format(saved_scaler_filepath))
    pickle.dump(min_max_scaler, open(saved_scaler_filepath, 'wb'))

if __name__ == '__main__':
    main()