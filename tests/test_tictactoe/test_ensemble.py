import argparse
import logging
import genprog.core as gp
import genprog.evolution as gpevo
import autoencoder.position
import tictactoe
import ast
import pickle
import numpy
import xml.etree.ElementTree as ET

parser = argparse.ArgumentParser()
parser.add_argument('autoencoder', help='The filepath to the autoencoder')
parser.add_argument('featuresPreprocessor', help="The preprocessor for features")
parser.add_argument('--ensembleMembersFilepathPrefix', help="The filepath prefix of the ensemble members. Default: './outputs/residualChampion_'", default='./outputs/residualChampion_')
parser.add_argument('--variableNameToTypeDict',
                    help="The dictionary of variable names to their types. Default: {'p0': 'float', 'p1': 'float', 'p2': 'float', 'p3': 'float', 'p4': 'float', 'p5': 'float', 'p6': 'float', 'p7': 'float', 'p8': 'float', 'p9': 'float'}",
                    default="{'p0': 'float', 'p1': 'float', 'p2': 'float', 'p3': 'float', 'p4': 'float', 'p5': 'float', 'p6': 'float', 'p7': 'float', 'p8': 'float', 'p9': 'float'}")
parser.add_argument('--domainPrimitivesFilepath', help="The filepath to the domain primitives. Default: './genprog/domains/arithmetics.xml'", default='./genprog/domains/arithmetics.xml')

args = parser.parse_args()
variableNameToTypeDict = ast.literal_eval(args.variableNameToTypeDict)

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s %(message)s')

def main():
    logging.info("test_ensemble.py main()")
    residualsPopulation = gpevo.ArithmeticsPopulation()
    residualsPopulation.LoadIndividuals(args.ensembleMembersFilepathPrefix)

    # Create the autoencoder
    encoder = autoencoder.position.Net()
    encoder.Load(args.autoencoder)

    # Load the features preprocessor
    preprocessor = pickle.load(open(args.featuresPreprocessor, 'rb'))

    # Game authority
    authority = tictactoe.Authority()
    position = authority.InitialPosition()
    position[0, 0, 1, 1] = 1
    position[1, 0, 0, 1] = 1
    position[0, 0, 0, 2] = 1
    position[1, 0, 2, 0] = 1
    position[0, 0, 2, 2] = 1
    #position = authority.SwapPositions(position, 'X', 'O')
    authority.Display(position)

    # Encode the position
    encoding = encoder.Encode(position.unsqueeze(0))
    logging.debug("encoding = {}".format(encoding))
    # Preprocess the encoding
    preprocessedEncoding = preprocessor.transform(encoding.detach().numpy())[0]

    print ("preprocessedEncoding = {}".format(preprocessedEncoding))

    # Load the population
    population = gpevo.ArithmeticsPopulation()
    population.LoadIndividuals(args.ensembleMembersFilepathPrefix)

    # Variable names
    variableNames = list(variableNameToTypeDict)
    variableNameToValueDict = dict(zip(variableNames, preprocessedEncoding))
    logging.debug("variableNameToValueDict = {}".format(variableNameToValueDict))

    # Load the interpreter
    domainFunctionsTree: ET.ElementTree = ET.parse(args.domainPrimitivesFilepath)
    interpreter: gp.ArithmeticsInterpreter = gp.ArithmeticsInterpreter(domainFunctionsTree)

    outputsSum = population.SumOfEvaluations([variableNameToValueDict],
    interpreter,
    variableNameToTypeDict,
    'float')
    logging.info("outputsSum = {}".format(outputsSum))

if __name__ == '__main__':
    main()