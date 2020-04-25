import logging
from genprog import core as gp, evolution as gpevo
from typing import Dict, List, Tuple, Any, Optional, Set, Union
import random
import math
import argparse
import xml.etree.ElementTree as ET
import ast
import sys
import create_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--ensembleMembersFilepathPrefix', help="The filepath prefix of the ensemble members. Default: './outputs/member_'", default='./outputs/member_')
parser.add_argument('--variableNameToTypeDict', help="The dictionary of variable names to their types. Default: {'x': 'float'}", default="{'x': 'float'}")
parser.add_argument('--datasetPrototype', help="The function prototype to generate the dataset. Default: 'sin'", default='sin')
parser.add_argument('--numberOfSamples', help="The number of dataset samples to generate. Default: 1000", type=int, default=1000)
parser.add_argument('--noiseStdDev', help='The standard deviation of the white gaussian noise. Default: 0.05', type=float, default=0.05)
args = parser.parse_args()

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s %(message)s')

variableNameToTypeDict: Dict[str, str] = ast.literal_eval(args.variableNameToTypeDict)

def main():
    logging.info("test_ensemble.py main()")

    tree_filepath: str = '../../src/genprog/domains/arithmetics.xml'
    returnType: str = 'float'

    # Load the ensemble
    ensemble: gpevo.ArithmeticsPopulation = gpevo.ArithmeticsPopulation()
    ensemble.LoadIndividuals(args.ensembleMembersFilepathPrefix)
    logging.debug("len(ensemble._individualsList) = {}".format(len(ensemble._individualsList)))

    # Generate a dataset
    logging.info("Generating the dataset '{}'...".format(args.datasetPrototype))
    if args.datasetPrototype.lower().endswith('2d'):
        xDictOutputValueTupleList: List[Tuple[Dict[str, float], float]] = create_dataset.CreateDataset_2D(
            args.datasetPrototype, args.numberOfSamples, args.noiseStdDev
        )
    else:
        xDictOutputValueTupleList = create_dataset.CreateDataset(
            args.datasetPrototype, args.numberOfSamples, args.noiseStdDev
        )

    # Create the interpreter
    domainFunctionsTree: ET.ElementTree = ET.parse(tree_filepath)
    interpreter: gp.ArithmeticsInterpreter = gp.ArithmeticsInterpreter(domainFunctionsTree)

    # Comparison file
    comparisonFile = open('./outputs/comparison.csv', 'w', buffering=1)
    comparisonFile.write('x,target,prediction\n')

    inputsList: List[ Dict[str, Any] ] = [t[0] for t in xDictOutputValueTupleList] # element 0 of each tuple in the list

    inputPredictionList: List[ Tuple[ Dict[str, Any], Any] ] = ensemble.AverageEvaluation(
        inputsList,
        interpreter,
        variableNameToTypeDict,
        returnType
    )
    for sampleNdx in range(len(xDictOutputValueTupleList)):
        input = xDictOutputValueTupleList[sampleNdx][0]
        targetOutput = xDictOutputValueTupleList[sampleNdx][1]
        prediction = inputPredictionList[sampleNdx][1]
        comparisonFile.write('{},{},{}\n'.format(input['x'], targetOutput, prediction))

if __name__ == '__main__':
    main()