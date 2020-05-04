import logging
from genprog import core as gp, evolution as gpevo
from typing import Dict, List, Tuple, Any, Optional, Set, Union
import create_dataset
import xml.etree.ElementTree as ET
import ast
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--datasetPrototype', help="The function prototype to generate the dataset. Default: 'sin'", default='sin')
parser.add_argument('--variableNameToTypeDict', help="The dictionary of variable names to their types. Default: {'x': 'float'}", default="{'x': 'float'}")
parser.add_argument('--numberOfSamples', help="The number of dataset samples to generate. Default: 1000", type=int, default=1000)
parser.add_argument('--noiseStdDev', help='The standard deviation of the white gaussian noise. Default: 0.05', type=float, default=0.05)
parser.add_argument('--datasetFilepath', help="The filepath to the dataset. 'None' means the dataset will be generated. Default: 'None'", default='None')
parser.add_argument('--returnType', help="The trees return type. Default: 'float'", default='float')
args = parser.parse_args()


logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s %(message)s')


def main() -> None:
    logging.info("test_sum_of_residuals.py main()")

    variableNameToTypeDict: Dict[str, str] = ast.literal_eval(args.variableNameToTypeDict)

    dataset: List[Tuple[Dict[str, float], float]] = []
    if args.datasetFilepath is not 'None':
        dataset = create_dataset.LoadDataset(args.datasetFilepath, variableNameToTypeDict, args.returnType)
    else:
        if args.datasetPrototype.lower().endswith('2d'):
            dataset = create_dataset.CreateDataset_2D(
                args.datasetPrototype, args.numberOfSamples, args.noiseStdDev
            )
        else:
            dataset = create_dataset.CreateDataset(
                args.datasetPrototype, args.numberOfSamples, args.noiseStdDev
            )
        create_dataset.SaveDataset(dataset,
                                   './outputs/test_sum_of_residuals_generated_{}.csv'.format(args.datasetPrototype))

    # Load the population
    population: gpevo.ArithmeticsPopulation = gpevo.ArithmeticsPopulation()
    population.LoadIndividuals('./outputs/residualChampion_')

    # Create the interpreter
    tree_filepath: str = '../../src/genprog/domains/arithmetics.xml'
    domainFunctionsTree: ET.ElementTree = ET.parse(tree_filepath)
    interpreter: gp.ArithmeticsInterpreter = gp.ArithmeticsInterpreter(domainFunctionsTree)

    # Evaluate
    evaluations: List[float] = []
    for sample in dataset:
        inputsDict = sample[0]
        evaluationSum = 0
        for individual in population._individualsList:
            individualOutput: float = interpreter.Evaluate(
                individual,
                variableNameToTypeDict,
                inputsDict,
                args.returnType
            )
            evaluationSum += individualOutput
        evaluations.append(evaluationSum)

    # Write to a comparison file
    comparisonFile = open('./outputs/comparison.csv', 'w', buffering=1)
    sample0 = dataset[0]
    featuresDict = sample0[0]
    featureNames = list(featuresDict.keys())
    comparisonFile.write('{},target,prediction\n'.format('x'))#,'.join(featureNames)))
    for index, xTarget in enumerate(dataset):
        featureValues = list(xTarget[0].values())
        featureValues = [str(v) for v in featureValues] # Convert to strings
        target = xTarget[1]
        prediction = evaluations[index]
        comparisonFile.write(featureValues[0] + ',' + str(target) + ',' + str(prediction) + '\n')

if __name__ == '__main__':
    main()