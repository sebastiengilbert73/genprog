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
parser.add_argument('--treeFilepath', help="The filepath to the arithmetics tree. Default: './samples/addition.xml'", default='./samples/sin.xml')
parser.add_argument('--variableNameToTypeDict', help="The dictionary of variable name to their type. Default: {'x': 'float'}", default="{'x': 'float'}")
parser.add_argument('--returnType', help="The tree return type. Default: 'float'", default='float')
parser.add_argument('--datasetPrototype', help="The function prototype to generate the dataset. Default: 'sin'", default='sin')
parser.add_argument('--numberOfSamples', help="The number of dataset samples to generate. Default: 1000", type=int, default=1000)
parser.add_argument('--trainingProportion', help="The proportion of training samples [0, 1]. Default: 0.8", type=float, default=0.8)
parser.add_argument('--numberOfEpochs', help='The number of epochs. Default: 100', type=int, default=100)
parser.add_argument('--learningRate', help='The learning rate. Default: 0.001', type=float, default=0.001)
args = parser.parse_args()

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s %(message)s')
variableNameToTypeDict: Dict[str, str] = ast.literal_eval(args.variableNameToTypeDict)

def main():
    logging.info("test_backpropagation.py main()")

    # Create interpreter
    arithmeticsFunctionsFilepaths: str = './genprog/domains/arithmetics.xml'
    arithmeticsFunctionsTree: ET.ElementTree = ET.parse(arithmeticsFunctionsFilepaths)
    interpreter: gp.ArithmeticsInterpreter = gp.ArithmeticsInterpreter(arithmeticsFunctionsTree)

    # Load the individual
    individual: gp.Individual = gp.LoadIndividual(args.treeFilepath)
    headElm: ET.Element = list(individual._tree.getroot())[0]

    # Create a population with the single individual
    population: gpevo.ArithmeticsPopulation = gpevo.ArithmeticsPopulation([individual])

    # Generate the dataset
    logging.info("Generating the dataset '{}'...".format(args.datasetPrototype))
    if args.datasetPrototype.lower().endswith('2d'):
        xDictOutputValueTupleList: List[Tuple[Dict[str, float], float]] = create_dataset.CreateDataset_2D(
            args.datasetPrototype, args.numberOfSamples
        )
    else:
        xDictOutputValueTupleList = create_dataset.CreateDataset(
            args.datasetPrototype, args.numberOfSamples
        )
    # Split dataset
    (trainingDataset, validationDataset) = create_dataset.SplitDataset(
        xDictOutputValueTupleList, trainingProportion=args.trainingProportion
    )

    # Evaluate the population
    (validationChampion, championTrainingCost, championValidationCost, medianTrainingCost, medianValidationCost,
     training_individualToCostDict) = \
        create_dataset.EvaluateIndividuals(
            population,
            trainingDataset,
            validationDataset,
            variableNameToTypeDict,
            interpreter,
            args.returnType
        )

    for epoch in range(1, args.numberOfEpochs + 1):
        logging.info("Epoch {}".format(epoch))
        individual = interpreter.EpochOfTraining(
            individual,
            variableNameToTypeDict,
            args.returnType,
            trainingDataset,
            args.learningRate
        )

        # Evaluate the population
        (validationChampion, championTrainingCost, championValidationCost, medianTrainingCost, medianValidationCost,
         training_individualToCostDict) = \
            create_dataset.EvaluateIndividuals(
                population,
                trainingDataset,
                validationDataset,
                variableNameToTypeDict,
                interpreter,
                args.returnType
            )

        # Comparison file
        comparisonFile = open('./outputs/comparison.csv', 'w', buffering=1)
        comparisonFile.write('x,target,prediction\n')
        for xTarget in validationDataset:
            x = xTarget[0]['x']
            target = xTarget[1]
            prediction = interpreter.Evaluate(individual, variableNameToTypeDict, xTarget[0], args.returnType)
            comparisonFile.write(str(x) + ',' + str(target) + ',' + str(prediction) + '\n')

    """elementToEvaluationDict = interpreter.EvaluateElements(
        headElm,
        variableNameToTypeDict,
        variableNameToValueDict,
        expectedReturnType
    )
    logging.info("elementToEvaluationDict = {}".format(elementToEvaluationDict))

    elementToGradientDict = interpreter.Backpropagate(
        headElm,
        elementToEvaluationDict
    )
    logging.info("elementToGradientDict = {}".format(elementToGradientDict))

    # Create a dataset
    xDictOutputValueTupleList: List[Tuple[Dict[str, float], float]] = create_dataset.CreateDataset(
        'parabola', 1000
    )
    # Split dataset
    (trainingDataset, validationDataset) = create_dataset.SplitDataset(
        xDictOutputValueTupleList, trainingProportion=0.8
    )

    validationCostBeforeLearn: float = 0
    #numberOfTrainingSamples = len(trainingDataset)
    numberOfValidationSamples = len(validationDataset)
    learningRate = 0.0001
    for (validationXDict, targetOutput) in validationDataset:
        elementToEvaluationDict = interpreter.EvaluateElements(
            headElm,
            variableNameToTypeDict,
            validationXDict,
            expectedReturnType
        )
        delta = elementToEvaluationDict[headElm] - targetOutput
        validationCostBeforeLearn += abs(delta)/numberOfValidationSamples
    logging.info("validationCostBeforeLearn = {}".format(validationCostBeforeLearn))

    logging.info("Learning...")
    individual = interpreter.EpochOfTraining(
        individual,
        variableNameToTypeDict,
        expectedReturnType,
        trainingDataset,
        learningRate
    )

    logging.info("Evaluating the updated individual...")
    validationCostAfterLearn: float = 0
    for (validationXDict, targetOutput) in validationDataset:
        elementToEvaluationDict = interpreter.EvaluateElements(
            headElm,
            variableNameToTypeDict,
            validationXDict,
            expectedReturnType
        )
        delta = elementToEvaluationDict[headElm] - targetOutput
        validationCostAfterLearn += abs(delta) / numberOfValidationSamples
    logging.info("validationCostAfterLearn = {}".format(validationCostAfterLearn))
    individual.Save('./samples/individualAfterLearning.xml')
    """


if __name__ == '__main__':
    main()