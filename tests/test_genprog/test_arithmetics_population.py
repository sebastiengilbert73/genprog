import logging
from genprog import core as gp, evolution as gpevo
from typing import Dict, List, Tuple, Any, Optional, Set, Union
import random
import math
import argparse
import xml.etree.ElementTree as ET
import ast
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--datasetPrototype', help="The function prototype to generate the dataset. Default: 'sin'", default='sin')
parser.add_argument('--numberOfSamples', help="The number of dataset samples to generate. Default: 1000", type=int, default=1000)
parser.add_argument('--trainingProportion', help="The proportion of training samples [0, 1]. Default: 0.8", type=float, default=0.8)
parser.add_argument('--numberOfIndividuals', help="The number of individuals in the population. Default: 300", type=int, default=300)
parser.add_argument('--levelToFunctionProbabilityDict', help="The level to function probability dict. Default: '{0: 1, 1: 0.9, 2: 0.9, 3: 0.5}'", default='{0: 1.0, 1: 0.9, 2: 0.9, 3: 0.5}')
parser.add_argument('--proportionOfConstants', help='When choosing between a variable and a constant, probability to choose a constant. Default: 0.8', type=float, default=0.8)
parser.add_argument('--constantCreationParametersList', help="The list of parameters that the interpreter will use to create a constant (CreateConstant()). In case of ArithmeticsInterpreter, it is [minValue, maxValue]. Default: '[-10, 10]'", default='[-10, 10]')
parser.add_argument('--numberOfTournamentParticipants', help="The number of participants in a tournament. Default: 3", type=int, default=3)
parser.add_argument('--numberOfGenerations', help="The number of generations. Default: 100", type=int, default=100)
parser.add_argument('--mutationProbability', help='The probability to mutate a child. Default: 0.25', type=float, default=0.25)
parser.add_argument('--proportionOfNewIndividuals', help='The proportion of new individuals created at each generation. Default: 0.05', type=float, default=0.05)
parser.add_argument('--weightForNumberOfElements', help='The multiplicative factor to the number of elements, as a penalty for large trees. Default: 0.0001', type=float, default=0.0001)
args = parser.parse_args()
levelToFunctionProbabilityDict = ast.literal_eval(args.levelToFunctionProbabilityDict)
constantCreationParametersList = ast.literal_eval(args.constantCreationParametersList)

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s %(message)s')


def CreateDataset(prototype: str, numberOfSamples: int) -> List[Tuple[Dict[str, float], float]]:
    xyDictOutputValueTupleList: List[Tuple[Dict[str, float], float]] = []

    for sampleNdx in range(numberOfSamples):
        x: float = random.random()
        #y: float = random.random()
        xyToValuesDict: Dict[str, float] = {'x': x}#, 'y': y}
        outputValue: float = 0
        if prototype == 'sin':
            outputValue = math.sin(2 * math.pi * (x))# + y))
        elif prototype == 'parabola':
            outputValue = 6 * x**2 - 5 * x + 0.4
        else:
            raise NotImplementedError("CreateDataset(): Not implemented prototype '{}'".format(prototype))
        xyDictOutputValueTupleList.append((xyToValuesDict, outputValue))

    return xyDictOutputValueTupleList

def SplitDataset(datasetList: List[Tuple[Dict[str, float], float]], trainingProportion: float) -> \
        Tuple[ List[Tuple[Dict[str, float], float]], List[Tuple[Dict[str, float], float]]]:
    if trainingProportion < 0 or trainingProportion > 1:
        raise ValueError("SplitDataset(): trainingProportion ({}) is out of [0, 1]".format(trainingProportion))
    if len(datasetList) == 0:
        raise ValueError("SplitDataset(): Empty datasetList")
    trainingDataset: List[Tuple[Dict[str, float], float]] = []
    validationDataset: List[Tuple[Dict[str, float], float]] = []

    numberOfTrainingSamples: int = round(trainingProportion * len(datasetList))
    for (xyDict, value) in datasetList:
        if len(trainingDataset) < numberOfTrainingSamples:
            trainingDataset.append((xyDict, value))
        else:
            validationDataset.append((xyDict, value))
    return (trainingDataset, validationDataset)

def main() -> None:
    logging.info("test_arithmetics_population.py main()")
    logging.info("Generating the dataset '{}'...".format(args.datasetPrototype))
    xyDictOutputValueTupleList = CreateDataset(args.datasetPrototype, args.numberOfSamples)
    (trainingDatasetList, validationDatasetList) = SplitDataset(xyDictOutputValueTupleList, args.trainingProportion)

    variableNameToTypeDict: Dict[str, str] = {'x': 'float'}#, 'y': 'float'}
    tree_filepath: str = '../../src/genprog/domains/arithmetics.xml'
    returnType: str = 'float'
    domainFunctionsTree: ET.ElementTree = ET.parse(tree_filepath)
    interpreter: gp.ArithmeticsInterpreter = gp.ArithmeticsInterpreter(domainFunctionsTree)
    population: gpevo.ArithmeticsPopulation = gpevo.ArithmeticsPopulation()
    population.Generate(
        args.numberOfIndividuals,
        interpreter,
        returnType,
        levelToFunctionProbabilityDict,
        args.proportionOfConstants,
        constantCreationParametersList,
        variableNameToTypeDict
    )

    logging.info("Evaluating the individuals...")
    training_individualToCostDict: Dict[gp.Individual, float] = population.EvaluateIndividualCosts(
        trainingDatasetList,
        variableNameToTypeDict,
        interpreter,
        returnType
    )

    (champion, championTrainingCost) = population.Champion(training_individualToCostDict)
    medianTrainingCost: float = population.MedianCost(training_individualToCostDict)

    # Validation cost
    validation_individualToCostDict: Dict[gp.Individual, float] = population.EvaluateIndividualCosts(
        validationDatasetList,
        variableNameToTypeDict,
        interpreter,
        returnType
    )
    medianValidationCost = population.MedianCost(validation_individualToCostDict)
    championValidationCost = validation_individualToCostDict[champion]

    logging.info(
        "championTrainingCost = {};    championValidationCost = {};    medianTrainingCost = {};    medianValidationCost = {}".format(
            championTrainingCost, championValidationCost, medianTrainingCost, medianValidationCost))

    # Output monitoring file
    generationsCostFile = open('./outputs/generationsCost.csv', "w", buffering=1)  # Flush the buffer at each line
    generationsCostFile.write(
        "generation,championTrainingCost,championValidationCost,medianTrainingCost,medianValidationCost\n")
    generationsCostFile.write('0,' + str(championTrainingCost) + ',' + str(championValidationCost) + ',' + str(medianTrainingCost) + ',' + str(medianValidationCost) + '\n')

    lowestChampionValidationCost = sys.float_info.max

    for generationNdx in range(1, args.numberOfGenerations + 1):
        logging.info("Generation {}".format(generationNdx))
        training_individualToCostDict = population.NewGenerationWithTournament(
            trainingDatasetList,
            variableNameToTypeDict,
            interpreter,
            returnType,
            args.numberOfTournamentParticipants,
            args.mutationProbability,
            training_individualToCostDict,
            args.proportionOfConstants,
            levelToFunctionProbabilityDict,
            None,
            constantCreationParametersList,
            args.proportionOfNewIndividuals,
            args.weightForNumberOfElements
        )
        (champion, championTrainingCost) = population.Champion(training_individualToCostDict)
        medianTrainingCost = population.MedianCost(training_individualToCostDict)

        # Validation cost
        validation_individualToCostDict = population.EvaluateIndividualCosts(
        validationDatasetList,
        variableNameToTypeDict,
        interpreter,
        returnType
        )
        medianValidationCost = population.MedianCost(validation_individualToCostDict)
        championValidationCost = validation_individualToCostDict[champion]

        logging.info("championTrainingCost = {};    championValidationCost = {};    medianTrainingCost = {};    medianValidationCost = {}".format(championTrainingCost, championValidationCost, medianTrainingCost, medianValidationCost))
        generationsCostFile.write(
            str(generationNdx) + ',' + str(championTrainingCost) + ',' + str(championValidationCost) + ',' + str(medianTrainingCost) + ',' + str(medianValidationCost) + '\n')
        if championValidationCost < lowestChampionValidationCost:
            champion.Save('./outputs/champion_' + str(generationNdx) + '.xml')
            lowestChampionValidationCost = championValidationCost

        # Comparison file
        comparisonFile = open('./outputs/comparison.csv', 'w', buffering=1)
        comparisonFile.write('x,target,prediction\n')
        xTargetPredictionTuplesList: List[Tuple[float, float, float]] = []
        for xTarget in validationDatasetList:
            x = xTarget[0]['x']
            target = xTarget[1]
            prediction = interpreter.Evaluate( champion, variableNameToTypeDict, xTarget[0], returnType)
            comparisonFile.write(str(x) + ',' + str(target) + ',' + str(prediction) + '\n')



if __name__ == '__main__':
    main()