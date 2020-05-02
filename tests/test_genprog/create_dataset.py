from typing import Dict, List, Tuple, Any, Optional, Set, Union
import random
import math
import logging
from genprog import core as gp, evolution as gpevo
import numpy


def CreateDataset(prototype: str, numberOfSamples: int, noiseStdDev: float=0.05) -> List[Tuple[Dict[str, float], float]]:
    xDictOutputValueTupleList: List[Tuple[Dict[str, float], float]] = []

    for sampleNdx in range(numberOfSamples):
        x: float = random.random()
        xToValuesDict: Dict[str, float] = {'x': x}
        outputValue: float = 0
        if prototype == 'sin':
            outputValue = math.sin(2 * math.pi * (x)) + noiseStdDev * numpy.random.normal()
        elif prototype == 'parabola':
            outputValue = 6 * x**2 - 5 * x + 0.4 + noiseStdDev * numpy.random.normal()
        elif prototype == 'steps':
            if x < 0.2:
                outputValue = -0.3
            elif x < 0.7:
                outputValue = 0.7
            else:
                outputValue = 0.4
            outputValue += noiseStdDev * numpy.random.normal()
        else:
            raise NotImplementedError("CreateDataset(): Not implemented prototype '{}'".format(prototype))
        xDictOutputValueTupleList.append((xToValuesDict, outputValue))

    return xDictOutputValueTupleList

def CreateDataset_2D(prototype: str, numberOfSamples: int, noiseStdDev: float=0.05) -> List[Tuple[Dict[str, float], float]]:
    xDictOutputValueTupleList: List[Tuple[Dict[str, float], float]] = []

    for sampleNdx in range(numberOfSamples):
        x: float = random.random()
        y: float = random.random()
        xToValuesDict: Dict[str, float] = {'x': x, 'y': y}
        outputValue: float = 0
        if prototype == 'sin2d':
            outputValue = math.sin(2 * math.pi * x + 0.7 * math.pi * y - 0.4) + noiseStdDev * numpy.random.normal()
        elif prototype == 'parabola2d':
            outputValue = 6 * x**2 - 5 * x + 0.4 - 9.5 * y**2 + 1.7 * y + 2.1 * x * y + noiseStdDev * numpy.random.normal()
        else:
            raise NotImplementedError("CreateDataset(): Not implemented prototype '{}'".format(prototype))
        xDictOutputValueTupleList.append((xToValuesDict, outputValue))

    return xDictOutputValueTupleList

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

def EvaluateIndividuals(population: gpevo.ArithmeticsPopulation,
                         trainingDataset: List[Tuple[Dict[str, Union[float, bool]], float]],
                         validationDataset: List[Tuple[Dict[str, Union[float, bool]], float]],
                         variableNameToTypeDict: Dict[str, str],
                         interpreter: gp.ArithmeticsInterpreter,
                         returnType: str
                         ) -> Tuple[gp.Individual, float, float, float, float, Dict[gp.Individual, float]]:
    logging.info("Evaluating the individuals...")
    training_individualToCostDict: Dict[gp.Individual, float] = population.EvaluateIndividualCosts(
        trainingDataset,
        variableNameToTypeDict,
        interpreter,
        returnType
    )

    (trainingChampion, championTrainingCost) = population.Champion(training_individualToCostDict)
    medianTrainingCost: float = population.MedianCost(training_individualToCostDict)

    # Validation cost
    validation_individualToCostDict: Dict[gp.Individual, float] = population.EvaluateIndividualCosts(
        validationDataset,
        variableNameToTypeDict,
        interpreter,
        returnType
    )
    (validationChampion, championValidationCost) = population.Champion(validation_individualToCostDict)
    medianValidationCost = population.MedianCost(validation_individualToCostDict)

    logging.info(
        "championTrainingCost = {};    championValidationCost = {};    medianTrainingCost = {};    medianValidationCost = {}".format(
            championTrainingCost, championValidationCost, medianTrainingCost, medianValidationCost))
    return (validationChampion, championTrainingCost, championValidationCost, medianTrainingCost, medianValidationCost, training_individualToCostDict)

def SaveDataset(dataset: List[Tuple[Dict[str, Union[float, bool]], Union[float, bool]]],
                filepath: str) -> None:
    if len(dataset) < 1:
        raise ValueError("create_dataset.SaveDataset(): Empty dataset")
    input0 = dataset[0][0]
    inputVariableNames = list(input0.keys())
    with open(filepath, 'w') as outputFile:
        for variableName in inputVariableNames:
            outputFile.write("{},".format(variableName))
        outputFile.write('target\n')
        for sample in dataset:
            inputDict = sample[0]
            for variableName in inputVariableNames:
                if variableName not in inputDict:
                    raise ValueError("create_dataset.SaveDataset(): For a sample, there is no variable '{}'".format(variableName))
                variableValue = inputDict[variableName]
                outputFile.write("{},".format(variableValue))
            outputFile.write("{}\n".format(sample[1]))

def LoadDataset(filepath: str,
                variableNameToTypeDict: Dict[str, str],
                returnType: str) -> List[Tuple[Dict[str, Union[float, bool]], Union[float, bool]]]:
    dataset = []
    with open(filepath, 'r') as inputFile:
        # Read the header
        line = inputFile.readline()
        tokens = line.split(',')
        if tokens[-1] != 'target\n':
            raise ValueError("The last header token ({}) is not 'target'".format(tokens[-1]))
        variableNames = tokens[:-1]
        while(line):
            line = inputFile.readline()
            if line:
                tokens = line.split(',')
                inputDict = dict(zip(variableNames, tokens[: -1]))
                inputDict = ConvertVariableTypes(inputDict, variableNameToTypeDict)
                targetValue = tokens[-1]
                targetValue = ConvertType(returnType, targetValue)
                dataset.append((inputDict, targetValue))
    return dataset

def ConvertVariableTypes(inputDict: Dict[str, str],
                         variableNameToTypeDict: Dict[str, str]) -> Dict[str, Union[float, bool]]:
    convertedDict = {}
    for variableName, value in inputDict.items():
        variableType = variableNameToTypeDict[variableName]
        convertedValue = ConvertType(variableType, value)
        convertedDict[variableName] = convertedValue
    return convertedDict

def ConvertType(expectedType: str, value: str) -> Union[float, bool]:
    if expectedType == 'float':
        convertedValue = float(value)
    elif expectedType == 'bool':
        if value.lower() == 'true':
            convertedValue = True
        elif value.lower() == 'false':
            convertedValue = False
        else:
            raise ValueError("ConvertType(): Unknown value '{}'. A boolean is expected.".format(value))
    else:
        raise NotImplementedError("ConvertType(): Unsupported variable type '{}'".format(variableType))
    return convertedValue