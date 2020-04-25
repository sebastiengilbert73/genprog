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