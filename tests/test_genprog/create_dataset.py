from typing import Dict, List, Tuple, Any, Optional, Set, Union
import random
import math


def CreateDataset(prototype: str, numberOfSamples: int) -> List[Tuple[Dict[str, float], float]]:
    xDictOutputValueTupleList: List[Tuple[Dict[str, float], float]] = []

    for sampleNdx in range(numberOfSamples):
        x: float = random.random()
        xToValuesDict: Dict[str, float] = {'x': x}
        outputValue: float = 0
        if prototype == 'sin':
            outputValue = math.sin(2 * math.pi * (x))
        elif prototype == 'parabola':
            outputValue = 6 * x**2 - 5 * x + 0.4
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