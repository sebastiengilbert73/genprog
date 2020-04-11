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
parser.add_argument('--treeFilepath', help="The filepath to the arithmetics tree. Default: './samples/addition.xml'", default='./samples/addition.xml')
args = parser.parse_args()

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s %(message)s')

def main():
    logging.info("test_backpropagation.py main()")

    arithmeticsFunctionsFilepaths: str = './genprog/domains/arithmetics.xml'
    arithmeticsFunctionsTree: ET.ElementTree = ET.parse(arithmeticsFunctionsFilepaths)
    individual: gp.Individual = gp.LoadIndividual(args.treeFilepath)
    #individual.Save('./samples/individual_right_after_loading.xml')
    #treeDoc: ET.ElementTree = ET.parse(args.treeFilepath)
    headElm: ET.Element = list(individual._tree.getroot())[0]
    interpreter: gp.ArithmeticsInterpreter = gp.ArithmeticsInterpreter( arithmeticsFunctionsTree)

    variableNameToTypeDict: Dict[str, str] = {'x': 'float'}
    variableNameToValueDict: Dict[str, Union[float, bool]] = {'x': 7.0}
    expectedReturnType:str = 'float'

    elementToEvaluationDict = interpreter.EvaluateElements(
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
    numberOfTrainingSamples = len(trainingDataset)
    numberOfValidationSamples = len(validationDataset)
    learningRate = 0.01
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



if __name__ == '__main__':
    main()