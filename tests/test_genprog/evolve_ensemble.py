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
parser.add_argument('--datasetPrototype', help="The function prototype to generate the dataset. Default: 'sin'", default='sin')
parser.add_argument('--variableNameToTypeDict', help="The dictionary of variable names to their types. Default: {'x': 'float'}", default="{'x': 'float'}")
parser.add_argument('--numberOfSamples', help="The number of dataset samples to generate. Default: 1000", type=int, default=1000)
parser.add_argument('--noiseStdDev', help='The standard deviation of the white gaussian noise. Default: 0.05', type=float, default=0.05)
parser.add_argument('--trainingProportion', help="The proportion of training samples [0, 1]. Default: 0.8", type=float, default=0.8)
parser.add_argument('--numberOfIndividuals', help="The number of individuals in the population. Default: 100", type=int, default=100)
parser.add_argument('--levelToFunctionProbabilityDict', help="The level to function probability dict. Default: '{0: 1.0, 1: 1.0, 2: 1.0, 3: 0.75, 4: 0.5}'", default='{0: 1.0, 1: 1.0, 2: 1.0, 3: 0.75, 4: 0.5}')
parser.add_argument('--proportionOfConstants', help='When choosing between a variable and a constant, probability to choose a constant. Default: 0.6', type=float, default=0.6)
parser.add_argument('--constantCreationParametersList', help="The list of parameters that the interpreter will use to create a constant (CreateConstant()). In case of ArithmeticsInterpreter, it is [minValue, maxValue]. Default: '[-10, 10]'", default='[-10, 10]')
parser.add_argument('--numberOfTournamentParticipants', help="The number of participants in a tournament. Default: 2", type=int, default=2)
parser.add_argument('--numberOfGenerations', help="The number of generations. Default: 10", type=int, default=10)
parser.add_argument('--mutationProbability', help='The probability to mutate a child. Default: 0.25', type=float, default=0.25)
parser.add_argument('--proportionOfNewIndividuals', help='The proportion of new individuals created at each generation. Default: 0.15', type=float, default=0.15)
parser.add_argument('--weightForNumberOfElements', help='The multiplicative factor to the number of elements, as a penalty for large trees. Default: 0.0001', type=float, default=0.0001)
parser.add_argument('--learningRate', help='The learning rate for backpropagation. Default: 0.01', type=float, default=0.01)
parser.add_argument('--numberOfEpochsPerGeneration', help='At each generation, the number of training epochs for constant optimization. default: 3', type=int, default=3)
parser.add_argument('--numberOfEnsembleMembers', help='The number of champion individuals to generate. Default: 31', type=int, default=31)
parser.add_argument('--numberOfTrainingSamplesPerMember', help='The number of training samples used to train each member. Default: 200', type=int, default=200)
args = parser.parse_args()
levelToFunctionProbabilityDict = ast.literal_eval(args.levelToFunctionProbabilityDict)
constantCreationParametersList = ast.literal_eval(args.constantCreationParametersList)
variableNameToTypeDict: Dict[str, str] = ast.literal_eval(args.variableNameToTypeDict)

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s %(message)s')

def evaluate_individuals(population: gpevo.ArithmeticsPopulation,
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


def main() -> None:
    logging.info("evolve_ensemble.py main()")

    tree_filepath: str = '../../src/genprog/domains/arithmetics.xml'
    returnType: str = 'float'

    # Generate the dataset
    logging.info("Generating the dataset '{}'...".format(args.datasetPrototype))
    if args.datasetPrototype.lower().endswith('2d'):
        xDictOutputValueTupleList: List[Tuple[Dict[str, float], float]] = create_dataset.CreateDataset_2D(
            args.datasetPrototype, args.numberOfSamples, args.noiseStdDev
        )
    else:
        xDictOutputValueTupleList = create_dataset.CreateDataset(
            args.datasetPrototype, args.numberOfSamples, args.noiseStdDev
        )
    # Split dataset
    (trainingDataset, validationDataset) = create_dataset.SplitDataset(
        xDictOutputValueTupleList, trainingProportion=args.trainingProportion
    )

    # Create the interpreter
    domainFunctionsTree: ET.ElementTree = ET.parse(tree_filepath)
    interpreter: gp.ArithmeticsInterpreter = gp.ArithmeticsInterpreter(domainFunctionsTree)

    # Output monitoring file
    generationsCostFile = open('./outputs/generationsCost.csv', "w", buffering=1)  # Flush the buffer at each line
    generationsCostFile.write(
        "generation,championTrainingCost,championValidationCost,medianTrainingCost,medianValidationCost\n")

    for memberNdx in range(args.numberOfEnsembleMembers):
        logging.info("Evolving member {}".format(memberNdx))
        # Choose the training samples
        memberTrainingDataset = random.choices(trainingDataset, k=args.numberOfTrainingSamplesPerMember)

        # Generate the population
        logging.info("Generating the population...")
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

        # Optimize constants with backpropagation
        logging.info("Optimizing constants with backpropagation...")
        for epoch in range(args.numberOfEpochsPerGeneration):
            for individualNdx in range(len(population._individualsList)):
                population._individualsList[individualNdx] = interpreter.EpochOfTraining(
                    population._individualsList[individualNdx],
                    variableNameToTypeDict,
                    returnType,
                    memberTrainingDataset,
                    args.learningRate
                )

        lowestChampionValidationCost = sys.float_info.max
        training_individualToCostDict = None
        memberChampion = None

        for generationNdx in range(1, args.numberOfGenerations + 1):
            logging.info("Generation {}".format(generationNdx))
            training_individualToCostDict = population.NewGenerationWithTournament(
                memberTrainingDataset,
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

            # Optimize constants with backpropagation
            for epoch in range(args.numberOfEpochsPerGeneration):
                for individualNdx in range(len(population._individualsList)):
                    population._individualsList[individualNdx] = interpreter.EpochOfTraining(
                        population._individualsList[individualNdx],
                        variableNameToTypeDict,
                        returnType,
                        memberTrainingDataset,
                        args.learningRate
                    )
            # Population evaluation
            (validationChampion, championTrainingCost, championValidationCost, medianTrainingCost,
             medianValidationCost,
             training_individualToCostDict) = \
                evaluate_individuals(
                    population,
                    memberTrainingDataset,
                    validationDataset,
                    variableNameToTypeDict,
                    interpreter,
                    returnType
                )

            # Output monitoring file
            generationsCostFile.write(str(memberNdx * args.numberOfGenerations + generationNdx) + ',' + str(championTrainingCost) + ',' + str(
                championValidationCost) + ',' + str(
                medianTrainingCost) + ',' + str(medianValidationCost) + '\n')

            # Is it a new champion?
            if championValidationCost < lowestChampionValidationCost:
                #validationChampion.Save('./outputs/champion_' + str(generationNdx) + '.xml')
                memberChampion = validationChampion
                lowestChampionValidationCost = championValidationCost

            # Comparison file
            comparisonFile = open('./outputs/comparison.csv', 'w', buffering=1)
            comparisonFile.write('x,target,prediction\n')
            for xTarget in validationDataset:
                x = xTarget[0]['x']
                target = xTarget[1]
                prediction = interpreter.Evaluate(validationChampion, variableNameToTypeDict, xTarget[0],
                                                  returnType)
                comparisonFile.write(str(x) + ',' + str(target) + ',' + str(prediction) + '\n')
        if memberChampion is not None:
            memberChampion.Save('./outputs/member_{}.xml'.format(memberNdx))

if __name__ == '__main__':
    main()