import argparse
import numpy
import torch
import os
import ast
import tictactoe
import logging
import autoencoder.position # autoencoder
import winRatesRegression
import genprog.core as gp
import genprog.evolution as gpevo
import xml.etree.ElementTree as ET
import pickle


parser = argparse.ArgumentParser()
parser.add_argument('autoencoderFilepath', help='The filepath of the autoencoder')
parser.add_argument('featuresPreprocessor', help="The filepath of the preprocessor for features")
parser.add_argument('populationMembersFilepathPrefix', help='The prefix of the filepath to the population')
parser.add_argument('--numberOfSimulationsPerPosition', help='For each starting position, the number of simulations. Default: 64', type=int, default=64)
parser.add_argument('--epsilon', help='Epsilon (probability of a random move) for the simulations. Default: 0.5', type=float, default=0.5)
parser.add_argument('--numberOfPositions', help='The number of positions to evaluate. Default: 1000', type=int, default=1000)
parser.add_argument('--outputFilepath', help="The csv file where the results will be written. Default: './outputs/samples.csv'", default='./outputs/samples.csv')
parser.add_argument('--domainPrimitivesFilepath', help="The filepath to the domain primitives. Default: './genprog/domains/arithmetics.xml'", default='./genprog/domains/arithmetics.xml')
parser.add_argument('--variableNameToTypeDict',
                    help="The dictionary of variable names to their types. Default: {'p0': 'float', 'p1': 'float', 'p2': 'float', 'p3': 'float', 'p4': 'float', 'p5': 'float', 'p6': 'float', 'p7': 'float', 'p8': 'float', 'p9': 'float'}",
                    default="{'p0': 'float', 'p1': 'float', 'p2': 'float', 'p3': 'float', 'p4': 'float', 'p5': 'float', 'p6': 'float', 'p7': 'float', 'p8': 'float', 'p9': 'float'}")
args = parser.parse_args()
variableNameToTypeDict = ast.literal_eval(args.variableNameToTypeDict)

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(message)s')

class RegressorPopulation(winRatesRegression.Regressor, gpevo.ArithmeticsPopulation):
    def __init__(self, interpreter, variableNameToTypeDict, returnType):
        self._interpreter = interpreter
        self._variableNameToTypeDict = variableNameToTypeDict
        self._returnType = returnType
    def WinRates(self, positionEncoding):
        variableNames = list(self._variableNameToTypeDict)
        variableNameToValueDict = dict(zip(variableNames, positionEncoding))
        outputsSum = self.SumOfEvaluations([variableNameToValueDict],
                                                 self._interpreter,
                                                 self._variableNameToTypeDict,
                                                 self._returnType)
        if outputsSum[0][1] >= 0:
            return (outputsSum[0][1], 0, 0)
        else:
            return (0, 0, -outputsSum[0][1])

def main():
    logging.info("regressorCreatesSamples.py main()")

    authority = tictactoe.Authority()
    #positionTsrShape = authority.PositionTensorShape()
    playersList = authority.PlayersList()

    # Load the interpreter
    domainFunctionsTree: ET.ElementTree = ET.parse(args.domainPrimitivesFilepath)
    interpreter: gp.ArithmeticsInterpreter = gp.ArithmeticsInterpreter(domainFunctionsTree)

    # Load the ensemble
    if args.epsilon < 1.0:
        population = RegressorPopulation(interpreter, variableNameToTypeDict, 'float')#gpevo.ArithmeticsPopulation()
        population.LoadIndividuals(args.populationMembersFilepathPrefix)

    else:
        population = None

    # Create the autoencoder
    encoder = autoencoder.position.Net()
    encoder.Load(args.autoencoderFilepath)
    numberOfLatentVariables = encoder.numberOfLatentVariables
    header = ''
    for latentNdx in range(numberOfLatentVariables):
        header += 'p' + str(latentNdx) + ','

    # Load the features preprocessor
    preprocessor = pickle.load(open(args.featuresPreprocessor, 'rb'))

    # Create the output file
    outputFile = open(args.outputFilepath, "w",
                         buffering=1)  # Flush the buffer at each line
    outputFile.write(
        header + "player0WinRate,drawRate,player1WinRate\n")



    for positionNdx in range(1, args.numberOfPositions + 1):
        logging.info("Generating position {}...".format(positionNdx))
        startingPosition = winRatesRegression.SimulateRandomGames(authority,
                                                                    encoder=encoder,
                                                                    minimumNumberOfMovesForInitialPositions=0,
                                                                    maximumNumberOfMovesForInitialPositions=7,
                                                                    numberOfPositions=1,
                                                                    swapIfOddNumberOfMoves=False)[0]
        authority.Display(startingPosition)
        numberOfWinsForPlayer0 = 0
        numberOfWinsForPlayer1 = 0
        numberOfDraws = 0
        for simulationNdx in range(args.numberOfSimulationsPerPosition):
            (positionsList, winner) = winRatesRegression.SimulateAGame(population, encoder, authority,
                                                                       startingPosition=startingPosition,
                                                                       nextPlayer=playersList[1],
                                                                       playerToEpsilonDict={playersList[0]: args.epsilon,
                                                                                    playersList[1]: args.epsilon},
                                                                       encodingPreprocessor=preprocessor)
            if winner == playersList[0]:
                numberOfWinsForPlayer0 += 1
            elif winner == playersList[1]:
                numberOfWinsForPlayer1 += 1
            elif winner == 'draw':
                numberOfDraws += 1
            else:
                raise ValueError("Unknown winner '{}'".format(winner))
            # print ("positionsList = \n{}\nwinner = {}".format(positionsList, winner))
        player0WinRate = numberOfWinsForPlayer0/args.numberOfSimulationsPerPosition
        player1WinRate = numberOfWinsForPlayer1/args.numberOfSimulationsPerPosition
        drawRate = numberOfDraws/args.numberOfSimulationsPerPosition
        logging.info("winRateForPlayer0 = {}; drawRate = {}; winRateForPlayer1 = {}".format(
            player0WinRate, drawRate, player1WinRate ))

        #positionList = startingPosition.flatten().tolist()
        positionEncoding = encoder.Encode(startingPosition.unsqueeze(0)).flatten().tolist()
        print ("positionEncoding = {}".format(positionEncoding))
        for encodingNdx in range(len(positionEncoding)):
            outputFile.write("{},".format(positionEncoding[encodingNdx]))
        outputFile.write("{},{},{}\n".format(player0WinRate, drawRate, player1WinRate))


if __name__ == '__main__':
    main()