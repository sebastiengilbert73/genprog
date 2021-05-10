import argparse
import genprog.core as gp
import genprog.domains.arithmetic as gparithm
import xml.etree.ElementTree as ET
import pandas
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s %(message)s')

def main():
    logging.info ("evolve_population_with_backprop.py main()")

    # Hyperparameters
    numberOfIndividuals = 30
    levelToFunctionProbabilityDict = {0: 1, 1: 1, 2: 1, 3: 0.5, 4: 0.5}
    proportionOfConstants = 0.6
    constantCreationParametersList = [-10.0, 10.0]
    variableNameToTypeDict = {'x': 'float'}
    numberOfGenerations = 50
    trainingDataFilepath = './data/samples_1d_train.csv'
    validationDataFilepath = './data/samples_1d_validation.csv'
    testDataFilepath = './data/samples_1d_test.csv'
    returnType = 'float'
    numberOfTournamentParticipants = 2
    mutationProbability = 0.25
    proportionOfNewIndividuals = 0.1
    weightForNumberOfElements = 0.01
    # Backpropagation parameters
    numberOfEpochs = 4
    learningRate = 0.001

    # Create an interpreter
    domainFunctionsTree: ET.ElementTree = ET.parse('../src/genprog/domains/gaussianMixture.xml')
    interpreter: gparithm.ArithmeticInterpreter = gparithm.ArithmeticInterpreter(domainFunctionsTree)

    # Create the population
    population: gparithm.ArithmeticPopulation = gparithm.ArithmeticPopulation()
    population.Generate(
        numberOfIndividuals=numberOfIndividuals,
        interpreter=interpreter,
        returnType='float',
        levelToFunctionProbabilityDict=levelToFunctionProbabilityDict,
        proportionOfConstants=proportionOfConstants,
        constantCreationParametersList=constantCreationParametersList,
        variableNameToTypeDict=variableNameToTypeDict,
        functionNameToWeightDict=None
    )

    # Load the datasets
    train_dataset = gparithm.load_dataset(filepath=trainingDataFilepath,
                                          variableNameToTypeDict=variableNameToTypeDict,
                                          outputName='y',
                                          outputType='float')
    validation_dataset = gparithm.load_dataset(filepath=validationDataFilepath,
                                               variableNameToTypeDict=variableNameToTypeDict,
                                               outputName='y',
                                               outputType='float')
    test_dataset = gparithm.load_dataset(filepath=testDataFilepath,
                                         variableNameToTypeDict=variableNameToTypeDict,
                                         outputName='y',
                                         outputType='float')

    # Output monitoring file
    generationsCostFile = open('./outputs/generationsCost.csv', "w", buffering=1)  # Flush the buffer at each line
    generationsCostFile.write(
        "generation,championTrainingCost,championValidationCost,medianTrainingCost,medianValidationCost\n")

    training_individualToCostDict = None

    for generationNdx in range(1, numberOfGenerations + 1):
        logging.info("*** Generation {} ***".format(generationNdx))

        training_individualToCostDict = population.NewGenerationWithTournament(
            train_dataset,
            variableNameToTypeDict,
            interpreter,
            returnType,
            numberOfTournamentParticipants,
            mutationProbability,
            training_individualToCostDict,
            proportionOfConstants,
            levelToFunctionProbabilityDict,
            None,
            constantCreationParametersList,
            proportionOfNewIndividuals,
            weightForNumberOfElements
        )

        logging.info("Training each individual with backpropagation...")
        for individualNdx in range(len(population._individualsList)):
            individual = population._individualsList[individualNdx]
            for epoch in range(1, numberOfEpochs + 1):
                #logging.info("Epoch {}".format(epoch))
                individual = interpreter.EpochOfTraining(
                    individual,
                    variableNameToTypeDict,
                    returnType,
                    train_dataset,
                    learningRate
                )
            # Replace the individual
            population._individualsList[individualNdx] = individual
            print ('.', end='', flush=True)
        print('')

        (trainingChampion, championTrainingCost) = population.Champion(training_individualToCostDict)
        medianTrainingCost = population.MedianCost(training_individualToCostDict)


        # Validation
        validation_individualToCostDict = population.EvaluateIndividualCosts(
                                inputOutputTuplesList=validation_dataset,
                                variableNameToTypeDict=variableNameToTypeDict,
                                interpreter=interpreter,
                                returnType=returnType,
                                weightForNumberOfElements=weightForNumberOfElements)
        (validationChampion, championValidationCost) = population.Champion(validation_individualToCostDict)
        medianValidationCost = population.MedianCost(validation_individualToCostDict)

        logging.info("championTrainingCost = {};     medianTrainingCost = {}".format(championTrainingCost, medianTrainingCost))
        logging.info("championValidationCost = {};   medianValidationCost = {}".format(championValidationCost, medianValidationCost))

        # Output to cost monitoring file
        generationsCostFile.write(
            str(generationNdx) + ',' + str(championTrainingCost) + ',' + str(championValidationCost) + ',' + str(
                medianTrainingCost) + ',' + str(medianValidationCost) + '\n')

        # Comparison file
        comparisonFile = open('./outputs/comparison.csv', 'w', buffering=1)
        comparisonFile.write('x,target,prediction\n')
        for xTarget in validation_dataset:
            x = xTarget[0]['x']
            target = xTarget[1]
            prediction = interpreter.Evaluate(validationChampion, variableNameToTypeDict, xTarget[0], returnType)
            comparisonFile.write(str(x) + ',' + str(target) + ',' + str(prediction) + '\n')

if __name__ == '__main__':
    main()