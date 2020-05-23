#!/bin/bash
python /home/sebastien/projects/genprog/tests/test_genprog/test_evolution_residuals.py \
	--datasetPrototype='None' \
	--variableNameToTypeDict="{'p0': 'float', 'p1': 'float', 'p2': 'float', 'p3': 'float', 'p4': 'float', 'p5': 'float', 'p6': 'float', 'p7': 'float', 'p8': 'float', 'p9': 'float'}" \
	--datasetFilepath='/home/sebastien/projects/genprog/tests/test_tictactoe/data/tictactoe_random_reward_x0p0069.csv' \
	--numberOfSamples=1000 \
	--noiseStdDev=0.05 \
	--trainingProportion=0.8 \
	--originalPopulationFilepathPrefix='None' \
  --numberOfIndividuals=300 \
  --levelToFunctionProbabilityDict='{0: 1.0, 1: 0.7, 2: 0.7, 3: 0.5}' \
  --proportionOfConstants=0.6 \
  --constantCreationParametersList='[-10, 10]' \
  --numberOfTournamentParticipants=2 \
  --numberOfGenerations=100 \
  --mutationProbability=0.25 \
  --proportionOfNewIndividuals=0.1 \
  --weightForNumberOfElements=0.0001 \
  --learningRate=0.01 \
  --numberOfEpochsPerGeneration=4 \
  --numberOfGenerationsPerResidual=20 \
  --domainPrimitivesFilepath='/home/sebastien/projects/genprog/src/genprog/domains/arithmetics.xml' \


