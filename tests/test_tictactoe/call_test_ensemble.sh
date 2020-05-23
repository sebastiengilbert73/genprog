#!/bin/bash
python test_ensemble.py \
	'/home/sebastien/projects/DeepReinforcementLearning/autoencoder/outputs/AutoencoderNet_(2,1,3,3)_[(2,128,1),(2,128,1)]_10_noZeroPadding_tictactoeAutoencoder_600.pth' \
	--ensembleMembersFilepathPrefix='./outputs/residualChampion_' \
	--variableNameToTypeDict="{'p0': 'float', 'p1': 'float', 'p2': 'float', 'p3': 'float', 'p4': 'float', 'p5': 'float', 'p6': 'float', 'p7': 'float', 'p8': 'float', 'p9': 'float'}" \

