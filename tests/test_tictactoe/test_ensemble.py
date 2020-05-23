import argparse
import logging
import genprog.core as gp
import genprog.evolution as gpevo
import autoencoder.position
import tictactoe
import ast

parser = argparse.ArgumentParser()
parser.add_argument('autoencoder', help='The filepath to the autoencoder')
parser.add_argument('--ensembleMembersFilepathPrefix', help="The filepath prefix of the ensemble members. Default: './outputs/residualChampion_'", default='./outputs/residualChampion_')
parser.add_argument('--variableNameToTypeDict',
                    help="The dictionary of variable names to their types. Default: {'p0': 'float', 'p1': 'float', 'p2': 'float', 'p3': 'float', 'p4': 'float', 'p5': 'float', 'p6': 'float', 'p7': 'float', 'p8': 'float', 'p9': 'float'}",
                    default="{'p0': 'float', 'p1': 'float', 'p2': 'float', 'p3': 'float', 'p4': 'float', 'p5': 'float', 'p6': 'float', 'p7': 'float', 'p8': 'float', 'p9': 'float'}")
parser.add_argument('--variableScalingFactors',
                    help='The list of scaling factors',
                    default="[0.0069, 0.0069, 0.0069, 0.0069, 0.0069, 0.0069, 0.0069, 0.0069, 0.0069, 0.0069]")

args = parser.parse_args()
variableScalingFactors = ast.literal_eval(args.variableScalingFactors)

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s %(message)s')

def main():
    logging.info("test_ensemble.py main()")
    residualsPopulation = gpevo.ArithmeticsPopulation()
    residualsPopulation.LoadIndividuals(args.ensembleMembersFilepathPrefix)

    # Create the autoencoder
    encoder = autoencoder.position.Net()
    encoder.Load(args.autoencoder)

    # Game authority
    authority = tictactoe.Authority()
    position = authority.InitialPosition()

    # Encode the position
    encoding = encoder.Encode(position.unsqueeze(0))
    # Scale the encoding
    scaledEncoding = []
    for valueNdx in range(len(variableScalingFactors)):
        value = encoding[0][valueNdx].item() * variableScalingFactors[valueNdx]
        scaledEncoding.append(value)

    print ("scaledEncoding = {}".format(scaledEncoding))
if __name__ == '__main__':
    main()