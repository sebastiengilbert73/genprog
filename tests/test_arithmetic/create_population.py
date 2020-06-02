import genprog.core as gp
import genprog.domains.arithmetic as gparithm
import xml.etree.ElementTree as ET


def main():
    print ("create_population.py main()")

    domainFunctionsTree: ET.ElementTree = ET.parse('../../src/genprog/domains/arithmetic.xml')
    interpreter: gparithm.ArithmeticInterpreter = gparithm.ArithmeticInterpreter(domainFunctionsTree)

    population: gparithm.ArithmeticPopulation = gparithm.ArithmeticPopulation()
    population.Generate(
        numberOfIndividuals=100,
        interpreter=interpreter,
        returnType='float',
        levelToFunctionProbabilityDict={0: 1, 1: 1, 2: 1, 3: 0.5, 4: 0.5},
        proportionOfConstants=0.6,
        constantCreationParametersList=[-10.0, 10.0],
        variableNameToTypeDict={'x': 'float'},
        functionNameToWeightDict=None
    )


if __name__ == '__main__':
    main()