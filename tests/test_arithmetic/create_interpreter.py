import genprog.core as gp
import genprog.domains.arithmetic as gparithm
import xml.etree.ElementTree as ET

def main():
    print ("create_interpreter.py main()")

    domainFunctionsTree: ET.ElementTree = ET.parse('../../src/genprog/domains/arithmetic.xml')
    interpreter: gparithm.ArithmeticInterpreter = gparithm.ArithmeticInterpreter(domainFunctionsTree)
    x: float = interpreter.FunctionDefinition('addition_float', [5.4, -3.1])
    print ("x = {}".format(x))

if __name__ == '__main__':
    main()