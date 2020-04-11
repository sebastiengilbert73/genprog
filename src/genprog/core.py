import logging
import abc
import xml.etree.ElementTree as ET
from typing import Dict, List, Any, Set, Optional, Union
import ast
import math
import random
import xml.dom.minidom


class Individual():
    """
    Abstract class an individual class must inherit
    """

    def __init__(self, tree: ET.ElementTree):
        super().__init__()
        self._tree = tree

    """@abc.abstractmethod
    def Cost(self, inputData: Any, targetOutputData: Any) -> float:
        pass
    """

    def Save(self, filepath: str):
        rootElm: ET.Element = self._tree.getroot()
        treeStr: str = prettify(rootElm)
        with open(filepath, 'w') as file:
            file.write(treeStr)

    def Elements(self) -> List[ET.Element]:
        elementsList: List[ET.Element] = self.SubTreeElements(list(self._tree.getroot())[0])
        return elementsList

    def SubTreeElements(self, headElm: ET.Element) -> List[ET.Element]:
        subTreeElementsList: List[ET.Element] = [headElm]
        for childElm in list(headElm):
            childSubTreeElementsList: List[ET.Element] = self.SubTreeElements(childElm)
            subTreeElementsList.extend(childSubTreeElementsList)
        return subTreeElementsList




def prettify(elem): # Cf. https://stackoverflow.com/questions/17402323/use-xml-etree-elementtree-to-print-nicely-formatted-xml-files
    """Return a pretty-printed XML string for the Element.
    """
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = xml.dom.minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="\t")

def LoadIndividual(filepath: str) -> Individual:
    tree = ET.parse(filepath)
    individual: Individual = Individual(tree)
    return individual

class FunctionSignature():
    def __init__(self, parameterTypesList: List[str], returnType: str) -> None:
        self._parameterTypesList = parameterTypesList
        self._returnType = returnType





class Interpreter(abc.ABC):
    def __init__(self, domainFunctionsTree: ET.ElementTree) -> None:
        super().__init__()
        _domainFunctionsTree = domainFunctionsTree

        # Create dictionaries from function name to parameters
        self._functionNameToSignatureDict: Dict[str, FunctionSignature] = {}

        # Check that each function name is unique
        root: ET.Element = _domainFunctionsTree.getroot()
        functionNamesSet: Set = set()
        for rootChild in root:
            #print("rootChild = {}".format(rootChild))
            #print ("rootChild.tag = {}".format(rootChild.tag))
            if rootChild.tag == 'function':
                functionNameElm: Optional[ET.Element] = rootChild.find('name')
                if functionNameElm is None:
                    raise ValueError("genetic_programming.py Interpreter.__init__(): A function doesn't have a <name> element")
                functionName: Optional[str] = functionNameElm.text
                if functionName is None:
                    raise ValueError(
                        "genetic_programming.py Interpreter.__init__(): A function have an empty <name> element")
                if functionName in functionNamesSet:
                    raise ValueError("genetic_programming.py Interpreter.__init__(): The function name '{}' is encountered more than once in the domain functions tree".format(functionName))
                functionNamesSet.add(functionName)

                parameterTypesElm: Optional[ET.Element] = rootChild.find('parameter_types')
                if parameterTypesElm is None:
                    raise ValueError(
                        "genetic_programming.py Interpreter.__init__(): The function {} doesn't have a <parameter_types> element".format(functionName))
                parameterTypesListStr: Optional[str] = parameterTypesElm.text
                if parameterTypesListStr is None:
                    raise ValueError("genetic_programming.py Interpreter.__init__(): The function {} have an empty <parameter_types> element.".format(
                        functionName))
                parameterTypesListStr = parameterTypesListStr.replace(' ', '')
                parameterTypesListStr = parameterTypesListStr.replace('[', "['")
                parameterTypesListStr = parameterTypesListStr.replace(']', "']")
                parameterTypesListStr = parameterTypesListStr.replace(',', "','")
                #print ("genetic_programming.py Interpreter.__init__(): parameterTypesListStr = {}".format(parameterTypesListStr))
                parameterTypesList = ast.literal_eval(parameterTypesListStr)
                #print ("genetic_programming.py Interpreter.__init__(): parameterTypesList = {}".format(parameterTypesList))

                returnTypeElm: Optional[ET.Element] = rootChild.find('return_type')
                if returnTypeElm is None:
                    raise ValueError(
                        "genetic_programming.py Interpreter.__init__(): The function {} doesn't have a <return_type> element".format(functionName))
                returnType: Optional[str] = returnTypeElm.text
                if returnType is None:
                    raise ValueError(
                        "genetic_programming.py Interpreter.__init__(): The function {} have an empty <return_type> element".format(
                            functionName))

                signature: FunctionSignature = FunctionSignature(parameterTypesList, returnType)
                self._functionNameToSignatureDict[functionName] = signature
            else:
                raise ValueError("genetic_programming.py Interpreter.__init__(): An child of the root element has tag '{}'".format(rootChild.tag))

    def TypeConverter(self, type: str, value: str) -> Any:
        if type == 'float':
            return float(value)
        elif type == 'int':
            return int(value)
        elif type == 'bool':
            if value.lower() in ['true', 'yes']:
                return True
            else:
                return False
        elif type == 'string':
            return value
        else:
            raise NotImplementedError("Interpreter.TypeConverter(): The type {} is not implemented".format(type))

    def Evaluate(self, individual: Individual, variableNameToTypeDict: Dict[str, str], variableNameToValueDict: Dict[str, Any],
                        expectedReturnType: Any) -> Any:
        individualRoot: ET.Element = individual._tree.getroot()
        if len(list(individualRoot)) != 1:
            raise ValueError("Interpreter.Evaluate(): The root has more than one children ({})".format(len(list(individualRoot))))
        headElement = list(individualRoot)[0]
        return self.EvaluateElement(headElement, variableNameToTypeDict, variableNameToValueDict, expectedReturnType)

    def EvaluateElement(self, element: ET.Element, variableNameToTypeDict: Dict[str, str], variableNameToValueDict: Dict[str, Any],
                        expectedReturnType: Any) -> Any:
        childrenList: List[ET.Element] = list(element)
        elementTag = element.tag

        if elementTag == 'constant':
            valueStr: Optional[str] = element.text
            if valueStr is None:
                raise ValueError("Interpreter.EvaluateElement(): A constant has no value")
            return self.TypeConverter(expectedReturnType, valueStr)
        elif elementTag == 'variable':
            variableName: Optional[str] = element.text
            if variableName is None:
                raise ValueError("Interpreter.EvaluateElement(): A variable has no name")
            if variableName not in variableNameToValueDict:
                raise KeyError("Interpreter.EvaluateElement(): Variable '{}' doesn't exist as a key in variableNameToValueDict".format(variableName))
            return variableNameToValueDict[variableName]
        else: # Function
            self.CheckIfSignatureMatches(elementTag, childrenList, variableNameToTypeDict, expectedReturnType)
            childrenEvaluationsList: List[Any] = []
            for childNdx in range(len(childrenList)):
                childExpectedReturnType = self._functionNameToSignatureDict[elementTag]._parameterTypesList[childNdx]
                childEvaluation: Any = self.EvaluateElement(childrenList[childNdx], variableNameToTypeDict, variableNameToValueDict, childExpectedReturnType)
                childrenEvaluationsList.append(childEvaluation)
            return self.FunctionDefinition(elementTag, childrenEvaluationsList)

    def EvaluateElements(self, element: ET.Element, variableNameToTypeDict: Dict[str, str], variableNameToValueDict: Dict[str, Any],
                        expectedReturnType: Any, elementToEvaluationDict: Dict[ET.Element, Any]=None) -> Dict[ET.Element, Any]:
        if elementToEvaluationDict is None:
            elementToEvaluationDict = {}
        childrenList: List[ET.Element] = list(element)
        elementTag = element.tag

        if elementTag == 'constant':
            valueStr: Optional[str] = element.text
            if valueStr is None:
                raise ValueError("Interpreter.EvaluateElement(): A constant has no value")
            elementToEvaluationDict[element] = self.TypeConverter(expectedReturnType, valueStr)
            return elementToEvaluationDict
        elif elementTag == 'variable':
            variableName: Optional[str] = element.text
            if variableName is None:
                raise ValueError("Interpreter.EvaluateElement(): A variable has no name")
            if variableName not in variableNameToValueDict:
                raise KeyError("Interpreter.EvaluateElement(): Variable '{}' doesn't exist as a key in variableNameToValueDict".format(variableName))
            elementToEvaluationDict[element] = variableNameToValueDict[variableName]
            return elementToEvaluationDict
        else: # Function
            self.CheckIfSignatureMatches(elementTag, childrenList, variableNameToTypeDict, expectedReturnType)
            childrenEvaluationsList: List[Any] = []
            for childNdx in range(len(childrenList)):
                childExpectedReturnType = self._functionNameToSignatureDict[elementTag]._parameterTypesList[childNdx]
                elementToEvaluationDict = self.EvaluateElements(childrenList[childNdx], variableNameToTypeDict, variableNameToValueDict, childExpectedReturnType,
                                                                elementToEvaluationDict)
                childrenEvaluationsList.append(elementToEvaluationDict[ childrenList[childNdx] ])
            elementToEvaluationDict[element] = self.FunctionDefinition(elementTag, childrenEvaluationsList)
            return elementToEvaluationDict

    def Backpropagate(self, headElement: ET.Element,
                      elementToEvaluationDict: Dict[ET.Element, Any],
                      elementToGradientDict: Dict[ET.Element, Any]=None) -> Dict[ET.Element, Any]:
        if elementToGradientDict is None: # The head of the tree: df1/df1 = 1.0
            elementToGradientDict = {headElement: 1.0}
        childrenList: List[ET.Element] = list(headElement)
        headElementTag = headElement.tag
        if headElement not in elementToGradientDict.keys():
            raise KeyError("Backpropagate(): The head element is not found in elementToGradientDict.keys()")
        headElementGradient = elementToGradientDict[headElement]
        argumentsList: List[Any] = [elementToEvaluationDict[child] for child in childrenList]
        if headElementTag != 'constant' and headElementTag != 'variable':
            childrenPartialDerivatives = self.FunctionDerivative(headElementTag, argumentsList)

            for childNdx in range(len(childrenList)):
                elementToGradientDict[childrenList[childNdx]] = headElementGradient * childrenPartialDerivatives[childNdx]
                elementToGradientDict = self.Backpropagate(childrenList[childNdx],
                                                           elementToEvaluationDict,
                                                           elementToGradientDict)
        return elementToGradientDict

    @abc.abstractmethod
    def FunctionDefinition(self, functionName: str, argumentsList: List[Any]) -> Any:
        pass

    def FunctionDerivative(self, functionName: str, argumentsList: List[Any]) -> List[Any]:
        pass

    @abc.abstractmethod
    def CreateConstant(self, returnType: str, parametersList: Optional[ List[Any] ] ) -> str:
        pass

    @abc.abstractmethod
    def PossibleTypes(self) -> List[str]:
        pass

    def CheckIfSignatureMatches(self, functionName: str, argumentsList: List[ET.Element], variableNameToTypeDict: Dict[str, str],
                                expectedReturnType) -> None:
        if functionName not in self._functionNameToSignatureDict:
            raise KeyError("Interpreter.CheckIfSignatureMatches(): The function name '{}' doesn't exist as a key in self._functionNameToSignatureDict".format(functionName))
        expectedSignature: FunctionSignature = self._functionNameToSignatureDict[functionName]
        if len(argumentsList) != len(expectedSignature._parameterTypesList):
            raise ValueError("Interpreter.CheckIfSignatureMatches(): For function '{}', len(argumentsList) ({}) != len(expectedSignature._parameterTypesList) ({})".format(
                functionName, len(argumentsList), len(expectedSignature._parameterTypesList)
            ))
        for argumentNdx in range(len(argumentsList)):
            argumentTag = argumentsList[argumentNdx].tag
            if argumentTag == 'variable':
                variableName: Optional[str] = argumentsList[argumentNdx].text
                if variableName is None:
                    raise ValueError("Interpreter.CheckIfSignatureMatches(): An variable argument has no name")
                if variableName not in variableNameToTypeDict:
                    raise KeyError("Interpreter.CheckIfSignatureMatches(): The variable name '{}' is not in variableNameToTypeDict".format(
                            variableName))
                argumentType: Optional[str] = variableNameToTypeDict[variableName]
            elif argumentTag == 'constant':
                argumentType = expectedSignature._parameterTypesList[argumentNdx] # Will be cast at evaluation
            else: # Function
                if argumentTag not in self._functionNameToSignatureDict:
                    raise KeyError("Interpreter.CheckIfSignatureMatches(): Function name '{}' doesn't exist as a key in self._functionNameToSignatureDict".format(argumentTag))
                argumentType = self._functionNameToSignatureDict[argumentTag]._returnType

            if argumentType != expectedSignature._parameterTypesList[argumentNdx]:
                raise ValueError("Interpreter.CheckIfSignatureMatches(): Expected paremter type {}, got argument type {}".format(expectedSignature._parameterTypesList[argumentNdx], argumentType))

        if expectedReturnType != expectedSignature._returnType:
            raise ValueError("Interpreter.CheckIfSignatureMatches(): The expected return type ({}) do not match the function signature return type ({})".format(expectedReturnType, expectedSignature._returnType))

    def FunctionsWhoseReturnTypeIs(self, returnType: str):
        functionsNamesList: List[str] = []
        for functionName, signature in self._functionNameToSignatureDict.items():
            if signature._returnType == returnType:
                functionsNamesList.append(functionName)
        return functionsNamesList

    def CreateElement(self, returnType: str,
                      level: int,
                      levelToFunctionProbabilityDict: Dict[int, float],
                      proportionOfConstants: float,
                      functionNameToWeightDict: Dict[str, float],
                      constantCreationParametersList: List[Any],
                      variableNameToTypeDict: Dict[str, str] ) -> ET.Element:
        # Is it a function?
        functionProbability: float = 0.0
        if level in levelToFunctionProbabilityDict:
            functionProbability = levelToFunctionProbabilityDict[level]
        if random.random() < functionProbability: # It is a function
            functionNamesList = self.FunctionsWhoseReturnTypeIs(returnType)
            # Normalize the probabilities
            functionNameToProbabilityDict: Dict[str, float] = {}
            weightsSum: float = 0.
            for functionName in functionNamesList:
                weightsSum += max(functionNameToWeightDict[functionName], 0.)
            if weightsSum == 0:
                raise ValueError("Interpreter.CreateElement(): returnType = {}; level = {}; The sum of weights is 0".format(returnType, level))
            for functionName in functionNamesList:
                functionNameToProbabilityDict[functionName] = max(functionNameToWeightDict[functionName], 0.0) / weightsSum
            randomNbr: float = random.random()
            runningSum: float = 0.0
            theRandomNbrIsReached: bool = False
            for functionName, probability in functionNameToProbabilityDict.items():
                runningSum += probability
                if runningSum >= randomNbr and not theRandomNbrIsReached:
                    element = ET.Element(functionName)
                    signature = self._functionNameToSignatureDict[functionName]
                    for parameterType in signature._parameterTypesList:
                        childElement = self.CreateElement(parameterType, level + 1, levelToFunctionProbabilityDict, proportionOfConstants,
                                                          functionNameToWeightDict, constantCreationParametersList,
                                                          variableNameToTypeDict)
                        element.append(childElement)
                    theRandomNbrIsReached = True
            return element
        else: # A variable or a constant
            # Try a variable
            itMustBeAConstant = False
            candidateVariableNamesList = []
            for variableName, type in variableNameToTypeDict.items():
                if type == returnType:
                    candidateVariableNamesList.append(variableName)
            if len(candidateVariableNamesList) == 0:
                itMustBeAConstant = True

            if itMustBeAConstant or random.random() < proportionOfConstants: # A constant
                element = ET.Element('constant')
                constantValueStr = self.CreateConstant(returnType, constantCreationParametersList)
                element.text = constantValueStr
                return element
            else: # A variable
                element = ET.Element('variable')
                chosenNdx: int = random.randint(0, len(candidateVariableNamesList) - 1)
                element.text = candidateVariableNamesList[chosenNdx]
                return element

    def CreateIndividual(self,
                         returnType: str,
                         levelToFunctionProbabilityDict: Dict[int, float],
                         proportionOfConstants: float,
                         constantCreationParametersList: List[Any],
                         variableNameToTypeDict: Dict[str, str],
                         functionNameToWeightDict: Dict[str, float]=None) -> Individual:
        if functionNameToWeightDict is None: # Create a uniform dictionary
            functionNameToWeightDict = {}
            for functionName, signature in self._functionNameToSignatureDict.items():
                functionNameToWeightDict[functionName] = 1.0
        root: ET.Element = ET.Element('individual')
        headElm: ET.Element = self.CreateElement(
            returnType,
            0,
            levelToFunctionProbabilityDict,
            proportionOfConstants,
            functionNameToWeightDict,
            constantCreationParametersList,
            variableNameToTypeDict
        )
        root.append(headElm)
        individual: Individual = Individual(ET.ElementTree(root))
        return individual

    def Mutate(self,
               individual: Individual,
               levelToFunctionProbabilityDict: Dict[int, float],
               proportionOfConstants: float,
               functionNameToWeightDict: Optional[Dict[str, float] ],
               constantCreationParametersList: List[Any],
               variableNameToTypeDict: Dict[str, str]
               ) -> Individual:
        if functionNameToWeightDict is None:
            functionNameToWeightDict = {}
            for functionName, signature in self._functionNameToSignatureDict.items():
                functionNameToWeightDict[functionName] = 1.0
        # Randomly select an element
        pivotElementReturnType: str = random.choice(self.PossibleTypes())
        candidateElementsList = ElementsWhoseReturnTypeIs(individual, pivotElementReturnType, self, variableNameToTypeDict)
        if len(candidateElementsList) == 0: # Don't mutate
            return individual
        pivotElement: ET.Element = random.choice(candidateElementsList)
        mutatedElement: ET.Element = self.CreateElement(pivotElementReturnType,
                                                        0,
                                                        levelToFunctionProbabilityDict,
                                                        proportionOfConstants,
                                                        functionNameToWeightDict,
                                                        constantCreationParametersList,
                                                        variableNameToTypeDict)

        parent_map: Dict[ET.Element, ET.Element] = {c: p for p in individual._tree.iter() for c in
                                                    p}  # Cf. https://stackoverflow.com/questions/2170610/access-elementtree-node-parent-node
        pivotParentElm = parent_map[pivotElement]
        # What is the child index?
        childNdx: int = -1
        for candidateChildNdx in range(len(list(pivotParentElm))):
            if pivotParentElm[candidateChildNdx] == pivotElement:
                childNdx = candidateChildNdx
        if childNdx == -1:
            raise ValueError(
                "Interpreter.Mutate(): The child element could not be found in its parent children...(?)")
        # Replace the pivot
        pivotParentElm[childNdx] = mutatedElement
        return individual


def ElementsWhoseReturnTypeIs(individual: Individual, returnType: str, interpreter: Interpreter, variableNameToTypeDict: Dict[str, str]):
    elementsList: List[ET.Element] = individual.Elements()
    elementsWithReturnTypeList: List[ET.Element] = []
    for element in elementsList:
        if element.tag == 'variable':
            variableName: Optional[str] = element.text
            if variableName is None:
                raise ValueError("ElementsWhoseReturnTypeIs(): A variable has no name")
            if variableName not in variableNameToTypeDict:
                raise KeyError("ElementsWhoseReturnTypeIs(): The variable name '{}' is not in variableNameToTypeDict")
            variableReturnType: str = variableNameToTypeDict[variableName]
            if variableReturnType == returnType:
                elementsWithReturnTypeList.append(element)
        elif element.tag == 'constant':
            pass # Don't list constants, as their type is not known
        else: # Function
            functionName: str = element.tag
            functionReturnType: str = interpreter._functionNameToSignatureDict[functionName]._returnType
            if functionReturnType == returnType:
                elementsWithReturnTypeList.append(element)
    return elementsWithReturnTypeList








class ArithmeticsInterpreter(Interpreter): # An example to follow for other domains

    def FunctionDefinition(self, functionName: str, argumentsList: List[ Union[float, bool] ]) -> Union[float, bool]:
        if functionName == "addition_float":
            floatArg1: float = float(argumentsList[0])
            floatArg2: float = float(argumentsList[1])
            return floatArg1 + floatArg2
        elif functionName == "subtraction_float":
            floatArg1 = float(argumentsList[0])
            floatArg2 = float(argumentsList[1])
            return floatArg1 - floatArg2
        elif functionName == "multiplication_float":
            floatArg1 = float(argumentsList[0])
            floatArg2 = float(argumentsList[1])
            return floatArg1 * floatArg2
        elif functionName == "division_float":
            floatArg1 = float(argumentsList[0])
            floatArg2 = float(argumentsList[1])
            if floatArg2 == 0:
                return 0.0
            return floatArg1 / floatArg2
        elif functionName == "greaterThan_float":
            floatArg1 = float(argumentsList[0])
            floatArg2 = float(argumentsList[1])
            return floatArg1 > floatArg2
        elif functionName == "greaterThanOrEqual_float":
            floatArg1 = float(argumentsList[0])
            floatArg2 = float(argumentsList[1])
            return floatArg1 >= floatArg2
        elif functionName == "lessThan_float":
            floatArg1 = float(argumentsList[0])
            floatArg2 = float(argumentsList[1])
            return floatArg1 < floatArg2
        elif functionName == "lessThanOrEqual_float":
            floatArg1 = float(argumentsList[0])
            floatArg2 = float(argumentsList[1])
            return floatArg1 <= floatArg2
        elif functionName == "almostEqual_float":
            floatArg1 = float(argumentsList[0])
            floatArg2 = float(argumentsList[1])
            floatArg3: float = abs( float(argumentsList[2]) )
            return abs(floatArg1 - floatArg2) <= floatArg3
        elif functionName == "inverse_bool":
            boolArg1: bool = bool(argumentsList[0])
            return not boolArg1
        elif functionName == "log":
            floatArg1 = float(argumentsList[0])
            if floatArg1 <= 0.0:
                return 0.0
            return math.log(floatArg1)
        elif functionName == "exp":
            floatArg1 = float(argumentsList[0])
            if floatArg1 >= 20.0:
                return 0.0
            return math.exp(floatArg1)
        elif functionName == "pow_float":
            floatArg1 = float(argumentsList[0])
            floatArg2 = float(argumentsList[1])
            try:
                result: float = math.pow(floatArg1, floatArg2)
                return result
            except:
                return 0.0
        elif functionName == 'if_float':
            boolArg1 = bool(argumentsList[0])
            floatArg1 = float(argumentsList[1])
            floatArg2 = float(argumentsList[2])
            if boolArg1:
                return floatArg1
            else:
                return floatArg2
        elif functionName == 'sin':
            try:
                floatArg1 = float(argumentsList[0])
                return math.sin(floatArg1)
            except:
                return 0.0
        elif functionName == 'cos':
            try:
                floatArg1 = float(argumentsList[0])
                return math.cos(floatArg1)
            except:
                return 0.0
        elif functionName == 'tan':
            try:
                floatArg1 = float(argumentsList[0])
                return math.tan(floatArg1)
            except:
                return 0.0
        elif functionName == 'atan':
            floatArg1 = float(argumentsList[0])
            return math.atan(floatArg1)
        elif functionName == 'sigmoid':
            floatArg1 = float(argumentsList[0])
            try:
                return 1.0 / (1.0 + math.exp(-floatArg1))
            except:
                return 0.0
        elif functionName == 'ispositive_float':
            floatArg1 = float(argumentsList[0])
            return floatArg1 >= 0.0
        elif functionName == 'inverse_float':
            floatArg1 = float(argumentsList[0])
            try:
                if floatArg1 == 0:
                    return 0
                else:
                    return 1.0/floatArg1
            except:
                return 0.0
        elif functionName == 'isinbetween_float':
            floatArg1 = float(argumentsList[0])
            floatArg2 = float(argumentsList[1])
            floatArg3 = float(argumentsList[2])
            if floatArg1 >= floatArg2 and floatArg1 <= floatArg3:
                return True
            else:
                return False
        elif functionName == 'abs_float':
            floatArg1 = float(argumentsList[0])
            return abs(floatArg1)
        elif functionName == 'relu_float':
            floatArg1 = float(argumentsList[0])
            if floatArg1 >= 0:
                return floatArg1
            else:
                return 0.0
        elif functionName == 'sign_float':
            floatArg1 = float(argumentsList[0])
            if floatArg1 < 0:
                return -1.0
            elif floatArg1 > 0:
                return 1.0
            else:
                return 0.0
        elif functionName == 'max_float':
            floatArg1 = float(argumentsList[0])
            floatArg2 = float(argumentsList[1])
            return max(floatArg1, floatArg2)
        elif functionName == 'min_float':
            floatArg1 = float(argumentsList[0])
            floatArg2 = float(argumentsList[1])
            return min(floatArg1, floatArg2)
        elif functionName == 'gaussian':
            floatArg1 = float(argumentsList[0])
            floatArg2 = float(argumentsList[1])
            try:
                sigma2 = floatArg2 ** 2
                f = math.exp(-(floatArg1 ** 2)/(2 * sigma2) )
                return f
            except:
                return 0.0
        elif functionName == 'pow2_float':
            floatArg1 = float(argumentsList[0])
            try:
                return floatArg1 ** 2
            except:
                return 0.0
        elif functionName == 'sqrt':
            floatArg1 = float(argumentsList[0])
            if floatArg1 >= 0:
                return math.sqrt(floatArg1)
            else:
                return 0.0
        else:
            raise NotImplementedError("ArithmeticsInterpreter.FunctionDefinition(): Not implemented function '{}'".format(functionName))

    def FunctionDerivative(self, functionName: str, argumentsList: List[ Union[float, bool] ]) -> List[ Union[float, bool] ]:
        if functionName == "addition_float":
            floatArg1: float = float(argumentsList[0])
            floatArg2: float = float(argumentsList[1])
            return [1.0, 1.0]
        elif functionName == "subtraction_float":
            floatArg1 = float(argumentsList[0])
            floatArg2 = float(argumentsList[1])
            return [1.0, -1.0]
        elif functionName == "multiplication_float":
            floatArg1 = float(argumentsList[0])
            floatArg2 = float(argumentsList[1])
            return [floatArg2 , floatArg1]
        else:
            raise NotImplementedError("ArithmeticsInterpreter.FunctionDerivative(): Not implemented function '{}'".format(functionName))
        """
        elif functionName == "division_float":
            floatArg1 = float(argumentsList[0])
            floatArg2 = float(argumentsList[1])
            if floatArg2 == 0:
                return 0.0
            return floatArg1 / floatArg2
        elif functionName == "greaterThan_float":
            floatArg1 = float(argumentsList[0])
            floatArg2 = float(argumentsList[1])
            return floatArg1 > floatArg2
        elif functionName == "greaterThanOrEqual_float":
            floatArg1 = float(argumentsList[0])
            floatArg2 = float(argumentsList[1])
            return floatArg1 >= floatArg2
        elif functionName == "lessThan_float":
            floatArg1 = float(argumentsList[0])
            floatArg2 = float(argumentsList[1])
            return floatArg1 < floatArg2
        elif functionName == "lessThanOrEqual_float":
            floatArg1 = float(argumentsList[0])
            floatArg2 = float(argumentsList[1])
            return floatArg1 <= floatArg2
        elif functionName == "almostEqual_float":
            floatArg1 = float(argumentsList[0])
            floatArg2 = float(argumentsList[1])
            floatArg3: float = abs( float(argumentsList[2]) )
            return abs(floatArg1 - floatArg2) <= floatArg3
        elif functionName == "inverse_bool":
            boolArg1: bool = bool(argumentsList[0])
            return not boolArg1
        elif functionName == "log":
            floatArg1 = float(argumentsList[0])
            if floatArg1 <= 0.0:
                return 0.0
            return math.log(floatArg1)
        elif functionName == "exp":
            floatArg1 = float(argumentsList[0])
            if floatArg1 >= 20.0:
                return 0.0
            return math.exp(floatArg1)
        elif functionName == "pow_float":
            floatArg1 = float(argumentsList[0])
            floatArg2 = float(argumentsList[1])
            try:
                result: float = math.pow(floatArg1, floatArg2)
                return result
            except:
                return 0.0
        elif functionName == 'if_float':
            boolArg1 = bool(argumentsList[0])
            floatArg1 = float(argumentsList[1])
            floatArg2 = float(argumentsList[2])
            if boolArg1:
                return floatArg1
            else:
                return floatArg2
        elif functionName == 'sin':
            try:
                floatArg1 = float(argumentsList[0])
                return math.sin(floatArg1)
            except:
                return 0.0
        elif functionName == 'cos':
            try:
                floatArg1 = float(argumentsList[0])
                return math.cos(floatArg1)
            except:
                return 0.0
        elif functionName == 'tan':
            try:
                floatArg1 = float(argumentsList[0])
                return math.tan(floatArg1)
            except:
                return 0.0
        elif functionName == 'atan':
            floatArg1 = float(argumentsList[0])
            return math.atan(floatArg1)
        elif functionName == 'sigmoid':
            floatArg1 = float(argumentsList[0])
            try:
                return 1.0 / (1.0 + math.exp(-floatArg1))
            except:
                return 0.0
        elif functionName == 'ispositive_float':
            floatArg1 = float(argumentsList[0])
            return floatArg1 >= 0.0
        elif functionName == 'inverse_float':
            floatArg1 = float(argumentsList[0])
            try:
                if floatArg1 == 0:
                    return 0
                else:
                    return 1.0/floatArg1
            except:
                return 0.0
        elif functionName == 'isinbetween_float':
            floatArg1 = float(argumentsList[0])
            floatArg2 = float(argumentsList[1])
            floatArg3 = float(argumentsList[2])
            if floatArg1 >= floatArg2 and floatArg1 <= floatArg3:
                return True
            else:
                return False
        elif functionName == 'abs_float':
            floatArg1 = float(argumentsList[0])
            return abs(floatArg1)
        elif functionName == 'relu_float':
            floatArg1 = float(argumentsList[0])
            if floatArg1 >= 0:
                return floatArg1
            else:
                return 0.0
        elif functionName == 'sign_float':
            floatArg1 = float(argumentsList[0])
            if floatArg1 < 0:
                return -1.0
            elif floatArg1 > 0:
                return 1.0
            else:
                return 0.0
        elif functionName == 'max_float':
            floatArg1 = float(argumentsList[0])
            floatArg2 = float(argumentsList[1])
            return max(floatArg1, floatArg2)
        elif functionName == 'min_float':
            floatArg1 = float(argumentsList[0])
            floatArg2 = float(argumentsList[1])
            return min(floatArg1, floatArg2)
        elif functionName == 'gaussian':
            floatArg1 = float(argumentsList[0])
            floatArg2 = float(argumentsList[1])
            try:
                sigma2 = floatArg2 ** 2
                f = math.exp(-(floatArg1 ** 2)/(2 * sigma2) )
                return f
            except:
                return 0.0
        elif functionName == 'pow2_float':
            floatArg1 = float(argumentsList[0])
            try:
                return floatArg1 ** 2
            except:
                return 0.0
        elif functionName == 'sqrt':
            floatArg1 = float(argumentsList[0])
            if floatArg1 >= 0:
                return math.sqrt(floatArg1)
            else:
                return 0.0
        """



    def CreateConstant(self, returnType: str, parametersList: Optional[List[Union[float, bool] ]]) -> str:
        if returnType == 'float':
            if parametersList is None:
                raise ValueError("ArithmeticsInterpreter.CreateConstant(): returnType = float; There is no list of parameters")
            if len(parametersList) != 2:
                raise ValueError("ArithmeticsInterpreter.CreateConstant(): returnType = float; The length of the list of parameters ({}) is not 2".format(
                    len(parametersList) ))
            minValue = float(parametersList[0])
            maxValue = float(parametersList[1])
            return str( minValue + (maxValue - minValue) * random.random() )
        elif returnType == 'bool':
            if random.random() < 0.5:
                return 'True'
            else:
                return 'False'
        else:
            raise NotImplementedError("ArithmeticsInterpreter.CreateConstant(): Not implemented return type '{}'".format(returnType))

    def PossibleTypes(self) -> List[str]:
        return ['float', 'bool']

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s %(message)s')

    logging.debug("core.py main()")
    domainFunctionsTree: ET.ElementTree = ET.parse('../../domains/arithmetics.xml')
    #print ("domainFunctionsTree = {}".format(domainFunctionsTree))

    individual: Individual = Individual(ET.parse('../../domains/arithmetics_individual_example2.xml') )
    interpreter: ArithmeticsInterpreter = ArithmeticsInterpreter(domainFunctionsTree)
    #print ("interpreter._functionNameToSignatureDict = {}".format(interpreter._functionNameToSignatureDict))

    variableNameToTypeDict: Dict[str, str] = {'x': 'float', 'y': 'float', 'cond': 'bool'}
    variableNameToValueDict: Dict[str, Union[float, bool]] = {'x': 10, 'y': 3.0, 'cond': False}

    output: bool = interpreter.Evaluate(individual, variableNameToTypeDict, variableNameToValueDict, 'bool')
    print ("output = {}".format(output))

    candidateFunctionsList: List[str] = interpreter.FunctionsWhoseReturnTypeIs('bool')
    print ("candidateFunctionsList = {}".format(candidateFunctionsList))

    returnType: str = 'float'
    levelToFunctionProbabilityDict: Dict[int, float] = {0: 1.0, 1: 0.8, 2: 0.5}
    proportionOfConstants: float = 0.5
    functionNameToWeightDict: Dict[str, float] = {
        "addition_float": 1,
        "subtraction_float": 1,
        "multiplication_float": 1,
        "division_float": 1,
        "greaterThan_float": 1,
        "greaterThanOrEqual_float": 1,
        "lessThan_float": 1,
        "lessThanOrEqual_float": 1,
        "almostEqual_float": 1,
        "inverse_bool": 1,
        "log": 1,
        "exp": 1,
        "pow_float": 1,
        "if_float": 1
    }
    constantCreationParametersList: List[Union[float, bool]] = [-100, 100]


    createdIndividual = interpreter.CreateIndividual(
        returnType,
        levelToFunctionProbabilityDict,
        proportionOfConstants,
        constantCreationParametersList,
        variableNameToTypeDict,
        functionNameToWeightDict
    )
    createdIndividual.Save('./outputs/createdIndividual.xml')

    elementsList: List[ET.Element] = ElementsWhoseReturnTypeIs(createdIndividual, 'float', interpreter, variableNameToTypeDict)
    for element in elementsList:
        print("element.tag = {}".format(element.tag))