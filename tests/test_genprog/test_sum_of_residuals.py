import logging
from genprog import core as gp, evolution as gpevo
from typing import Dict, List, Tuple, Any, Optional, Set, Union
import create_dataset


logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s %(message)s')


def main() -> None:
    logging.info("test_sum_of_residuals.py main()")
    """dataset: List[Tuple[Dict[str, float], float]] = create_dataset.CreateDataset(prototype='steps',
                                                                                 numberOfSamples=1000, noiseStdDev=0.05)
    create_dataset.SaveDataset(dataset, './outputs/steps.csv')
    """
    variableNameToTypeDict: Dict[str, str] = {'x': 'float'}
    returnType: str = 'float'
    dataset: List[Tuple[Dict[str, float], float]] = create_dataset.LoadDataset('./outputs/steps.csv',
                                                                               variableNameToTypeDict,
                                                                               returnType)
    print (dataset)

if __name__ == '__main__':
    main()