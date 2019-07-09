import sys
sys.path.append('../')

from core.graphlightweight import TFTokenExplorer
from config.config import LightWeightMethodArgParser, LightWeightMethodConfig


def run_lightweight_method(args):

    config = LightWeightMethodConfig(LightWeightMethodArgParser().get_args(args))
    explorer = TFTokenExplorer(config)
    explorer.explore_code_repository()


if __name__ == "__main__":

    # append argument -h to see more options
    # ex1: python script_run_lightweight_method.py -ipt ../test/fashion_mnist
    # ex2: python script_run_lightweight_method.py -ipt ../test/VGG16 -opt 3 (get RDF graphs)
    # ex3: python script_run_lightweight_method.py -ipt ../test/Alexnet -opt 2 (get call trees)
    # ex4: python script_run_lightweight_method.py -ipt ../test/Xception
    # ex5: python script_run_lightweight_method.py -ipt ../test/NeuralStyle
    run_lightweight_method(sys.argv[1:])
