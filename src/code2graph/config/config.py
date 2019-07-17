from argparse import ArgumentParser
import os

try:
    from pathlib import Path
except ImportError:
    from pathlib2 import Path

class LightWeightMethodArgParser:
    '''
        config class argument parser used solely for lightweight method.
    '''

    def __init__(self):
        self.parser = ArgumentParser(
            description='The parameters for the Lightweight Approach.')

        # default code_path is pointed to fashion mnist example.
        self.parser.add_argument('-ipt', dest='code_path',
                                 default='../test/fashion_mnist', type=str,
                                 help='Path to the source code. Default: ../test/fashion_mnist')
        # python interpreter version, default: python 3
        self.parser.add_argument('-pyver', default=3, type=int, choices={2, 3}, help='Python interpreter version.')
        self.parser.add_argument('-opt', dest='output_types',
                                 metavar='N', type=int,
                                 nargs='+', choices={1, 2, 3, 4, 5},
                                 default={1},
                                 help='Types of output: 1 = call graph, 2 = call tress, 3 = RDF graphs, 4 = TensorFlow sequences, 5 = Extract triples.')
        self.parser.add_argument('--arg', dest='show_arg',
                                 action='store_true',
                                 help='Show arguments on graph')
        self.parser.set_defaults(Pshow_arg=False)
        self.parser.add_argument('--url', dest='show_url',
                                 action='store_true',
                                 help='Show url on graph')
        self.parser.set_defaults(show_url=False)

    def get_args(self, args):
        return self.parser.parse_args(args)


class LightWeightMethodConfig:
    '''
        config class controllable in lightweight method.
    '''

    def __init__(self, arg):
        self.code_path = Path(arg.code_path).resolve()
        self.pyversion = arg.pyver
        self.output_types = arg.output_types
        self.show_arg = arg.show_arg
        self.show_url = arg.show_url


class GenerateSummaryArgParser:
    '''
        Argument Parser for script to generate summary file.
    '''

    def __init__(self):
        self.parser = ArgumentParser(
            description='The parameters for script to generate summary file.')
        self.parser.add_argument('-p', '--path',
                                 type=str,
                                 help="Path to code directory.")

    def get_args(self):
        return self.parser.parse_args()


class GraphHandlerArgParser:
    '''
        Argument Parser for GraphHandler.
    '''

    def __init__(self):
        self.parser = ArgumentParser(
            description="The parameters for tf.graph handler.")
        default_path = Path(".")
        default_path = default_path.resolve()

        self.parser.add_argument('-ld', '--logdir',
                                 default=default_path,
                                 type=str,
                                 help='directory for saved graph')

    def get_args(self):
        return self.parser.parse_args()

class PWCConfigArgParser:

    '''
        Argument Parser for Paperswithcode service.
    '''

    def __init__(self):
        self.parser = ArgumentParser(description="The parameters for PWC service.")
        
        self.parser.add_argument('-cp', dest="chromedriver", default="./chromedriver",  type=str, help='path of chromedriver.')
        self.parser.add_argument('-sp', dest="savedpath", default="./data", type=str, help="path of storing data.")

    def get_args(self, args):
        return self.parser.parse_args(args)


class PWCConfig:
    ''' Config for Paperswithcode service '''

    def __init__(self, args):
        self.chrome_driver_path = Path(args.chromedriver)
        self.chrome_driver_path = str(self.chrome_driver_path.resolve())

        self.storage_path = Path(args.savedpath)

        self.tot_paper_to_scrape_per_shot = 1