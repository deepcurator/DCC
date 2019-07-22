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
        self.parser.add_argument('--ds', dest='is_dataset', 
                                 action='store_true',
                                 help='The provided path is the path to a collection of repositories.')
        self.parser.set_defaults(is_dataset=False)
        self.parser.add_argument('-dp', '--dest_path',
                                 default='../rdf', type=str,
                                 help='Path to store generated triples/graphs.')
        self.parser.add_argument('--ct', dest='combined_triples_only',
                                 action='store_true',
                                 help='Only save the combined_triples in destination path.')
        self.parser.set_defaults(combined_triples_only=False)
        self.parser.add_argument('-opt', dest='output_types',
                                 metavar='N', type=int,
                                 nargs='+', choices={1, 2, 3, 4, 5, 6},
                                 default={5},
                                 help='Types of output: 1 = call graph, 2 = call tress, 3 = RDF graphs, 4 = TensorFlow sequences, 5 = Extract triples, 6 = Export RDF (turtle format).')
        self.parser.add_argument('--arg', dest='show_arg',
                                 action='store_true',
                                 help='Show arguments on graph.')
        self.parser.set_defaults(show_arg=False)
        self.parser.add_argument('--url', dest='show_url',
                                 action='store_true',
                                 help='Show url on graph.')
        self.parser.set_defaults(show_url=False)

    def get_args(self, args):
        return self.parser.parse_args(args)


class LightWeightMethodConfig:
    '''
        config class controllable in lightweight method.
    '''

    def __init__(self, arg):
        self.code_path = Path(arg.code_path).resolve()
        self.is_dataset = arg.is_dataset
        self.dest_path = Path(arg.dest_path).resolve()
        self.combined_triples_only = arg.combined_triples_only
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
        
        self.parser.add_argument('-cp', dest="chromedriver", default="../core/chromedriver",  type=str, help='path of chromedriver.')
        self.parser.add_argument('-sp', dest="savedpath", default="./data", type=str, help="path of storing data.")
        self.parser.add_argument('-cred', dest="cred_path", default="../config/credentials.cfg", type=str, help='Path to .cfg file with email credentials.' )

    def get_args(self, args):
        return self.parser.parse_args(args)


class PWCConfig:
    ''' Config for Paperswithcode service '''

    def __init__(self, args):
        self.chrome_driver_path = Path(args.chromedriver)
        self.chrome_driver_path = str(self.chrome_driver_path.resolve())
        self.cred_path = str(Path(args.cred_path).resolve())
        self.storage_path = Path(args.savedpath)
        self.tot_paper_to_scrape_per_shot = 1