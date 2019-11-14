from argparse import ArgumentParser
import os
from pykg2vec.config.config import KGEArgParser

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
        self.parser.add_argument('-ip', dest='input_path',
                                 default='../test/fashion_mnist', type=str,
                                 help='Path to the source code. Default: ../test/fashion_mnist')
        self.parser.add_argument('-r', dest='recursive', 
                                 action='store_true',
                                 help='Recursively apply Lightweight method on all the papers in the code path.')
        self.parser.set_defaults(recursive=False)
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
                                 help='Types of output: 1 = Call graph, 2 = Call trees, 3 = RDF graph (html format),' 
                                      '4 = TensorFlow sequences, 5 = Extract triples, 6 = RDF graph (turtle format).')
        self.parser.add_argument('-ont', dest='ontology',
                                 default='../core/DeepSciKG.nt', type=str, 
                                 help='Path to ontology manager.')
        self.parser.add_argument('-pid', dest='paper_id',
                                 default=None, type=str, 
                                 help='Paper ID of the input ML paper.')
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
        self.input_path = Path(arg.input_path).resolve()
        self.recursive = arg.recursive
        self.dest_path = Path(arg.dest_path).resolve()
        self.ontology = Path(arg.ontology).resolve()
        self.paper_id = arg.paper_id
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

    def get_args(self, args):
        return self.parser.parse_args(args)


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

    def get_args(self, args):
        return self.parser.parse_args(args)

class PWCConfigArgParser:

    '''
        Argument Parser for Paperswithcode service.
    '''

    def __init__(self):
        self.parser = ArgumentParser(description="The parameters for PWC service.")
        
        self.parser.add_argument('-cp', dest="chromedriver", default="../core/chromedriver",  type=str, help='path of chromedriver.')
        self.parser.add_argument('-sp', dest="save_path", default="./data", type=str, help="path of storing data.")
        self.parser.add_argument('-cred', dest="cred_path", default="../config/credentials.cfg", type=str, help='Path to .cfg file with email credentials.' )

    def get_args(self, args):
        return self.parser.parse_args(args)


class PWCConfig:
    ''' Config for Paperswithcode service '''

    def __init__(self, args):
        self.chrome_driver_path = Path(args.chromedriver)
        self.chrome_driver_path = str(self.chrome_driver_path.resolve())
        self.cred_path = str(Path(args.cred_path).resolve())
        self.storage_path = Path(args.save_path)
        self.tot_paper_to_scrape_per_shot = 1

class GraphASTArgParser:
    '''
        Argument Parser for graphast script.
    '''
    
    def __init__(self):
        self.parser = ArgumentParser(description='The parameters for graphast method.')
        
        self.parser.add_argument('-ip', dest='input_path',
                                 default='../test/fashion_mnist', type=str,
                                 help='Path to the source code. Default: ../test/fashion_mnist')
        self.parser.add_argument('-r', dest='recursive', 
                                 action='store_true',
                                 help='Recursively apply graphast method on all papers in the input path.')
        self.parser.set_defaults(recursive=False)
        self.parser.add_argument('-dp', '--dest_path',
                                 default='../graphast_output', type=str,
                                 help='Path to save output files.')
        self.parser.add_argument('-res', dest='resolution',
                                 default='function', type=str,
                                 help='Processing resolution of the method: function or method. Default: function.')

    def get_args(self, args):
        return self.parser.parse_args(args)


class GraphASTConfig:
    '''
        Config class for graphast method.
    '''

    def __init__(self, arg):
        self.input_path = Path(arg.input_path).resolve()
        self.recursive = arg.recursive
        self.dest_path = Path(arg.dest_path).resolve()
        self.resolution = arg.resolution


class PyKG2VecArgParser (KGEArgParser):
    """The class implements the argument parser for the pykg2vec script."""

    def __init__(self):
        super().__init__()
        self.general_group.set_defaults(dataset_name='lightweight')
        self.general_group.add_argument('-tp', dest='triples_path', default=None, type=str, 
                                        help='The path to output triples from lightweight method.')