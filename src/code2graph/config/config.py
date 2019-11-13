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

class PyKG2VecArgParser:
    """The class implements the argument parser for the pykg2vec script.

    KGEArgParser defines all the necessary arguements for the global and local 
    configuration of all the modules.

    Attributes:
        general_group (object): It parses the general arguements used by most of the modules.
        general_hyper_group (object): It parses the arguments for the hyper-parameter tuning.
        SME_group (object): It parses the arguments for SME and KG2E algorithms.
        conv_group (object): It parses the arguments for convE algorithms.
        misc_group (object): It prases other necessary arguments.
    
    Examples:
        >>> from pykg2vec.config.config import KGEArgParser
        >>> args = KGEArgParser().get_args()
    """

    def __init__(self):
        self.parser = ArgumentParser(description='Knowledge Graph Embedding tunable configs.')

        ''' basic configs '''
        self.general_group = self.parser.add_argument_group('Generic')

        self.general_group.add_argument('-mn', dest='model_name', default='TransE', type=str, help='Name of model')
        self.general_group.add_argument('-db', dest='debug', default=False, type=lambda x: (str(x).lower() == 'true'),
                                        help='To use debug mode or not.')
        self.general_group.add_argument('-ghp', dest='golden', default=False, type=lambda x: (str(x).lower() == 'true'),
                                        help='Use Golden Hyper parameters!')
        self.general_group.add_argument('-ds', dest='dataset_name', default='lightweight_triples', type=str, 
                                        help='The dataset name.')
        self.general_group.add_argument('-dsp', dest='triples_path', default=None, type=str, 
                                        help='The path to output triples from lightweight method.')
        self.general_group.add_argument('-ld', dest='load_from_data', default=False,
                                        type=lambda x: (str(x).lower() == 'true'), help='load_from_data!')
        self.general_group.add_argument('-sv', dest='save_model', default=True,
                                        type=lambda x: (str(x).lower() == 'true'), help='Save the model!')

        ''' arguments regarding hyperparameters '''
        self.general_hyper_group = self.parser.add_argument_group('Generic Hyperparameters')
        self.general_hyper_group.add_argument('-b', dest='batch_training', default=128, type=int,
                                              help='training batch size')
        self.general_hyper_group.add_argument('-bt', dest='batch_testing', default=16, type=int,
                                              help='testing batch size')
        self.general_hyper_group.add_argument('-mg', dest='margin', default=0.8, type=float, help='Margin to take')
        self.general_hyper_group.add_argument('-opt', dest='optimizer', default='adam', type=str,
                                              help='optimizer to be used in training.')
        self.general_hyper_group.add_argument('-s', dest='sampling', default='uniform', type=str,
                                              help='strategy to do negative sampling.')
        self.general_hyper_group.add_argument('-l', dest='epochs', default=100, type=int,
                                              help='The total number of Epochs')
        self.general_hyper_group.add_argument('-tn', dest='test_num', default=1000, type=int,
                                              help='The total number of test triples')
        self.general_hyper_group.add_argument('-ts', dest='test_step', default=10, type=int, help='Test every _ epochs')
        self.general_hyper_group.add_argument('-lr', dest='learning_rate', default=0.01, type=float,
                                              help='learning rate')
        self.general_hyper_group.add_argument('-k', dest='hidden_size', default=50, type=int,
                                              help='Hidden embedding size.')
        self.general_hyper_group.add_argument('-km', dest='ent_hidden_size', default=50, type=int,
                                              help="Hidden embedding size for entities.")
        self.general_hyper_group.add_argument('-kr', dest='rel_hidden_size', default=50, type=int,
                                              help="Hidden embedding size for relations.")
        self.general_hyper_group.add_argument('-l1', dest='l1_flag', default=True,
                                              type=lambda x: (str(x).lower() == 'true'),
                                              help='The flag of using L1 or L2 norm.')
        self.general_hyper_group.add_argument('-c', dest='C', default=0.0125, type=float,
                                              help='The parameter C used in transH.')

        ''' arguments regarding SME and KG2E '''
        self.SME_group = self.parser.add_argument_group('SME KG2E function selection')
        self.SME_group.add_argument('-func', dest='function', default='bilinear', type=str,
                                    help="The name of function used in SME model.")
        self.SME_group.add_argument('-cmax', dest='cmax', default=0.05, type=float,
                                    help="The parameter for clipping values for KG2E.")
        self.SME_group.add_argument('-cmin', dest='cmin', default=5.00, type=float,
                                    help="The parameter for clipping values for KG2E.")

        ''' arguments regarding TransG '''
        self.TransG_group = self.parser.add_argument_group('TransG function selection')
        self.TransG_group.add_argument('-th', dest='training_threshold', default=3.5, type=float,
                                    help="Training Threshold for updateing the clusters.")
        self.TransG_group.add_argument('-nc', dest='ncluster', default=4, type=int,
                                       help="Number of clusters")
        self.TransG_group.add_argument('-crp', dest='crp_factor', default=0.01, type=float,
                                       help="Chinese Restaurant Process Factor.")
        self.TransG_group.add_argument('-stb', dest='step_before', default=10, type=int,
                                       help="Steps before")
        self.TransG_group.add_argument('-wn', dest='weight_norm', default=False,
                                              type=lambda x: (str(x).lower() == 'true'),
                                       help="normalize the weights!")

        ''' for conve '''
        self.conv_group = self.parser.add_argument_group('ConvE specific Hyperparameters')
        self.conv_group.add_argument('-lmda', dest='lmbda', default=0.1, type=float, help='The lmbda used in ConvE.')
        self.conv_group.add_argument('-fmd', dest='feature_map_dropout', default=0.2, type=float,
                                     help="feature map dropout value used in ConvE.")
        self.conv_group.add_argument('-idt', dest="input_dropout", default=0.3, type=float,
                                     help="input dropout value used in ConvE.")
        self.conv_group.add_argument('-hdt', dest="hidden_dropout", default=0.3, type=float,
                                     help="hidden dropout value used in ConvE.")
        self.conv_group.add_argument('-hdt2', dest="hidden_dropout2", default=0.3, type=float,
                                     help="hidden dropout value used in ConvE.")
        self.conv_group.add_argument('-ubs', dest='use_bias', default=True, type=lambda x: (str(x).lower() == 'true'),
                                     help='The boolean indicating whether use biases or not in ConvE.')
        self.conv_group.add_argument('-lbs', dest='label_smoothing', default=0.1, type=float,
                                     help="The parameter used in label smoothing.")
        self.conv_group.add_argument('-lrd', dest='lr_decay', default=0.995, type=float,
                                     help="The parameter for learning_rate decay used in ConvE.")

        '''for convKB'''
        self.convkb_group = self.parser.add_argument_group('ConvKB specific Hyperparameters')
        self.convkb_group.add_argument('-fsize', dest='filter_sizes', default=[1,2,3],nargs='+', type=int, help='Filter sizes to be used in convKB which acts as the widths of the kernals')
        self.convkb_group.add_argument('-fnum', dest='num_filters', default=500, type=int, help='Filter numbers to be used in convKB')
        self.convkb_group.add_argument('-cnum', dest='num_classes', default=2, type=int, help='Number of classes for triples')
        self.convkb_group.add_argument('-snum', dest='sequence_length', default=3, type=int, help='Sequence length or height of the convolution kernel')
        self.convkb_group.add_argument('-istrain', dest='is_trainable', default=True, type=lambda x: (str(x).lower() == 'true'), help='Make parameters trainable')
        self.convkb_group.add_argument('-cinit', dest='useConstantInit', default=False, type=lambda x: (str(x).lower() == 'true'), help='Use constant initialization')


        ''' others '''
        self.misc_group = self.parser.add_argument_group('MISC')
        self.misc_group.add_argument('-t', dest='tmp', default='../intermediate', type=str,
                                     help='The folder name to store trained parameters.')
        self.misc_group.add_argument('-r', dest='result', default='../results', type=str,
                                     help="The folder name to save the results.")
        self.misc_group.add_argument('-fig', dest='figures', default='../figures', type=str,
                                     help="The folder name to save the figures.")
        self.misc_group.add_argument('-plote', dest='plot_embedding', default=False,
                                     type=lambda x: (str(x).lower() == 'true'), help='Plot the entity only!')
        self.misc_group.add_argument('-plot', dest='plot_entity_only', default=True,
                                     type=lambda x: (str(x).lower() == 'true'), help='Plot the entity only!')
        self.misc_group.add_argument('-gp', dest='gpu_frac', default=0.8, type=float, help='GPU fraction to use')

    def get_args(self, args):
      """This function parses the necessary arguments.

      This function is called to parse all the necessary arguments. 

      Returns:
          object: ArgumentParser object.
      """
      return self.parser.parse_args(args)