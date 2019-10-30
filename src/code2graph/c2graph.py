import os
from code2graph.script import script_lightweight
import shlex

# The following command runs the script on input code repository and generates an RDF graph in turtle format.
# The generated RDF graph file is placed inside the input code repository path.



def run(code_repository_path, outputFolder):

    c2g_output_dir = outputFolder + "\code2graph"

    if not os.path.exists(c2g_output_dir):
        os.makedirs(c2g_output_dir)

    # command_6 = "python script_lightweight.py -ip %s -opt 6 --arg --url" % code_repository_path
    # argv = shlex.split(command_6)
    # script_lightweight.pipeline_the_lightweight_approach(argv[2:])

    command_3 = "python script_lightweight.py -ip %s -opt 3 --arg --url" % code_repository_path
    argv = shlex.split(command_3)
    script_lightweight.pipeline_the_lightweight_approach(argv[2:])
    
    try:
        os.remove(c2g_output_dir + "/code2graph.rdf")
    except:
        pass
    
    try:
        os.remove(c2g_output_dir + "/main.html")
    except:
        pass
        
    try:
        os.remove(c2g_output_dir + "/metadata.html")
    except:
        pass
        
    try:
        os.remove(c2g_output_dir + "/generate_cifar10_tfrecords.html")
    except:
        pass

    os.rename(code_repository_path + "/rdf_graph.rdf", c2g_output_dir + "/code2graph.rdf")
    os.rename(code_repository_path + "/main.html", c2g_output_dir + "/main.html")
    os.rename(code_repository_path + "/metadata.html", c2g_output_dir + "/metadata.html")
    os.rename(code_repository_path + "/generate_cifar10_tfrecords.html", c2g_output_dir + "/generate_cifar10_tfrecords.html")
    
    print("[Info] Completed code2graph pipeline!")
