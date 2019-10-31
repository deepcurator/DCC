import os
from code2graph.script import script_lightweight
import shlex
import glob

def run(code_repository_path, outputFolder):

    existing_html_files = glob.glob(code_repository_path + '/*.html')
    
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
        os.remove(c2g_output_dir + "/code2graph.ttl")
    except:
        pass
        
    all_html_files = glob.glob(code_repository_path + '/*.html')
        
    for html_file in all_html_files:
        if html_file not in existing_html_files:
            html_file_name = os.path.basename(html_file)
            try:
                os.remove(c2g_output_dir + html_file_name)
            except:
                pass
                
            os.rename(html_file, c2g_output_dir + "/" + html_file_name)
    

    os.rename(code_repository_path + "/rdf_graph.rdf", c2g_output_dir + "/code2graph.ttl")
    # os.rename(code_repository_path + "/main.html", c2g_output_dir + "/main.html")
    # os.rename(code_repository_path + "/metadata.html", c2g_output_dir + "/metadata.html")
    # os.rename(code_repository_path + "/generate_cifar10_tfrecords.html", c2g_output_dir + "/generate_cifar10_tfrecords.html")
    
    print("[Info] Completed code2graph pipeline!")
