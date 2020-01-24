import os
from code2graph.script import script_lightweight
import shlex
import glob
import pandas as pd

def run(codeFolder, outputFolder, ontology_path, inputCSV):

    input_df = pd.read_csv(inputCSV)

    all_code_folders = [dI for dI in os.listdir(codeFolder) if os.path.isdir(os.path.join(codeFolder,dI))]
    for folder_name in all_code_folders:
        paper_id = input_df.loc[input_df["code"] == folder_name]["paper"].values[0]
        # print(paper_id)
        # code_dir = os.path.join(codeFolder, folder_name)
        code_repository_path = codeFolder + "/" + folder_name

        existing_html_files = glob.glob(code_repository_path + '/*.html')
        
        c2g_output_dir = outputFolder + "\\code2graph\\" + folder_name

        if not os.path.exists(c2g_output_dir):
            os.makedirs(c2g_output_dir)

        # command_6 = "python script_lightweight.py -ip %s -opt 6 --arg --url" % code_repository_path
        # argv = shlex.split(command_6)
        # script_lightweight.pipeline_the_lightweight_approach(argv[2:])

        # command_3 = "python script_lightweight.py -ip %s -r -opt 3 --arg --url" % code_repository_path
        command_3 = "python script_lightweight.py -ip %s -opt 3 -ont %s -pid %s --arg --url" % (code_repository_path, ontology_path, paper_id)
        print(command_3)

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
                    os.remove(c2g_output_dir + "/" + html_file_name)
                except:
                    pass
                    
                os.rename(html_file, c2g_output_dir + "/" + html_file_name)
        

        os.rename(code_repository_path + "/rdf_graph.rdf", c2g_output_dir + "/code2graph.ttl")
        # os.rename(code_repository_path + "/main.html", c2g_output_dir + "/main.html")
        # os.rename(code_repository_path + "/metadata.html", c2g_output_dir + "/metadata.html")
        # os.rename(code_repository_path + "/generate_cifar10_tfrecords.html", c2g_output_dir + "/generate_cifar10_tfrecords.html")
        
        print("[Info] Completed code2graph pipeline for " + str(folder_name) + "!")

    print("[Info] Completed code2graph pipeline for all repositories!")