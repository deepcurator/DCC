import sys

import glob
from pathlib import Path
from rdflib import RDF, RDFS
from sklearn.model_selection import train_test_split

from pykg2vec.utils.kgcontroller import KnowledgeGraph
from pykg2vec.config.config import Importer
from pykg2vec.utils.trainer import Trainer

sys.path.append('../')
from config.config import PyKG2VecArgParser


def save_to_file(data, name, path):
    file_path = Path(path) / name
    with open(str(file_path), "w") as f:
        for line in data:
            f.write(line)


def preprocess(triples_path, save_name):
    data = []
    type_info = []
    triples_path = Path(triples_path).resolve()
    for file in triples_path.rglob("combined_triples.triples"):
        with open(file, "r") as f:
            for line in f:
                if len(line.split('\t')) == 3:
                    # data.append(line)
                    if str(RDF.type) in line or str(RDFS.label) in line:
                        type_info.append(line)
                    else:
                        data.append(line)

    # import pdb; pdb.set_trace()
    test_size = 0.2 / (len(data) / len(data + type_info))
    valid_size = 0.2 / (1 - test_size)

    train, test = train_test_split(data, test_size=test_size)
    train, valid = train_test_split(train, test_size=valid_size)
    train = train + type_info

    Path("../pykg2vec_dataset").mkdir(exist_ok=True)
    save_path = Path("../pykg2vec_dataset/" + save_name).resolve()
    Path(save_path).mkdir(exist_ok=True)
    
    # save_to_file(data, "data.txt", save_path)
    save_to_file(train, save_name + "-train.txt", save_path)
    save_to_file(test, save_name + "-test.txt", save_path)
    save_to_file(valid, save_name + "-valid.txt", save_path)

    return save_path

def main():
    # getting the customized configurations from the command-line arguments.
    args = PyKG2VecArgParser().get_args(sys.argv[1:])
    args.dataset_path = preprocess(args.triples_path, args.dataset_name)
    
    # Preparing data and cache the data for later usage
    knowledge_graph = KnowledgeGraph(dataset=args.dataset_name, negative_sample=args.sampling, custom_dataset_path=args.dataset_path)
    knowledge_graph.prepare_data()

    # Extracting the corresponding model config and definition from Importer().
    config_def, model_def = Importer().import_model_config(args.model_name.lower())
    config = config_def(args=args)
    model = model_def(config)

    # Create, Compile and Train the model. While training, several evaluation will be performed.
    trainer = Trainer(model=model, debug=args.debug)
    trainer.build_model()
    trainer.train_model()


if __name__ == "__main__":
    main()