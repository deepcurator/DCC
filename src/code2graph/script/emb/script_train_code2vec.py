import sys
sys.path.append('../../')
from core.code2vec import *

def main():
	trainer = Trainer('C:\\Users\\AICPS\\Documents\\GitHub\\louisccc-DCC\\src\\code2graph\\graphast_output\\fashion_mnist\\triples_ast_function')
	trainer.build_model()
	trainer.train_model()

	code.interact(local=locals())

if __name__ == "__main__":
	main()