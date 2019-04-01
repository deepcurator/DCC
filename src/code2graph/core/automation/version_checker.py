import subprocess

# these scripts should run on linux environment.

def check_python_2(path):
	py2_result = False
	proc = subprocess.Popen(["./py2_checker.py %s"%path], stdout=subprocess.PIPE, shell=True)
	(out, err) = proc.communicate()

	for line in out.decode('utf-8').split('\n'):
		if "Result" and "1" in line:
			if "1" in line:
				py2_result = True

	return py2_result

def check_python_3(path):	
	py3_result = False
	proc = subprocess.Popen(["./py3_checker.py %s"%path], stdout=subprocess.PIPE, shell=True)
	(out, err) = proc.communicate()

	for line in out.decode('utf-8').split('\n'):
		if line.startswith("Result"):
			if "True" in line:
				py3_result = True

	return py3_result

def check(path):
	print("Py2?:", check_python_2(path))
	print("Py3?:", check_python_3(path))

if __name__ == "__main__":
	check("/Users/louisccc/Dropbox/louis_research/development/darpa/DARPA-DCC/data/data_paperswithcode/09/NPRF-master")