from pipreqs import pipreqs
# compatible to Python 2 & 3 
try:
    from pip import main as pipmain
except:
    from pip._internal import main as pipmain

def get_import_list(path):
	candidates = pipreqs.get_all_imports(path)
	candidates = pipreqs.get_pkg_names(candidates)
	imports = pipreqs.get_import_local(candidates)
	return imports

def generate_requirements_txt(path):
	pipreqs.output_requirements(get_import_list(path))

def resolve_requirements(imports_list):
	for item in imports_list:
		print(item)
		result = pipmain(['install', '--user', "%s==%s"%(item['name'], item['version'])])

