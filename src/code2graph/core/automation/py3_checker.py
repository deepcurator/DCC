#! /usr/bin/env python3.7

import sys
print(sys.version)

import compileall
# from pathlib import Path
try:
	# print(compileall.compile_dir(Path('..')/"test"/"fashion_mnist", force=True))
	# print(compileall.compile_dir(Path('..')/"test"/"adaptive-f-divergence-master", force=True))
	print(sys.argv)
	result = compileall.compile_dir(sys.argv[1], force=True)
	print("Result:", result)
except Exception as e:
	print(e)