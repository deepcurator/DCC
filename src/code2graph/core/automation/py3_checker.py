#! /usr/bin/env python3.7
import sys, compileall
result = compileall.compile_dir(sys.argv[1], force=True)
print("Result:", result)