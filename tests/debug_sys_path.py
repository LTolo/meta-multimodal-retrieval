import sys
import os

print("CWD:", os.getcwd())
print("---- sys.path ----")
for p in sys.path:
    print(p)

print("\nPIPELINE EXISTS:", os.path.isdir("pipeline"))
