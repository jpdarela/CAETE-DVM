import sys
import os
root_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(root_path)
os.chdir("../")
src = os.getcwd()
try:
    os.system(f'{sys.executable} -c "import caete_module"')
except Exception as e:
    pass # error silently, as this is just a warm-up script
finally:
    os.chdir(src)