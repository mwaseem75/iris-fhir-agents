import os
current_dir = os.path.dirname(os.path.abspath(__file__))

file_name_ep = os.path.join(os.path.dirname(current_dir),"iris_ep.py")
if os.path.exists(file_name_ep):
    from iris_ep import *
    from iris_ep import __getattr__

file_name_elsdk = os.path.join(current_dir, "_elsdk_.py")
if os.path.exists(file_name_elsdk):
    from iris._elsdk_ import *
