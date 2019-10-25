import sys
import os
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from net.dstg import *
if __name__ == '__main__':
    graph_args = {
        'layout':'openpose',
        'strategy':'spatial'
    }
    model = DSTG(graph_args,depth=16)
    for name,para in model.named_parameters():
        print(name)
    # print(model)