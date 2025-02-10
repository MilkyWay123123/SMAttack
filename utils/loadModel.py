import os.path
import sys

sys.path.append("..")

from models.stgcn.st_gcn import STGCN_Model
from models.msg3d.msg3d import Model as MCG3D_Model
from models.agcn.agcn import Model as AGCN_Model

import torch
import torch.nn as nn


def weights_init(model):
    with torch.no_grad():
        for child in list(model.children()):
            print("init ", child)
            for param in list(child.parameters()):
                if param.dim() == 2:
                    nn.init.xavier_uniform_(param)
    print('weights initialization finished!')


def loadSTGCN():
    load_path = './ModelWeights/STGCN/ntu60/ntu-xsub.pt'
    print(f'load STGCN:{load_path}')
    graph_args = {
        "layout": "ntu-rgb+d",
        "strategy": "spatial"
    }
    stgcn = STGCN_Model(3, 60, graph_args, True)
    stgcn.eval()
    pretrained_weights = torch.load(load_path)
    stgcn.load_state_dict(pretrained_weights)
    stgcn.cuda()

    return stgcn



def loadMSG3D():
    load_path = './ModelWeights/MSG3D/ntu60-xsub-joint-better.pt'
    print(f'load MSG3D:{load_path}')
    mcg3d = MCG3D_Model(
        num_class=60,
        num_point=25,
        num_person=2,
        num_gcn_scales=13,
        num_g3d_scales=6,
        graph='graph.ntu_rgb_d.AdjMatrixGraph'
    )
    mcg3d.eval()
    pretrained_weights = torch.load(load_path)
    mcg3d.load_state_dict(pretrained_weights)
    mcg3d.cuda()

    return mcg3d

def loadAGCN():
    load_path = './ModelWeights/AGCN/ntu60/ntu_cs_agcn_joint-49-31500.pt'
    print(f'load agcn:{load_path}')
    agcn = AGCN_Model(
        num_class=60,
        num_point=25,
        num_person=2,
        graph='graph.ntu_rgb_d.Graph', graph_args={'labeling_mode': 'spatial'})
    agcn.eval()
    pretrained_weights = torch.load(load_path)
    agcn.load_state_dict(pretrained_weights)
    agcn.cuda()

    return agcn


def getModel(AttackedModel):
    if AttackedModel == 'msg3d':
        model = loadMSG3D()
    elif AttackedModel == 'agcn':
        model = loadAGCN()
    else:
        model = loadSTGCN()
    model.eval()
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"Total number of model parameters：{total_params:.2f} M")
    return model
