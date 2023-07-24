from torch.optim import SGD, Adam, AdamW, NAdam
import torch

def get_optim(model, **kwargs):

    name = kwargs["name"]
    del kwargs["name"]

    assert name in ["SGD", "Adam", "AdamW", "Nadam"]

    if name == "SGD":
        optimizer = SGD(model.parameters(), **kwargs)
    elif name == "Adam":
        optimizer = Adam(model.parameters(), **kwargs)
    elif name == "AdamW":
        optimizer = AdamW(model.parameters(), **kwargs)
    else:
        optimizer = NAdam(model.parameters(), **kwargs)

    return optimizer


