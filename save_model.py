import argparse
import numpy as np
import json
import torch

from network import RnnNet

def save_model(args):
    model = torch.load(args.model)

    data_out = {"type": "GRU", 
                "output_channels": 1,
                "input_channels": 1,
                "hidden_size": args.hidden_size,
                "bias": args.bias,
                "variables": []}
    
    for parameter_tensor in model.state_dict():
        print(parameter_tensor, '\t', model.state_dict()[parameter_tensor].size())
        data_out["variables"].append({"name": parameter_tensor,
            "data": model.state_dict()[parameter_tensor].cpu().flatten().numpy().tolist()
            })
    # output final dictionary to json file
    with open('models/' + args.name + '.json', 'w') as outfile:
        json.dump(data_out, outfile)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="model.pth")
    parser.add_argument("--hidden_size", type=int, default=24)
    parser.add_argument("--bias", type=bool, default=True)
    parser.add_argument("--name", default="converted_model")
    args = parser.parse_args()
    save_model(args)