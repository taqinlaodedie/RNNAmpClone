import torch
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

from network import FxDataset

def wav_write(filename, signal):
    wavfile.write(filename, 44100, signal.astype(np.float32))

def main(args):
    model = torch.load(args.path)
    data = pickle.load(open(args.data, "rb"))
    
    if torch.cuda.is_available():
        print("Use GPU to predict")
        device = "cuda:0"
    else:
        print("Use CPU to predict")
        device = "cpu"
    model.to(device)
    dataset = FxDataset(
        torch.from_numpy(data["x_valid"]).to(device), 
        torch.from_numpy(data["y_valid"]).to(device), 
        args.sequence_length, 
        args.batch_size
    )

    with torch.no_grad():
        frame_size = args.batch_size * args.sequence_length
        pred = np.zeros(len(dataset) * frame_size)
        target = np.zeros(len(dataset) * frame_size)
        model.reset_hiddens(args.batch_size, device)
        for i in range(len(dataset)):
            x, y = dataset[i]
            y_pred = model(x)
            pred[i*frame_size : (i+1)*frame_size] = torch.flatten(y_pred).detach().to("cpu").numpy()
            target[i*frame_size : (i+1)*frame_size] = torch.flatten(y).detach().to("cpu").numpy()

    wav_write("pred.wav", pred)
    wav_write("target.wav", target)
    
    plt.figure(1)
    plt.plot(target[:441000], label='target')
    plt.plot(pred[:441000], label='pred')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data.pickle")
    parser.add_argument("--path", default="model.pth")
    parser.add_argument("--sequence_length", type=int, default=4410)
    parser.add_argument("--batch_size", type=int, default=10)
    args = parser.parse_args()
    main(args)