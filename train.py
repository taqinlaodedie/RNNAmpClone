import argparse
import pickle
import torch
from network import RnnNet, FxDataset, RMSELoss

def main(args):
    model = RnnNet(
        input_size=args.input_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        bias=args.bias
    )

    if torch.cuda.is_available():
        print("Use GPU to train")
        device = "cuda:0"
    else:
        print("Use CPU to train")
        device = "cpu"
    model.to(device)

    ds = lambda x, y: FxDataset(
        torch.from_numpy(x).to(device), 
        torch.from_numpy(y).to(device), 
        window_len=args.sequence_length, 
        batch_size=args.batch_size,
        shuffle=True
    )
    data = pickle.load(open(args.data, "rb"))
    train_dataset = ds(data["x_train"], data["y_train"])
    valid_dataset = ds(data["x_valid"], data["y_valid"])

    loss_function = RMSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    best_train_loss = 1e6

    for i in range(args.max_epochs):
        print("*********** Epoch {} **************".format(i+1))
        train_loss = model.train_epoch(train_dataset, loss_function, optimizer, device)
        valid_loss = model.valid_epoch(valid_dataset, loss_function, device)
        print("Train loss {}, Valid loss {}".format(train_loss, valid_loss))
        if train_loss < best_train_loss:
            torch.save(model, "model.pth")
            best_train_loss = train_loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_size", type=int, default=1)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--bias", type=bool, default=True)
    parser.add_argument("--sequence_length", type=int, default=22050)

    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=1e-3)

    parser.add_argument("--max_epochs", type=int, default=200)

    parser.add_argument("--data", default="data.pickle")
    args = parser.parse_args()
    main(args)