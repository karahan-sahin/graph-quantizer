import torch
from torch import nn, optim
from .autoencoder.gnn import GraphAutoencoder

def train(model, dataloader, optimizer, device):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(dataloader):
        inputs = data[0].to(device)
        optimizer.zero_grad()

        outputs, loss = model(inputs)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)
    return avg_loss

def evaluate(model, dataloader, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            inputs = data[0].to(device)

            outputs, loss = model(inputs)

            running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)
    return avg_loss

def main():

    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--input_dim', type=int, default=784)
    parser.add_argument('--hidden_dim', type=int, default=100)
    parser.add_argument('--num_embeddings', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    
    args = parser.parse_args()


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GraphAutoencoder().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    train_dataloader = ...  # Load your training data here
    val_dataloader = ...  # Load your validation data here

    for epoch in range(args.num_epochs):
        train_loss = train(model, train_dataloader, optimizer, device)
        val_loss = evaluate(model, val_dataloader, device)
        print(f'Epoch {epoch+1}, Train Loss: {train_loss}, Val Loss: {val_loss}')

if __name__ == '__main__':
    main()