import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from utils import calc_mean_RGB
from dataset import load_dataset
from model import VGG16

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:  # print every 100 mini-batches
            print(f'[{i + 1}] loss: {running_loss / 100:.3f}')
            running_loss = 0.0

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Calculate mean RGB
    mean_r, mean_g, mean_b = calc_mean_RGB(args.data_path)
    print(f"Mean R: {mean_r}, Mean G: {mean_g}, Mean B: {mean_b}")

    # Load dataset
    dataset, dataloader = load_dataset(args.data_path, mean_r, mean_g, mean_b, args.batch_size)

    # Create model
    model = VGG16(num_classes=len(dataset.classes)).to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Training loop
    num_epochs = args.epochs
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        train(model, dataloader, criterion, optimizer, device)

    print('Finished Training')

    # Save the model
    torch.save(model.state_dict(), 'vgg16_model.pth')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VGG16 Training")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train")
    args = parser.parse_args()

    main(args)