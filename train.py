import torch
from torch import optim, nn
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm

from unet_main import UNet
from data import SpectrogramDataset

if __name__ == "__main__":
    learning_rate = 3e-4
    batch_size = 4
    epochs = 10
    data_path = "data/processed/spectrogram"
    
    model_save_path = "models/unet.pth"

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    train_dataset = SpectrogramDataset(data_path)

    random_gen = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(train_dataset, [0.9, 0.1], generator = random_gen)

    train_dataloader = DataLoader(dataset = train_dataset,
                                  batch_size = batch_size,
                                  shuffle = True)
    val_dataloader = DataLoader(dataset = val_dataset,
                                  batch_size = batch_size,
                                  shuffle = True)
    
    model = UNet(in_channels = 1, num_classes = 1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr = learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    show_epoch = 1
    history = {"train_loss": [], "val_loss":[]}
    for epoch in tqdm(range(epochs), desc = f"Total Epochs: {epochs}"):
        model.train()
        train_running_loss = 0
        for idx, img_and_target in enumerate(tqdm(train_dataloader, desc = f"Epoch {show_epoch} of {epochs}")):
            img = img_and_target[0].float().to(device)
            target = img_and_target[1].float().to(device)

            pred = model(img)

            loss = criterion(pred, target)
            train_running_loss += loss.item()

            loss.backward()
            optimizer.step()

        train_loss = train_running_loss / (idx + 1)
        history["train_loss"].append(train_loss)
        model.eval()
        val_running_loss = 0
        with torch.no_grad():
            for idx, img_and_target in enumerate(tqdm(val_dataloader)):
                img = img_and_target[0].float().to(device)
                target = img_and_target[1].float().to(device)

                pred = model(img)

                loss = criterion(pred, target)
                val_running_loss += loss.item()

            val_loss = train_running_loss / (idx + 1)
            history["val_loss"].append(val_loss)

        
        print()
        print(f"\nEpoch {show_epoch} Summary:")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print()
        show_epoch += 1

    torch.save(model.state_dict(), model_save_path)
    print(history)


