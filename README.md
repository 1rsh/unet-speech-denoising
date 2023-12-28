# U-Net Implementation for Hindi Speech Denoising ![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)

![unet](https://github.com/1rsh/unet-speech-denoising/assets/93649948/25f99e4f-3af8-4ff1-80e2-131bc795dddb)


This repository contains an implementation of U-Net (originally intended for Image Segmentation) which was introduced in the paper [U-Net: Convolutional Networks for Biomedical
Image Segmentation](https://arxiv.org/pdf/1505.04597v1.pdf) using PyTorch.  
<br>
The model is trained on [Rajasthani Hindi Speech Data](https://www.microsoft.com/en-gb/download/details.aspx?id=105385) as the speech data and [ESC-50 Dataset](https://github.com/karolpiczak/ESC-50) as the noise data.
<br><br>
****
## Repository Walkthrough

### 1. Creating Noisy Data
```python
def CreateData(speech_dir, noise_dir, sample_rate, frame_length, min_duration, nb_samples, hop_length, path_save_sound):
    
    noise = audio_to_npy(noise_dir, sample_rate, frame_length, min_duration)
    voice = audio_to_npy(speech_dir, sample_rate, frame_length, min_duration)

    
    prod_voice, prod_noise, prod_noisy_voice = blend_noise_randomly(voice, noise, nb_samples, frame_length)
    print()
    for i in tqdm(range(len(prod_voice)), desc = f"Saving to {path_save_sound}"):

        noisy_voice_long = prod_noisy_voice[i]
        sf.write(path_save_sound + 'noisy_voice/'+str(i)+'.wav', noisy_voice_long[:], sample_rate)
        spectrogram_image(hop_length, noisy_voice_long, "noisy_voice/"+str(i)+".png")
        voice_long = prod_voice[i]
        sf.write(path_save_sound + 'voice/'+str(i)+'.wav', voice_long[:], sample_rate)
        spectrogram_image(hop_length, voice_long, "voice/"+str(i)+".png")        
        noise_long = prod_noise[i]
        sf.write(path_save_sound + 'noise/'+str(i)+'.wav', noise_long[:], sample_rate)
        spectrogram_image(hop_length, noise_long, "noise/"+str(i)+".png")
```
To check the code for the helper functions check [prepare_data.py](prepare_data.py)
### 2. Spectrogram Dataset Class
```python
class SpectrogramDataset(Dataset):
    def __init__(self, root_path):
        self.images = sorted([root_path+"/noisy_voice/"+x for x in os.listdir(root_path+"/noisy_voice/")])
        self.targets = sorted([root_path+"/noise/"+x for x in os.listdir(root_path+"/noise/")])

        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])
    
    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert("L")
        target = Image.open(self.targets[index]).convert("L")

        return self.transform(img), self.transform(target)
    
    def __len__(self):
        return len(self.images)
```
### 3. U-Net Model Class
```python
class UNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.downconv1 = DownSampleLayer(in_channels, 64)
        self.downconv2 = DownSampleLayer(64, 128)
        self.downconv3 = DownSampleLayer(128, 256)
        self.downconv4 = DownSampleLayer(256, 512)

        self.bottleneck = DoubleConvLayer(512, 1024)

        self.upconv1 = UpSampleLayer(1024, 512)
        self.upconv2 = UpSampleLayer(512, 256)
        self.upconv3 = UpSampleLayer(256, 128)
        self.upconv4 = UpSampleLayer(128, 64)

        self.out = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size = 1)

    def forward(self, x):
        down1, p1 = self.downconv1(x)
        down2, p2 = self.downconv2(p1)
        down3, p3 = self.downconv3(p2)
        down4, p4 = self.downconv4(p3)

        bottle = self.bottleneck(p4)

        up1 = self.upconv1(bottle, down4)
        up2 = self.upconv2(up1, down3)
        up3 = self.upconv3(up2, down2)
        up4 = self.upconv4(up3, down1)

        out = self.out(up4)

        return out
```
To check the code for the helper functions check [unet_utils.py](unet_utils.py)
### 4. Training
```python
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
```

## Footnote
If you wish to use the following PyTorch implementation of U-Net for your own project, just download the python scripts and the datasets from the provided links according to your file hierarchy. <br>
Feel free to contact me at <a href = "mailto:irsh.iitkgp@gmail.com">irsh.iitkgp@gmail.com</a>.

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
