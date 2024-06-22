from glob import glob
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import librosa
import numpy as np
from tqdm import tqdm

class Model:
    def __init__(self, generator, discriminator, data_dir, device, sample_rate=16000, audio_length=16000, batch_size=64):
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.data_dir = data_dir
        self.device = device
        self.sample_rate = sample_rate
        self.audio_length = audio_length
        self.batch_size = batch_size
        
        self.criterion = nn.BCELoss()
        self.optimizerD = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optimizerG = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        
        self.dataset = self.create_dataset()
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)

    def create_dataset(self):
        class AudioDataset(torch.utils.data.Dataset):
            def __init__(self, data_dir, sample_rate, audio_length):
                self.data_dir = data_dir
                self.sample_rate = sample_rate
                self.audio_length = audio_length
                print(glob(data_dir))
                self.file_list = glob(data_dir)
            
            def __len__(self):
                return len(self.file_list)
            
            def __getitem__(self, idx):
                file_path = self.file_list[idx]
                y, _ = librosa.load(file_path, sr=self.sample_rate, duration=self.audio_length/self.sample_rate)
                if len(y) < self.audio_length:
                    y = np.pad(y, (0, self.audio_length - len(y)), mode='constant')
                elif len(y) > self.audio_length:
                    y = y[:self.audio_length]
                return torch.from_numpy(y).float().unsqueeze(0)
        
        return AudioDataset(self.data_dir, self.sample_rate, self.audio_length)

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            for i, real_audio in tqdm(enumerate(self.dataloader), total=len(self.dataloader)):
                real_audio = real_audio.to(self.device)
                
                # Train Discriminator
                self.discriminator.zero_grad()
                batch_size = real_audio.size(0)
                label = torch.full((batch_size,), 1.0, device=self.device)
                output = self.discriminator(real_audio).view(batch_size, -1).mean(dim=1)
                errD_real = self.criterion(output, label)
                errD_real.backward()
                
                noise = torch.randn(batch_size, 100, 1, device=self.device)
                fake_audio = self.generator(noise)
                label.fill_(0)
                output = self.discriminator(fake_audio.detach()).view(-1)
                errD_fake = self.criterion(output, label)
                errD_fake.backward()
                errD = errD_real + errD_fake
                self.optimizerD.step()

                # Train Generator
                self.generator.zero_grad()
                label.fill_(1)
                output = self.discriminator(fake_audio).view(-1)
                errG = self.criterion(output, label)
                errG.backward()
                self.optimizerG.step()

                if i % 50 == 0:
                    print(f"[{epoch}/{num_epochs}][{i}/{len(self.dataloader)}] "
                          f"Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f}")
            
            # Save model checkpoints
            if (epoch + 1) % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.generator.state_dict(),
                    'optimizer_state_dict': self.optimizerG.state_dict(),
                    'loss': errG,
                }, f'./wavegan_gen_epoch_{epoch}.pth')

    def generate_audio(self, noise_dim=100, sample_rate=16000):
        self.generator.eval()
        noise = torch.randn(1, noise_dim, 1, device=self.device)
        generated_audio = self.generator(noise)
        audio = generated_audio.squeeze().detach().cpu().numpy()
        librosa.output.write_wav('generated.wav', audio, sample_rate)