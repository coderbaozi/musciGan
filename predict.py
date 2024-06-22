import torch
from generator import Generator  
import soundfile as sf

# 检查设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 定义生成器模型
netG = Generator().to(device)
basePath = '/mlx_devbox/users/qianziyang/playground/musciGan'
# 加载模型权重
checkpoint = torch.load(basePath+'/wavegan_gen_epoch_99.pth')
state_dict = checkpoint['model_state_dict']
new_state_dict = {k: v for k, v in state_dict.items() if k in netG.state_dict()}
netG.load_state_dict(new_state_dict)

netG.eval()  # 设置为评估模式

batch_size = 16
noise_dim = 100

# 生成2D的噪声张量
noise = torch.randn(batch_size, noise_dim).to(device)
noise = noise.view(batch_size, noise_dim, 1)

# 生成音频
with torch.no_grad():
    generated_audio = netG(noise)

# 将张量移动到CPU并转换为NumPy数组
audio_data = generated_audio.squeeze().cpu().numpy()
# 采样率，假设为16kHz
sample_rate = 16000
# 保存音频文件
sf.write('generated_audio.wav', audio_data, sample_rate)