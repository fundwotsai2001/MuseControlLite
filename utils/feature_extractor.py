import torch.nn as nn
import torch.nn.functional as F

class dynamics_extractor_full_stereo(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1d_1 = nn.Conv1d(2, 16, kernel_size=3, padding=1, stride=2)  
        self.conv1d_2 = nn.Conv1d(16, 16, kernel_size=3, padding=1)  
        self.conv1d_3 = nn.Conv1d(16, 128, kernel_size=3, padding=1, stride=2)
        self.conv1d_4 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.conv1d_5 = nn.Conv1d(128, 256, kernel_size=3, padding=1, stride=2)
    def forward(self, x):
        # original shape: (batchsize, 1, 8280)
        # x = x.unsqueeze(1) # shape: (batchsize, 1, 8280)
        x = self.conv1d_1(x)  # shape: (batchsize, 16, 4140)
        x = F.silu(x)
        x = self.conv1d_2(x)  # shape: (batchsize, 16, 4140)
        x = F.silu(x)
        x = self.conv1d_3(x)  # shape: (batchsize, 128, 2070)
        x = F.silu(x)
        x = self.conv1d_4(x)  # shape: (batchsize, 128, 2070)
        x = F.silu(x)
        x = self.conv1d_5(x)  # shape: (batchsize, 192, 1035)
        return x
class melody_extractor_full_mono(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1d_1 = nn.Conv1d(128, 256, kernel_size=3, padding=0, stride=2)  
        self.conv1d_2 = nn.Conv1d(256, 256, kernel_size=3, padding=1)  
        self.conv1d_3 = nn.Conv1d(256, 512, kernel_size=3, padding=1, stride=2)  
        self.conv1d_4 = nn.Conv1d(512, 512, kernel_size=3, padding=1)
        self.conv1d_5 = nn.Conv1d(512, 768, kernel_size=3, padding=1)
    def forward(self, x):
        # original shape: (batchsize, 12, 1296)
        x = self.conv1d_1(x)# shape: (batchsize, 64, 2048)
        x = F.silu(x)
        x = self.conv1d_2(x) # shape: (batchsize, 64, 2048)
        x = F.silu(x)
        x = self.conv1d_3(x) # shape: (batchsize, 128, 1024)
        x = F.silu(x)
        x = self.conv1d_4(x) # shape: (batchsize, 128, 1024)
        x = F.silu(x)
        x = self.conv1d_5(x) # shape: (batchsize, 768, 1024)
        return x
class melody_extractor_mono(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1d_1 = nn.Conv1d(128, 128, kernel_size=3, padding=0, stride=2)  
        self.conv1d_2 = nn.Conv1d(128, 192, kernel_size=3, padding=1, stride=2)  
        self.conv1d_3 = nn.Conv1d(192, 192, kernel_size=3, padding=1)
    def forward(self, x):
        # original shape: (batchsize, 12, 1296)
        x = self.conv1d_1(x)# shape: (batchsize, 64, 2048)
        x = F.silu(x)
        x = self.conv1d_2(x) # shape: (batchsize, 64, 2048)
        x = F.silu(x)
        x = self.conv1d_3(x) # shape: (batchsize, 128, 1024)
        return x
    
class melody_extractor_full_stereo(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(num_embeddings=129, embedding_dim=48)

        # Four Conv1d layers, each with kernel_size=3, padding=1:
        self.conv1 = nn.Conv1d(384, 384, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(384, 768, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(768, 768, kernel_size=3, padding=1)

    def forward(self, melody_idxs):
        # melody_idxs: LongTensor of shape (B, 8, 4096)
        B, eight, L = melody_idxs.shape  # L == 4096

        # 1) Embed:
        #    (B, 8, 4096) → (B, 8, 4096, 48)
        embedded = self.embed(melody_idxs)

        # 2) Permute & reshape → (B, 8*48, 4096) = (B, 384, 4096)
        x = embedded.permute(0, 1, 3, 2)      # (B, 8, 48, 4096)
        x = x.reshape(B, eight * 48, L)       # (B, 384, 4096)

        # 3) Conv1 → (B, 384, 4096)
        x = F.silu(self.conv1(x))

        # 4) Conv2 → (B, 768, 4096)
        x = F.silu(self.conv2(x))

        # 5) Conv3 → (B, 768, 4096)
        x = F.silu(self.conv3(x))

        # Now x is (B, 1536, 4096) and can be sent on to whatever comes next
        return x
class melody_extractor_stereo(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(num_embeddings=129, embedding_dim=4)

        # Four Conv1d layers, each with kernel_size=3, padding=1:
        self.conv1 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=3, padding=0, stride=2)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(128, 128, kernel_size=3, padding=1, stride=2)
        self.conv5 = nn.Conv1d(128, 256, kernel_size=3, padding=1)

    def forward(self, melody_idxs):
        # melody_idxs: LongTensor of shape (B, 8, 4096)
        B, eight, L = melody_idxs.shape  # L == 4096

        # 1) Embed:
        #    (B, 8, 4096) → (B, 8, 4096, 4)
        embedded = self.embed(melody_idxs)

        # 2) Permute & reshape → (B, 8*4, 4096) = (B, 32, 4096)
        x = embedded.permute(0, 1, 3, 2)      # (B, 8, 4, 4096)
        x = x.reshape(B, eight * 4, L)       # (B, 32, 4096)

        # 3) Conv1 → (B, 384, 4096)
        x = F.silu(self.conv1(x))

        # 4) Conv2 → (B, 768, 4096)
        x = F.silu(self.conv2(x))

        # 5) Conv3 → (B, 768, 4096)
        x = F.silu(self.conv3(x))

        x = F.silu(self.conv4(x))

        x = F.silu(self.conv5(x))

        # Now x is (B, 1536, 4096) and can be sent on to whatever comes next
        return x

class dynamics_extractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1d_1 = nn.Conv1d(1, 16, kernel_size=3, padding=1, stride=2)  
        self.conv1d_2 = nn.Conv1d(16, 16, kernel_size=3, padding=1)  
        self.conv1d_3 = nn.Conv1d(16, 128, kernel_size=3, padding=1, stride=2)
        self.conv1d_4 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.conv1d_5 = nn.Conv1d(128, 192, kernel_size=3, padding=1, stride=2)
    def forward(self, x):
        # original shape: (batchsize, 1, 8280)
        # x = x.unsqueeze(1) # shape: (batchsize, 1, 8280)
        x = self.conv1d_1(x)  # shape: (batchsize, 16, 4140)
        x = F.silu(x)
        x = self.conv1d_2(x)  # shape: (batchsize, 16, 4140)
        x = F.silu(x)
        x = self.conv1d_3(x)  # shape: (batchsize, 128, 2070)
        x = F.silu(x)
        x = self.conv1d_4(x)  # shape: (batchsize, 128, 2070)
        x = F.silu(x)
        x = self.conv1d_5(x)  # shape: (batchsize, 192, 1035)
        return x
class rhythm_extractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1d_1 = nn.Conv1d(2, 16, kernel_size=3, padding=1)  
        self.conv1d_2 = nn.Conv1d(16, 64, kernel_size=3, padding=1)  
        self.conv1d_3 = nn.Conv1d(64, 128, kernel_size=3, padding=1, stride=2)  
        self.conv1d_4 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.conv1d_5 = nn.Conv1d(128, 192, kernel_size=3, padding=1, stride=2)
    def forward(self, x):
        # original shape: (batchsize, 2, 3000)
        x = self.conv1d_1(x)# shape: (batchsize, 64, 3000)
        x = F.silu(x)
        x = self.conv1d_2(x) # shape: (batchsize, 64, 3000)
        x = F.silu(x)
        x = self.conv1d_3(x) # shape: (batchsize, 128, 1500)
        x = F.silu(x)
        x = self.conv1d_4(x) # shape: (batchsize, 128, 1500)
        x = F.silu(x)
        x = self.conv1d_5(x) # shape:
        return x