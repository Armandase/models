import torch.nn as nn
import torch

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.down64_1 = nn.Conv2d(1, 64, 3, padding=0)
        self.down64_2 = nn.Conv2d(64, 64, 3, padding=0)

        self.down128_1 = nn.Conv2d(64, 128, 3, padding=0)
        self.down128_2 = nn.Conv2d(128, 128, 3, padding=0)
        
        self.down256_1 = nn.Conv2d(128, 256, 3, padding=0)
        self.down256_2 = nn.Conv2d(256, 256, 3, padding=0)

        self.down512_1 = nn.Conv2d(256, 512, 3, padding=0)
        self.down512_2 = nn.Conv2d(512, 512, 3, padding=0)
        
        self.down1024_1 = nn.Conv2d(512, 1024, 3, padding=0)
        self.down1024_2 = nn.Conv2d(1024, 1024, 3, padding=0)

        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2)


    def forward(self, x):
        print(f"Input shape: {x.shape}")
        x = self.down64_1(x)
        x = self.relu(x)
        x = self.down64_2(x)
        x = self.relu(x)
        x1 = x
        x = self.maxpool(x)
        x = self.down128_1(x)
        x = self.relu(x)
        x = self.down128_2(x)
        x = self.relu(x)
        x2 = x
        x = self.maxpool(x)
        x = self.down256_1(x)
        x = self.relu(x)
        x = self.down256_2(x)
        x = self.relu(x)
        x3 = x
        x = self.maxpool(x)
        x = self.down512_1(x)
        x = self.relu(x)
        x = self.down512_2(x)
        x = self.relu(x)
        x4 = x
        x = self.maxpool(x)
        x = self.down1024_1(x)
        x = self.relu(x)
        x = self.down1024_2(x)
        x = self.relu(x)

        print("Shapes at each level:")
        print(f"Level 1: {x1.shape}")
        print(f"Level 2: {x2.shape}")
        print(f"Level 3: {x3.shape}")
        print(f"Level 4: {x4.shape}")
        print(f"Bottom Level: {x.shape}")
        return x
    
def test():
    model = UNet()
    x = torch.randn((1, 1, 572, 572))
    output = model(x)

if __name__ == "__main__":
    test()