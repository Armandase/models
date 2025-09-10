"""
Implementation based on the follwing paper:
Jordan, M. I., & Jacobs, R. A. (1993). Hierarchical mixtures of experts and the EM algorithm. Proceedings of 1993 International Conference on Neural Networks (IJCNN-93-Nagoya, Japan), 2, 1339–1344 vol.2. doi:10.1109/IJCNN.1993.716791
"""

import torch
import torch.nn as nn
import torchvision
from sklearn.metrics import f1_score
from model_hmoe import ClassicMediumCNN_HME

EPOCHS = 50
BATCH_SIZE = 256
LR = 0.001
NUM_CLASSES = 10
IMG_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# BASENAME = "hmoe_cnn_tiny_expert"
NB_EXPERTS = 4
BASENAME = "classic_cnn"

class ClassicMediumCNN(nn.Module):
    def __init__(self, input_channels=3, num_classes=NUM_CLASSES, img_size=IMG_SIZE):
        super(ClassicMediumCNN, self).__init__()
        self.img_size = img_size
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        final_size = self.img_size // 8
        self.classifier = nn.Sequential(
            nn.Linear(256 * final_size * final_size, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
class HierarchicalMoeMediumCNN(nn.Module):
    def __init__(self, input_channels=3, num_classes=NUM_CLASSES, img_size=IMG_SIZE, num_experts=NB_EXPERTS):
        super(HierarchicalMoeMediumCNN, self).__init__()
        self.img_size = img_size
        self.gating_network = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(128 * (img_size // 8) * (img_size // 8), 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 4),  # Assuming 4 experts
            nn.Softmax(dim=1)
        )
        # self.experts = nn.ModuleList([
        #     nn.Sequential(
        #         nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
        #         nn.ReLU(inplace=True),
        #         nn.MaxPool2d(kernel_size=2, stride=2),
        #         nn.Conv2d(64, 128, kernel_size=3, padding=1),
        #         nn.ReLU(inplace=True),
        #         nn.MaxPool2d(kernel_size=2, stride=2),
        #         nn.Conv2d(128, 256, kernel_size=3, padding=1),
        #         nn.ReLU(inplace=True),
        #         nn.MaxPool2d(kernel_size=2, stride=2),
        #         nn.Flatten(),
        #         nn.Linear(256 * (img_size // 8) * (img_size // 8), 512),
        #         nn.ReLU(inplace=True),
        #         nn.Linear(512, num_classes)
        #     ) for _ in range(4)
        # ])
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Flatten(),
                nn.Linear(256 * (img_size // 8) * (img_size // 8), 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, num_classes)
            ) for _ in range(4)
        ])

        final_size = self.img_size // 8
        self.classifier = nn.Sequential(
            nn.Linear(256 * final_size * final_size, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        gating_weights = self.gating_network(x)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        output = torch.sum(gating_weights.unsqueeze(2) * expert_outputs, dim=1)
        return output

def main():
    # model = ClassicMediumCNN_HME(input_channels=3, num_classes=NUM_CLASSES, img_size=IMG_SIZE)
    model = ClassicMediumCNN(input_channels=3, num_classes=NUM_CLASSES, img_size=IMG_SIZE)
    model.to(DEVICE)
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=torchvision.transforms.ToTensor())
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=torchvision.transforms.ToTensor())
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=10)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_test_f1 = 0.0
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        running_f1 = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            running_f1 += f1_score(labels.cpu().numpy(), predicted.cpu().numpy(), average='macro')
        torch.save(model.state_dict(), f'latest_{BASENAME}_model.pth')
        avg_f1 = running_f1 / len(trainloader)
        print(f'[Epoch {epoch + 1}] loss: {running_loss / len(trainloader):.3f}, F1 Score: {avg_f1:.3f}')

        model.eval()
        with torch.no_grad():
            test_f1 = 0.0
            for data in testloader:
                images, labels = data
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                test_f1 += f1_score(labels.cpu().numpy(), predicted.cpu().numpy(), average='macro')
            avg_test_f1 = test_f1 / len(testloader)
            print(f'Test F1 Score: {avg_test_f1:.3f}')

            if avg_test_f1 > best_test_f1:
                best_test_f1 = avg_test_f1
                torch.save(model.state_dict(), f'best_{BASENAME}_model.pth')

        with open(f'{BASENAME}_training_log.txt', 'a') as f:
            f.write(f'Epoch {epoch + 1}, Loss: {running_loss / len(trainloader):.3f}, Train F1: {avg_f1:.3f}, Test F1: {avg_test_f1:.3f}\n')
    print('Finished Training')

if __name__ == "__main__":
    main()
