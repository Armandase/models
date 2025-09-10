import torch
import torch.nn as nn
import torchvision
from sklearn.metrics import f1_score
from model_m3att import smallDIYTransformerForImage

EPOCHS = 50
BATCH_SIZE = 128
LR = 1e-4
NUM_CLASSES = 10
IMG_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# BASENAME = "hmoe_cnn_tiny_expert"
NB_PATCH = 4
BASENAME = "classic_transformer_default"

def main():
    # model = smallDIYTransformerForImage(input_channels=3, num_classes=NUM_CLASSES, img_size=IMG_SIZE)
    model = smallDIYTransformerForImage(img_size=IMG_SIZE,
        patch_size=NB_PATCH,
        in_channels=3,
        num_classes=NUM_CLASSES,
        d_model=128,
        n_heads=8,
        n_layers=4,
        d_ff=512,
        dropout=0.1)
    model.to(DEVICE)
    
    # Print model nb params
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # CIFAR-10 normalization
    transform_train = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    transform_test = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=10)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)  # AdamW with weight decay
    
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

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
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            running_f1 += f1_score(labels.cpu().numpy(), predicted.cpu().numpy(), average='macro')
        torch.save(model.state_dict(), f'latest_{BASENAME}_model.pth')
        avg_f1 = running_f1 / len(trainloader)
        # current_lr = scheduler.get_last_lr()[0]
        # print(f'[Epoch {epoch + 1}] loss: {running_loss / len(trainloader):.3f}, F1 Score: {avg_f1:.3f}, LR: {current_lr:.6f}')
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
        
        # Step the scheduler
        # scheduler.step()

        with open(f'{BASENAME}_training_log.txt', 'a') as f:
            f.write(f'Epoch {epoch + 1}, Loss: {running_loss / len(trainloader):.3f}, Train F1: {avg_f1:.3f}, Test F1: {avg_test_f1:.3f}\n')
    print('Finished Training')

if __name__ == "__main__":
    main()