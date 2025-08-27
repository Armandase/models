from transformers import ResNetConfig, ResNetForImageClassification, Dinov2WithRegistersForImageClassification
from datasets import load_dataset
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import f1_score
from constants import EPOCHS, BATCH_SIZE, LEARNING_RATE, MODEL_SAVE_PATH

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_dinov2(num_labels=None):
    # allow specifying the number of output labels so the pretrained head is resized correctly
    if num_labels is not None:
        model = Dinov2WithRegistersForImageClassification.from_pretrained(
            "facebook/dinov2-with-registers-base",
            num_labels=num_labels,
        )
    else:
        model = Dinov2WithRegistersForImageClassification.from_pretrained("facebook/dinov2-with-registers-base")
    return model
    
def get_resnet18(num_labels=None):
    configuration = ResNetConfig(
        depths=[2, 2, 2, 2],
        downsample_in_bottleneck=False,
        downsample_in_first_stage=False,
        embedding_size=64,
        hidden_act="relu",
        hidden_sizes=[64, 128, 256, 512],
        layer_type="basic",
        model_type="resnet",
        num_channels=3,
        out_features=["stage4"],
        out_indices=[4],
        stage_names=["stem", "stage1", "stage2", "stage3", "stage4"]
    )
    # set number of labels if provided so classification head matches the dataset
    if num_labels is not None:
        configuration.num_labels = num_labels
    model = ResNetForImageClassification(configuration)
    return model

def get_data():
    dataset = load_dataset("beans", split="train[:5000]")
    dataset = dataset.train_test_split(test_size=0.2)
    return dataset

def get_dataloaders(data, batch_size):
    trainset = data['train'].select_columns(['image', 'labels'])
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testset = data['test'].select_columns(['image', 'labels'])
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def get_dataloader(data, batch_size):
    dataset = data.select_columns(['image', 'labels'])
    dataset = dataset.map(lambda x: {'image': x['image'], 'labels': x['labels']})
    dataset.set_format(type='torch', columns=['image', 'labels'])
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader

def train(model, data):
    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    train_loader = get_dataloader(data, BATCH_SIZE)

    f1_scores = []
    for epoch in range(EPOCHS):
        total_loss = 0
        for i, batch in enumerate(train_loader):
            images = batch['image']
            labels = batch['labels']
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images).logits
            loss = criterion(outputs, labels)
            total_loss = total_loss + loss.item()
            loss.backward()
            optimizer.step()
            
            f1_scores.append(f1_score(labels.cpu(), outputs.argmax(dim=1).cpu(), average='weighted'))
        print(f"Epoch [{epoch+1}/{EPOCHS}] completed. Average F1 Score: {sum(f1_scores)/len(f1_scores):.4f}")
        print(f"Epoch [{epoch+1}/{EPOCHS}] completed. Average Loss: {total_loss/len(data):.4f}")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    return model
    
def test(model, data):
    model.to(device)
    model.eval()
    
    test_loader = get_dataloader(data, BATCH_SIZE)
    correct = 0
    total = 0
    f1 = 0
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            images = batch['image']
            labels = batch['labels']
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images).logits
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            f1 += f1_score(labels.cpu(), predicted.cpu(), average='weighted')
    print(f"Test Accuracy: {100 * correct / total:.2f}%")
    print(f"Test F1 Score: {f1 / (len(data) / BATCH_SIZE):.4f}")

def main():
    # student = get_resnet18() 
    # teacher = get_dinov2()
    beans = get_data()

    num_labels = beans['train'].features['labels'].num_classes
    dinov2 = get_dinov2(num_labels=num_labels)

    train_data = beans['train']
    test_data = beans['test']
    dinov2 = train(dinov2, train_data)
    test(dinov2, test_data)

if __name__ == '__main__':
    main()