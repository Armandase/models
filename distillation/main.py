from datasets import load_dataset
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from sklearn.metrics import f1_score
from constants import EPOCHS, BATCH_SIZE, LEARNING_RATE, MODEL_SAVE_PATH, LOG_PATH
from models import get_tinyHomemadeCNN, get_dinov2, get_resnet18

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_data():
    dataset = load_dataset("beans", split="train[:5000]")
    dataset = dataset.train_test_split(test_size=0.2)
    return dataset

def compute_mean_std(data):
    images = data['image']
    
    np_images = np.stack([np.array(img) for img in images])
    np_images = np_images / 255.0
    print(np_images.shape)  # (827, 500, 500, 3)
    mean = np.mean(np_images, axis=(0, 1, 2))
    std = np.std(np_images, axis=(0, 1, 2))
    return mean, std

def get_dataloader(data, batch_size, train=False, mean=[0.4840093, 0.52049197, 0.31205457], std=[0.21131694, 0.22427338, 0.20144724]):
    dataset = data.select_columns(['image', 'labels'])

    train_transform = T.Compose([
        T.Resize(256),
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std)
    ])

    test_transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std)
    ])

    transform = train_transform if train else test_transform

    class HFDataset(torch.utils.data.Dataset):
        def __init__(self, hf_dataset, transform):
            self.ds = hf_dataset
            self.transform = transform

        def __len__(self):
            return len(self.ds)

        def __getitem__(self, idx):
            item = self.ds[idx]
            img = item['image']
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)
            img = self.transform(img)
            label = torch.tensor(item['labels'], dtype=torch.long)
            return {'image': img, 'labels': label}

    wrapped = HFDataset(dataset, transform)
    data_loader = torch.utils.data.DataLoader(
        wrapped,
        batch_size=batch_size,
        shuffle=train,
        num_workers=4,
        pin_memory=(device.type == 'cuda')
    )
    return data_loader

def train(model, data):
    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss()

    train_loader = get_dataloader(data, BATCH_SIZE, train=True)

    for epoch in range(EPOCHS):
        all_preds = []
        all_labels = []
        total_loss = 0.0
        for i, batch in enumerate(train_loader):
            images = batch['image']
            labels = batch['labels']
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            # outputs = model(images).logits
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

            preds = outputs.argmax(dim=1).cpu().numpy()
            labels_np = labels.cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels_np)

        # concatenate and compute metrics on the whole epoch
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        epoch_f1 = f1_score(all_labels, all_preds, average='weighted')
        avg_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0.0
        print(f"Epoch [{epoch+1}/{EPOCHS}] completed. Average F1 Score: {epoch_f1:.4f}")
        print(f"Epoch [{epoch+1}/{EPOCHS}] completed. Average Loss: {avg_loss:.4f}")
        with open(LOG_PATH, "a") as log_file:
            if epoch == 0:
                log_file.write("epoch,loss,f1_score\n")
            log_file.write(f"{epoch+1},{avg_loss:.4f},{epoch_f1:.4f}\n")
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
    return model

def test(model, data):
    model.to(device)
    model.eval()
    
    test_loader = get_dataloader(data, BATCH_SIZE, train=False)
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            images = batch['image']
            labels = batch['labels']
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images).logits
            preds = outputs.argmax(dim=1).cpu().numpy()
            labels_np = labels.cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels_np)

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    accuracy = (all_preds == all_labels).mean() * 100
    test_f1 = f1_score(all_labels, all_preds, average='weighted')
    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"Test F1 Score: {test_f1:.4f}")

def main():
    beans = get_data()
    num_labels = beans['train'].features['labels'].num_classes

    # mean, std = compute_mean_std(beans['train'])
    # print(f"Computed mean: {mean}, std: {std}")

    model = get_resnet18(num_labels=num_labels, checkpoint_path="models/resnet18_model.pth")
    # model = get_dinov2(num_labels=num_labels)
    # model = get_tinyHomemadeCNN(num_labels=num_labels)

    train_data = beans['train']
    test_data = beans['test']

    # train(model, train_data)
    test(model, test_data)

if __name__ == '__main__':
    main()