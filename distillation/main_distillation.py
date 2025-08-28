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
    print(np_images.shape) # (827, 500, 500, 3)
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

def softmax_with_temperature(x, θ):
    # e ^ (x / θ) / sum(e ^ (x / θ))
    # x = x - np.max(x, axis=1, keepdims=True)
    x_exp = np.exp(x / θ)
    return x_exp / np.sum(x_exp / θ, axis=-1, keepdims=True)

def get_loss_distillation():
    '''
    min l =N∑θlc (X i , yi ; θ ) + ld (X i , zi ; θ ).

    Avec:
        => lc (X i , yi ; θ ) = H (σ ( f (X i ; θ )), yi )
        Avec    H: negative cross entropy
                σ: softmax function
                f (X i ; θ ): logits en sortie du modele student
        => ld (X i , zi ; θ ) = DKL(σT(f(Xi;θ);T),σT(zi;T))
        Avec    DKL: KL-divergence
            σT(zi;T): soft logits en sortie du modele teacher
            σT(f(Xi;θ);T): soft logits en sortie du modele student  
    '''
    lc = torch.nn.CrossEntropyLoss()
    ld = torch.nn.KLDivLoss(reduction="batchmean")
    
    return lc, ld
    

def train(student, teacher, data):
    student.to(device)
    teacher.to(device)
    teacher.eval()
    student.train()

    optimizer = torch.optim.Adam(student.parameters(), lr=LEARNING_RATE)
    criterion, distil_criterion = get_loss_distillation()

    train_loader = get_dataloader(data, BATCH_SIZE, train=True)

    for epoch in range(EPOCHS):
        all_preds_student = []
        all_labels_student = []
        all_preds_teacher = []
        all_labels_teacher = []
        total_loss = 0
        for i, batch in enumerate(train_loader):
            images = batch['image']
            labels = batch['labels']
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = student(images)
            loss = criterion(outputs, labels)
            with torch.no_grad():
                teacher_outputs = teacher(images).logits
            temperature = 4.0
            distil_loss = distil_criterion(
                torch.nn.functional.log_softmax(outputs / temperature, dim=1),
                torch.nn.functional.softmax(teacher_outputs / temperature, dim=1)
            ) * (temperature * temperature)
            loss = loss * 0.7 + distil_loss * 0.3
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

            preds_student = outputs.argmax(dim=1).cpu().numpy()
            labels_np = labels.cpu().numpy()
            all_preds_student.append(preds_student)
            all_labels_student.append(labels_np)
            preds_teacher = teacher_outputs.argmax(dim=1).cpu().numpy()
            all_preds_teacher.append(preds_teacher)
            all_labels_teacher.append(labels_np)
            
        all_preds_student = np.concatenate(all_preds_student)
        all_labels_student = np.concatenate(all_labels_student)
        all_preds_teacher = np.concatenate(all_preds_teacher)
        all_labels_teacher = np.concatenate(all_labels_teacher)
        f1_scores_stud = f1_score(all_labels_student, all_preds_student, average='weighted')
        f1_scores_teach = f1_score(all_labels_teacher, all_preds_teacher, average='weighted')
        print(f"Epoch [{epoch+1}/{EPOCHS}] completed. Student Average F1 Score: {f1_scores_stud:.4f}")
        print(f"Epoch [{epoch+1}/{EPOCHS}] completed. Teacher Average F1 Score: {f1_scores_teach:.4f}")
        print(f"Epoch [{epoch+1}/{EPOCHS}] completed. Average Loss: {total_loss/len(data):.4f}")
        with open(LOG_PATH, "a") as f:
            if epoch == 0:
                f.write("epoch,loss,f1_score\n")
            f.write(f"{epoch+1},{total_loss/len(data)},{f1_scores_stud:.4f}\n")
    torch.save(student.state_dict(), MODEL_SAVE_PATH)
    return student
    
def test(model, data):
    model.to(device)
    model.eval()
    
    test_loader = get_dataloader(data, BATCH_SIZE, train=False)
    correct = 0
    total = 0
    f1 = 0
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            images = batch['image']
            labels = batch['labels']
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            # outputs = model(images)
            predicted = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            f1 += f1_score(labels.cpu(), predicted.cpu(), average='weighted')
    print(f"Test Accuracy: {100 * correct / total:.2f}%")
    print(f"Test F1 Score: {f1 / (len(data) / BATCH_SIZE):.4f}")
    

def main():
    beans = get_data()
    num_labels = beans['train'].features['labels'].num_classes

    # mean, std = compute_mean_std(beans['train'])
    # print(f"Computed mean: {mean}, std: {std}")

    teacher = get_resnet18(num_labels=num_labels, checkpoint_path="models/resnet18_model.pth")
    
    student = get_tinyHomemadeCNN(num_labels=num_labels)

    train_data = beans['train']
    test_data = beans['test']

    train(student, teacher, train_data)
    test(student, test_data)

if __name__ == '__main__':
    main()