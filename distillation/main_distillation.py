from transformers import ResNetConfig, ResNetForImageClassification, Dinov2WithRegistersForImageClassification
from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import f1_score
from constants import EPOCHS, BATCH_SIZE, LEARNING_RATE, MODEL_SAVE_PATH

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_dinov2(num_labels=None):
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
    if num_labels is not None:
        configuration.num_labels = num_labels
    model = ResNetForImageClassification(configuration)
    return model

def get_data():
    dataset = load_dataset("beans", split="train[:5000]")
    dataset = dataset.train_test_split(test_size=0.2)
    return dataset

def get_dataloader(data, batch_size):
    dataset = data.select_columns(['image', 'labels'])
    dataset = dataset.map(lambda x: {'image': x['image'], 'labels': x['labels']})
    dataset.set_format(type='torch', columns=['image', 'labels'])
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
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

    optimizer = torch.optim.Adam(student.parameters(), lr=1e-4)
    criterion, distil_criterion = get_loss_distillation()

    train_loader = get_dataloader(data, BATCH_SIZE)

    for epoch in range(EPOCHS):
        f1_scores_stud = 0
        f1_scores_teach = 0
        total_loss = 0
        for i, batch in enumerate(train_loader):
            images = batch['image']
            labels = batch['labels']
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = student(images).logits
            loss = criterion(outputs, labels)
            with torch.no_grad():
                teacher_outputs = teacher(images).logits
            temperature = 4.0
            distil_loss = distil_criterion(
                torch.nn.functional.log_softmax(outputs / temperature, dim=1),
                torch.nn.functional.softmax(teacher_outputs / temperature, dim=1)
            ) * (temperature * temperature)
            loss = loss + distil_loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

            # f1_scores_stud.append(f1_score(labels.cpu(), outputs.argmax(dim=1).cpu(), average='weighted'))
            # f1_scores_teach.append(f1_score(labels.cpu(), teacher_outputs.argmax(dim=1).cpu(), average='weighted'))
            f1_scores_stud += f1_score(labels.cpu(), outputs.argmax(dim=1).cpu(), average='weighted')
            f1_scores_teach += f1_score(labels.cpu(), teacher_outputs.argmax(dim=1).cpu(), average='weighted')
        print(f"Epoch [{epoch+1}/{EPOCHS}] completed. Average Student F1 Score: {f1_scores_stud:.4f}")
        print(f"Epoch [{epoch+1}/{EPOCHS}] completed. Average Teacher F1 Score: {f1_scores_teach:.4f}")
        print(f"Epoch [{epoch+1}/{EPOCHS}] completed. Average Loss: {total_loss/len(data):.4f}")
    torch.save(student.state_dict(), MODEL_SAVE_PATH)
    return student
    
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
    beans = get_data()
    num_labels = beans['train'].features['labels'].num_classes

    teacher = get_dinov2(num_labels=num_labels)
    student = get_resnet18(num_labels=num_labels)

    train_data = beans['train']
    test_data = beans['test']
    dinov2 = train(dinov2, train_data)
    test(dinov2, test_data)

if __name__ == '__main__':
    main()