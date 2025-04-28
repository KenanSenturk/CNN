import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import matplotlib.pyplot as plt
import numpy as np
import time

# GPU kullanımını kontrol etme
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Veri ön işleme - AlexNet için girişler 224x224 olmalı
transform_train = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# CIFAR-10 veri setini yükle
batch_size = 64

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

def create_model(pretrained=False):
    """
    AlexNet modelini oluşturur ve ayarlar
    Args:
        pretrained: Önceden eğitilmiş ağırlıkları kullanıp kullanmama seçeneği
    Returns:
        model: Oluşturulan AlexNet modeli
    """
    if pretrained:
        model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
    else:
        model = models.alexnet(weights=None)

    # CIFAR-10 için son katmanı 10 olmalıdır
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs, 10)

    return model


model = create_model(pretrained=True)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)


def train_model(model, criterion, optimizer, scheduler, num_epochs=10):
    start_time = time.time()

    # Sonuçları kaydetmek için
    train_losses = []
    val_accuracies = []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        model.train()
        running_loss = 0.0

        for i, (inputs, labels) in enumerate(trainloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

            if i % 50 == 49:
                print(f'Batch {i + 1}/{len(trainloader)} - Loss: {loss.item():.4f}')

        epoch_loss = running_loss / len(trainset)
        train_losses.append(epoch_loss)
        print(f'Train Loss: {epoch_loss:.4f}')

        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in testloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                total += labels.size(0)
                correct += (preds == labels).sum().item()

        epoch_acc = correct / total
        val_accuracies.append(epoch_acc)
        print(f'Test Accuracy: {epoch_acc:.4f}')

        # Öğrenme oranı güncellemesi
        scheduler.step()

    time_elapsed = time.time() - start_time
    print(f'Eğitim {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s sürdü')
    print(f'En yüksek test doğruluğu: {max(val_accuracies):.4f}')

    return model, train_losses, val_accuracies

# Test seti üzerinde modelin performansını değerlendir
def evaluate_model(model, testloader, classes):
    model.eval()

    # Tüm sınıflar için tahminleri topla
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            c = (preds == labels).squeeze()

            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    # Her sınıf için doğruluk oranlarını yazdır
    for i in range(10):
        print(f'Accuracy of {classes[i]}: {100 * class_correct[i] / class_total[i]:.2f}%')

    # Genel doğruluk oranı
    print(f'Genel doğruluk: {100 * sum(class_correct) / sum(class_total):.2f}%')

if __name__ == "__main__":
    num_epochs = 1
    model, train_losses, val_accuracies = train_model(model, criterion, optimizer, scheduler, num_epochs=num_epochs)

    evaluate_model(model, testloader, classes)

