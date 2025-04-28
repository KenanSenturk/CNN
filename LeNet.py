import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from collections import OrderedDict

# 1. Veri setinin hazırlanması ve ön işleme
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # LeNet-5 32x32 giriş bekler
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST için standart normalizasyon değerleri
])

# Eğitim ve test veri setlerinin yüklenmesi
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Veri yükleyicilerin oluşturulması
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)



class LeNet5(nn.Module):
    """
    LeNet-5 CNN Modeli
    Input: 1x32x32
    Output: 10 sınıf (MNIST rakamları için)
    """

    def __init__(self):
        super(LeNet5, self).__init__()

        # Katman 1: Evrişimli Katman + ReLU + Maksimum Havuzlama
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Katman 2: Evrişimli Katman + ReLU + Maksimum Havuzlama
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Katman 3: Evrişimli Katman + ReLU
        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 120, kernel_size=5, stride=1, padding=0),
            nn.ReLU()
        )



        # Tam Bağlantılı Katmanlar
        self.fc1 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(84, 10)


    def forward(self, x):
        # Evrişimli katmanlar
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # Düzleştirme
        x = x.view(x.size(0), -1)

        # Tam bağlantılı katmanlar
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.fc2(x)

        return x


# Model oluşturma
model = LeNet5()
print(model)



def train(model, train_loader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0

    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()

        output = model(data)

        loss = criterion(output, target)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        # Her 100 batch'de bir ilerlemeyi göster
        if batch_idx % 100 == 99:
            print(f'Epoch: {epoch + 1}, Batch: {batch_idx + 1}, Loss: {running_loss / 100:.4f}')
            running_loss = 0.0


def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')
    return accuracy


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
num_epochs = 3

best_accuracy = 0.0

for epoch in range(num_epochs):
    train(model, train_loader, criterion, optimizer, epoch)
    accuracy = test(model, test_loader)

    if accuracy > best_accuracy:
        best_accuracy = accuracy


print(f'best accuracy: {best_accuracy:.2f}%')