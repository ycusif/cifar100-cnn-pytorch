import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100

# ============================================================

# 1. CONFIGURATION

# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 64
NUM_EPOCHS = 50
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 0.0005

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)

# ============================================================

# 2. DATA TRANSFORMATIONS

# ============================================================

def define_transformations(mean, std):
train_transform = transforms.Compose([
transforms.RandomHorizontalFlip(),
transforms.RandomVerticalFlip(),
transforms.RandomRotation(15),
transforms.ToTensor(),
transforms.Normalize(mean, std)
])

```
val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

return train_transform, val_transform
```

# ============================================================

# 3. LOAD DATASET

# ============================================================

train_transform, val_transform = define_transformations(
CIFAR100_MEAN,
CIFAR100_STD
)

train_dataset = CIFAR100(
root="./data",
train=True,
download=True,
transform=train_transform
)

val_dataset = CIFAR100(
root="./data",
train=False,
download=True,
transform=val_transform
)

train_loader = DataLoader(
train_dataset,
batch_size=BATCH_SIZE,
shuffle=True
)

val_loader = DataLoader(
val_dataset,
batch_size=BATCH_SIZE,
shuffle=False
)

num_classes = len(train_dataset.classes)

# ============================================================

# 4. CNN BLOCK

# ============================================================

class CNNBlock(nn.Module):

```
def __init__(
    self,
    in_channels,
    out_channels,
    kernel_size=3,
    padding=1
):
    super().__init__()

    self.block = nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
    )

def forward(self, x):
    return self.block(x)
```

# ============================================================

# 5. CNN MODEL

# ============================================================

class SimpleCNN(nn.Module):

```
def __init__(self, num_classes):
    super().__init__()

    self.conv1_block = CNNBlock(3, 32)
    self.conv2_block = CNNBlock(32, 64)
    self.conv3_block = CNNBlock(64, 128)

    self.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(128 * 4 * 4, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)
    )

def forward(self, x):
    x = self.conv1_block(x)
    x = self.conv2_block(x)
    x = self.conv3_block(x)
    x = self.classifier(x)

    return x
```

# ============================================================

# 6. TRAINING FUNCTION

# ============================================================

def train_epoch(
model,
train_loader,
loss_function,
optimizer,
device
):
model.train()

```
running_loss = 0.0
total = 0
correct = 0

for images, labels in train_loader:

    images = images.to(device)
    labels = labels.to(device)

    optimizer.zero_grad()

    outputs = model(images)

    loss = loss_function(
        outputs,
        labels
    )

    loss.backward()
    optimizer.step()

    running_loss += (
        loss.item() * images.size(0)
    )

    _, predicted = torch.max(
        outputs,
        1
    )

    total += labels.size(0)

    correct += (
        predicted == labels
    ).sum().item()

epoch_loss = (
    running_loss /
    len(train_loader.dataset)
)

epoch_accuracy = (
    100.0 * correct / total
)

return epoch_loss, epoch_accuracy
```

# ============================================================

# 7. VALIDATION FUNCTION

# ============================================================

def validate_epoch(
model,
val_loader,
loss_function,
device
):
model.eval()

```
running_loss = 0.0
total = 0
correct = 0

with torch.no_grad():

    for images, labels in val_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss = loss_function(
            outputs,
            labels
        )

        running_loss += (
            loss.item() * images.size(0)
        )

        _, predicted = torch.max(
            outputs,
            1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()

epoch_loss = (
    running_loss /
    len(val_loader.dataset)
)

epoch_accuracy = (
    100.0 * correct / total
)

return epoch_loss, epoch_accuracy
```

# ============================================================

# 8. TRAINING LOOP

# ============================================================

def training_loop(
model,
train_loader,
val_loader,
loss_function,
optimizer,
device,
num_epochs
):
model.to(device)

```
best_val_accuracy = 0.0
best_model_state = None
best_epoch = 0

train_losses = []
train_accuracies = []

val_losses = []
val_accuracies = []

print(f"Using device: {device}")
print("\n--- Training Started ---")

for epoch in range(num_epochs):

    train_loss, train_accuracy = train_epoch(
        model,
        train_loader,
        loss_function,
        optimizer,
        device
    )

    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    val_loss, val_accuracy = validate_epoch(
        model,
        val_loader,
        loss_function,
        device
    )

    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)

    print(
        f"Epoch [{epoch + 1}/{num_epochs}] "
        f"Train Loss: {train_loss:.4f} "
        f"Train Accuracy: {train_accuracy:.2f}% "
        f"Val Loss: {val_loss:.4f} "
        f"Val Accuracy: {val_accuracy:.2f}%"
    )

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy
        best_epoch = epoch + 1

        best_model_state = copy.deepcopy(
            model.state_dict()
        )

print("\n--- Finished Training ---")

if best_model_state is not None:

    model.load_state_dict(
        best_model_state
    )

    print(
        f"Best validation accuracy: "
        f"{best_val_accuracy:.2f}%"
    )

    print(
        f"Best epoch: {best_epoch}"
    )

metrics = [
    train_losses,
    train_accuracies,
    val_losses,
    val_accuracies
]

return model, metrics
```

# ============================================================

# 9. EVALUATION

# ============================================================

def evaluate_model(
model,
test_loader,
device
):
model.eval()

```
total = 0
correct = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(
            outputs,
            1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()

accuracy = 100.0 * correct / total

return accuracy
```

# ============================================================

# 10. PER-CLASS ACCURACY

# ============================================================

def calculate_class_accuracy(
model,
test_loader,
classes,
device
):
model.eval()

```
correct = [0] * len(classes)
total = [0] * len(classes)

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predictions = torch.max(
            outputs,
            1
        )

        for label, prediction in zip(
            labels,
            predictions
        ):
            label_index = label.item()

            total[label_index] += 1

            if label_index == prediction.item():
                correct[label_index] += 1

class_accuracies = {}

for i, class_name in enumerate(classes):

    if total[i] > 0:
        accuracy = (
            100.0 *
            correct[i] /
            total[i]
        )
    else:
        accuracy = 0.0

    class_accuracies[class_name] = accuracy

return class_accuracies
```

# ============================================================

# 11. MAIN

# ============================================================

def main():

```
model = SimpleCNN(
    num_classes=num_classes
)

loss_function = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

trained_model, training_metrics = training_loop(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    loss_function=loss_function,
    optimizer=optimizer,
    device=device,
    num_epochs=NUM_EPOCHS
)

test_accuracy = evaluate_model(
    trained_model,
    val_loader,
    device
)

print(
    f"\nFinal Test Accuracy: "
    f"{test_accuracy:.2f}%"
)

class_accuracies = calculate_class_accuracy(
    trained_model,
    val_loader,
    train_dataset.classes,
    device
)

print("\n--- Per-Class Accuracy ---\n")

for class_name, accuracy in class_accuracies.items():

    print(
        f"{class_name:15s}: "
        f"{accuracy:.2f}%"
    )

torch.save(
    trained_model.state_dict(),
    "simple_cnn_cifar100.pth"
)

print(
    "\nModel saved as "
    "simple_cnn_cifar100.pth"
)
```

if **name** == "**main**":
main()

