# ============================================================
# CIFAR-100 CNN Classifier - Version 2
# Improved CNN with deeper convolutional blocks
# ============================================================

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

import matplotlib.pyplot as plt


# ============================================================
# 1. Device
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")


# ============================================================
# 2. Normalization
# ============================================================

mean = (0.5071, 0.4867, 0.4408)
std = (0.2675, 0.2565, 0.2761)


# ============================================================
# 3. Data Transforms
# ============================================================

# Augmentation used only for training
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])


# No augmentation for validation/test
val_test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])


# ============================================================
# 4. Load CIFAR-100
# ============================================================

# Training dataset with augmentation
train_dataset_augmented = datasets.CIFAR100(
    root="./data",
    train=True,
    download=True,
    transform=train_transform
)


# Same 50,000 training images, but without augmentation
# This dataset is used to create the validation set.
train_dataset_clean = datasets.CIFAR100(
    root="./data",
    train=True,
    download=True,
    transform=val_test_transform
)


# Official CIFAR-100 test set
test_dataset = datasets.CIFAR100(
    root="./data",
    train=False,
    download=True,
    transform=val_test_transform
)


# ============================================================
# 5. Create Train / Validation Split
# ============================================================

train_size = 45000
val_size = 5000

# Fixed seed so the same split is created every time
generator = torch.Generator().manual_seed(42)

indices = torch.randperm(
    len(train_dataset_augmented),
    generator=generator
).tolist()


train_indices = indices[:train_size]
val_indices = indices[train_size:]


# Training subset uses augmented images
train_dataset = Subset(
    train_dataset_augmented,
    train_indices
)


# Validation subset uses clean images
val_dataset = Subset(
    train_dataset_clean,
    val_indices
)


print(f"\nTraining samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Test samples: {len(test_dataset)}")


# ============================================================
# 6. DataLoaders
# ============================================================

batch_size = 64

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False
)


# ============================================================
# 7. CNN Block
# ============================================================

class CNNBlock(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.block = nn.Sequential(

            # First convolution
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(),

            # Second convolution
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(),

            # Reduce spatial dimensions
            nn.MaxPool2d(kernel_size=2)
        )


    def forward(self, x):

        return self.block(x)


# ============================================================
# 8. Improved CNN Model
# ============================================================

class SimpleCNN(nn.Module):

    def __init__(self, num_classes):

        super().__init__()


        # Feature extractor
        self.conv1_block = CNNBlock(3, 64)

        self.conv2_block = CNNBlock(64, 128)

        self.conv3_block = CNNBlock(128, 256)


        # Classification head
        self.classifier = nn.Sequential(

            nn.Flatten(),

            # 256 × 4 × 4 = 4096
            nn.Linear(256 * 4 * 4, 512),

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


# ============================================================
# 9. Create Model
# ============================================================

num_classes = len(train_dataset_augmented.classes)

model = SimpleCNN(
    num_classes=num_classes
)

model = model.to(device)

print("\nModel:")
print(model)


# ============================================================
# 10. Loss Function and Optimizer
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.0005,
    weight_decay=0.0005
)


# ============================================================
# 11. Training Configuration
# ============================================================

num_epochs = 50

best_val_accuracy = 0.0

best_model_path = "best_cifar100_cnn_v2.pth"


# Store training history
train_losses = []
train_accuracies = []

val_losses = []
val_accuracies = []


# ============================================================
# 12. Training Loop
# ============================================================

for epoch in range(num_epochs):

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0


    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)


        # Clear previous gradients
        optimizer.zero_grad()


        # Forward pass
        outputs = model(images)


        # Calculate loss
        loss = criterion(outputs, labels)


        # Backpropagation
        loss.backward()


        # Update weights
        optimizer.step()


        # Track loss
        running_loss += loss.item() * images.size(0)


        # Track accuracy
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()


    train_loss = running_loss / total

    train_accuracy = 100 * correct / total


    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    model.eval()

    val_running_loss = 0.0
    val_correct = 0
    val_total = 0


    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)


            # Forward pass
            outputs = model(images)


            # Calculate loss
            loss = criterion(outputs, labels)


            # Track loss
            val_running_loss += loss.item() * images.size(0)


            # Track accuracy
            _, predicted = torch.max(outputs, 1)

            val_total += labels.size(0)

            val_correct += (predicted == labels).sum().item()


    val_loss = val_running_loss / val_total

    val_accuracy = 100 * val_correct / val_total


    # --------------------------------------------------------
    # Store history
    # --------------------------------------------------------

    train_losses.append(train_loss)

    train_accuracies.append(train_accuracy)

    val_losses.append(val_loss)

    val_accuracies.append(val_accuracy)


    # --------------------------------------------------------
    # Save best model
    # --------------------------------------------------------

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            best_model_path
        )

        best_marker = " ← Best"

    else:

        best_marker = ""


    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print(
        f"Epoch [{epoch + 1}/{num_epochs}] "
        f"Train Loss: {train_loss:.4f} "
        f"Train Accuracy: {train_accuracy:.2f}% "
        f"Val Loss: {val_loss:.4f} "
        f"Val Accuracy: {val_accuracy:.2f}%"
        f"{best_marker}"
    )


# ============================================================
# 13. Load Best Model
# ============================================================

print("\nTraining complete.")

print(
    f"Best Validation Accuracy: "
    f"{best_val_accuracy:.2f}%"
)

print(
    f"Loading best model from: "
    f"{best_model_path}"
)


model.load_state_dict(
    torch.load(
        best_model_path,
        map_location=device
    )
)


# ============================================================
# 14. Final Test Evaluation
# ============================================================

model.eval()

test_correct = 0
test_total = 0


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)


        outputs = model(images)


        _, predicted = torch.max(outputs, 1)


        test_total += labels.size(0)

        test_correct += (
            predicted == labels
        ).sum().item()


test_accuracy = 100 * test_correct / test_total


print(
    f"\nFinal Test Accuracy: "
    f"{test_accuracy:.2f}%"
)


# ============================================================
# 15. Per-Class Accuracy
# ============================================================

class_correct = [0] * num_classes
class_total = [0] * num_classes


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)


        outputs = model(images)

        _, predicted = torch.max(outputs, 1)


        for label, prediction in zip(labels, predicted):

            class_total[label.item()] += 1

            if label == prediction:

                class_correct[label.item()] += 1


print("\nPer-Class Accuracy:")


for i, class_name in enumerate(
    test_dataset.classes
):

    if class_total[i] > 0:

        accuracy = (
            100
            * class_correct[i]
            / class_total[i]
        )

    else:

        accuracy = 0.0


    print(
        f"{class_name:15s}: "
        f"{accuracy:.2f}%"
    )


# ============================================================
# 16. Plot Training and Validation Loss
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(
    train_losses,
    label="Training Loss"
)

plt.plot(
    val_losses,
    label="Validation Loss"
)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title("Training and Validation Loss")

plt.legend()

plt.show()


# ============================================================
# 17. Plot Training and Validation Accuracy
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(
    train_accuracies,
    label="Training Accuracy"
)

plt.plot(
    val_accuracies,
    label="Validation Accuracy"
)

plt.xlabel("Epoch")

plt.ylabel("Accuracy (%)")

plt.title("Training and Validation Accuracy")

plt.legend()

plt.show()
