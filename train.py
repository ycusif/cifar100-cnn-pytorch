import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
import matplotlib.pyplot as plt
import copy


# ============================================================
# 1. DEVICE
# ============================================================

device = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu'
)

print(f"Using device: {device}")


# ============================================================
# 2. CIFAR-100 NORMALIZATION VALUES
# ============================================================

cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)


# ============================================================
# 3. DATA TRANSFORMATIONS
# ============================================================

def define_transformation(mean, std):

    train_transformation = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    val_transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    return train_transformation, val_transformation


# Verify transformations
print("\n--- Verifying Transformations ---\n")

train_transform_verify, val_transform_verify = define_transformation(
    cifar100_mean,
    cifar100_std
)

print("Training Transformations:")
print(train_transform_verify)

print("-" * 30)

print("\nValidation Transformations:")
print(val_transform_verify)


# Create actual transformations
train_transform, val_transform = define_transformation(
    cifar100_mean,
    cifar100_std
)


# ============================================================
# 4. LOAD CIFAR-100 DATASET
# ============================================================

train_dataset = CIFAR100(
    root='./data',
    train=True,
    download=True,
    transform=train_transform
)

val_dataset = CIFAR100(
    root='./data',
    train=False,
    download=True,
    transform=val_transform
)


print("\nNumber of classes:", len(train_dataset.classes))
print("Number of training images:", len(train_dataset))
print("Number of validation images:", len(val_dataset))


# ============================================================
# 5. DATALOADERS
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


# ============================================================
# 6. CNN BLOCK
# ============================================================

class CNNBLOCK(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        padding=1
    ):

        super(CNNBLOCK, self).__init__()

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


# ============================================================
# 7. COMPLETE CNN
# ============================================================

class SimpleCNN(nn.Module):

    def __init__(self, num_classes):

        super(SimpleCNN, self).__init__()

        # Feature extractor

        self.conv1_block = CNNBLOCK(3, 32)

        self.conv2_block = CNNBLOCK(32, 64)

        self.conv3_block = CNNBLOCK(64, 128)


        # Classification head

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                128 * 4 * 4,
                512
            ),

            nn.ReLU(),

            nn.Dropout(0.5),

            nn.Linear(
                512,
                num_classes
            )
        )


    def forward(self, x):

        x = self.conv1_block(x)

        x = self.conv2_block(x)

        x = self.conv3_block(x)

        x = self.classifier(x)

        return x


# ============================================================
# 8. VERIFY CNN
# ============================================================

print("\n--- Verifying SimpleCNN ---\n")


num_classes = len(train_dataset.classes)

verify_simple_cnn = SimpleCNN(
    num_classes=num_classes
)


print("Model Structure:\n")
print(verify_simple_cnn)


# Dummy input
dummy_input = torch.randn(
    64,
    3,
    32,
    32
)

print(
    f"\nInput tensor shape: {dummy_input.shape}"
)


# Forward pass
output = verify_simple_cnn(dummy_input)

print(
    f"Output tensor shape: {output.shape}"
)


# ============================================================
# 9. CREATE FINAL MODEL
# ============================================================

num_classes = len(train_dataset.classes)

model = SimpleCNN(
    num_classes=num_classes
)


# ============================================================
# 10. LOSS FUNCTION + OPTIMIZER
# ============================================================

loss_function = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.0005,
    weight_decay=0.0005
)


# ============================================================
# 11. TRAINING FUNCTION
# ============================================================

def train_epoch(
    model,
    train_loader,
    loss_function,
    optimizer,
    device
):

    model.train()

    running_loss = 0.0

    total = 0
    correct = 0


    for images, labels in train_loader:

        # Move data to device
        images = images.to(device)
        labels = labels.to(device)


        # Clear old gradients
        optimizer.zero_grad()


        # Forward pass
        outputs = model(images)


        # Calculate loss
        loss = loss_function(
            outputs,
            labels
        )


        # Backpropagation
        loss.backward()


        # Update weights
        optimizer.step()


        # Accumulate loss
        running_loss += (
            loss.item() *
            images.size(0)
        )


        # Calculate predictions
        _, predicted = torch.max(
            outputs,
            1
        )


        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()


    # Average training loss
    epoch_loss = (
        running_loss /
        len(train_loader.dataset)
    )


    # Training accuracy
    epoch_accuracy = (
        100.0 *
        correct /
        total
    )


    return epoch_loss, epoch_accuracy


# ============================================================
# 12. VALIDATION FUNCTION
# ============================================================

def validate_epoch(
    model,
    val_loader,
    loss_function,
    device
):

    model.eval()

    running_val_loss = 0.0

    total = 0
    correct = 0


    with torch.no_grad():

        for images, labels in val_loader:

            # Move data to device
            images = images.to(device)
            labels = labels.to(device)


            # Forward pass
            outputs = model(images)


            # Calculate validation loss
            val_loss = loss_function(
                outputs,
                labels
            )


            # Accumulate loss
            running_val_loss += (
                val_loss.item() *
                images.size(0)
            )


            # Predictions
            _, predicted = torch.max(
                outputs,
                1
            )


            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()


    # Average validation loss
    epoch_val_loss = (
        running_val_loss /
        len(val_loader.dataset)
    )


    # Validation accuracy
    epoch_accuracy = (
        100.0 *
        correct /
        total
    )


    return epoch_val_loss, epoch_accuracy


# ============================================================
# 13. TRAINING LOOP
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

    # Move model to GPU
    model.to(device)


    # Best model tracking
    best_val_accuracy = 0.0

    best_model_state = None

    best_epoch = 0


    # Store metrics
    train_losses = []

    train_accuracies = []

    val_losses = []

    val_accuracies = []


    print("\n--- Training Started ---")


    for epoch in range(num_epochs):


        # ----------------------------------------------------
        # Training
        # ----------------------------------------------------

        epoch_loss, epoch_train_accuracy = train_epoch(
            model,
            train_loader,
            loss_function,
            optimizer,
            device
        )


        train_losses.append(
            epoch_loss
        )

        train_accuracies.append(
            epoch_train_accuracy
        )


        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        epoch_val_loss, epoch_val_accuracy = validate_epoch(
            model,
            val_loader,
            loss_function,
            device
        )


        val_losses.append(
            epoch_val_loss
        )

        val_accuracies.append(
            epoch_val_accuracy
        )


        # ----------------------------------------------------
        # Print results
        # ----------------------------------------------------

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] "
            f"Train Loss: {epoch_loss:.4f} "
            f"Train Accuracy: {epoch_train_accuracy:.2f}% "
            f"Val Loss: {epoch_val_loss:.4f} "
            f"Val Accuracy: {epoch_val_accuracy:.2f}%"
        )


        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        if epoch_val_accuracy > best_val_accuracy:

            best_val_accuracy = epoch_val_accuracy

            best_epoch = epoch + 1

            best_model_state = copy.deepcopy(
                model.state_dict()
            )


    print("\n--- Finished Training ---")


    # Restore best model
    if best_model_state is not None:

        print(
            f"\n--- Returning best model ---"
        )

        print(
            f"Best validation accuracy: "
            f"{best_val_accuracy:.2f}%"
        )

        print(
            f"Achieved at epoch: "
            f"{best_epoch}"
        )


        model.load_state_dict(
            best_model_state
        )


    # Store metrics
    metrics = [
        train_losses,
        train_accuracies,
        val_losses,
        val_accuracies
    ]


    return model, metrics


# ============================================================
# 14. TRAIN THE MODEL
# ============================================================

trained_model, training_metrics = training_loop(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    loss_function=loss_function,
    optimizer=optimizer,
    device=device,
    num_epochs=50
)


# ============================================================
# 15. EXTRACT TRAINING METRICS
# ============================================================

train_losses, train_accuracies, val_losses, val_accuracies = (
    training_metrics
)


# ============================================================
# 16. PLOT LOSS
# ============================================================

epochs = range(
    1,
    len(train_losses) + 1
)


plt.figure(figsize=(8, 5))


plt.plot(
    epochs,
    train_losses,
    label="Training Loss"
)


plt.plot(
    epochs,
    val_losses,
    label="Validation Loss"
)


plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title(
    "Training and Validation Loss"
)

plt.legend()

plt.grid(True)

plt.show()


# ============================================================
# 17. PLOT ACCURACY
# ============================================================

plt.figure(figsize=(8, 5))


plt.plot(
    epochs,
    train_accuracies,
    label="Training Accuracy"
)


plt.plot(
    epochs,
    val_accuracies,
    label="Validation Accuracy"
)


plt.xlabel("Epoch")

plt.ylabel("Accuracy (%)")

plt.title(
    "Training and Validation Accuracy"
)

plt.legend()

plt.grid(True)

plt.show()


# ============================================================
# 18. FINAL EVALUATION
# ============================================================

def evaluate_model(
    model,
    val_loader,
    device
):

    model.eval()

    total = 0
    correct = 0


    with torch.no_grad():

        for images, labels in val_loader:

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


    accuracy = (
        100.0 *
        correct /
        total
    )


    return accuracy


final_accuracy = evaluate_model(
    trained_model,
    val_loader,
    device
)


print(
    f"\nFinal Evaluation Accuracy: "
    f"{final_accuracy:.2f}%"
)


# ============================================================
# 19. VISUALIZE PREDICTIONS
# ============================================================

trained_model.eval()


images, labels = next(
    iter(val_loader)
)


images = images.to(device)

labels = labels.to(device)


with torch.no_grad():

    outputs = trained_model(images)

    _, predictions = torch.max(
        outputs,
        1
    )


# Undo normalization
mean = torch.tensor(
    cifar100_mean
).view(
    3,
    1,
    1
)

std = torch.tensor(
    cifar100_std
).view(
    3,
    1,
    1
)


plt.figure(
    figsize=(12, 8)
)


for i in range(12):

    image = images[i].cpu()


    # Reverse normalization
    image = (
        image *
        std +
        mean
    )


    # CHW → HWC
    image = image.permute(
        1,
        2,
        0
    )


    plt.subplot(
        3,
        4,
        i + 1
    )


    plt.imshow(
        image.clamp(0, 1)
    )


    true_class = val_dataset.classes[
        labels[i].item()
    ]


    predicted_class = val_dataset.classes[
        predictions[i].item()
    ]


    plt.title(
        f"True: {true_class}\n"
        f"Pred: {predicted_class}"
    )


    plt.axis("off")


plt.tight_layout()

plt.show()


# ============================================================
# 20. PER-CLASS ACCURACY
# ============================================================

def calculate_class_accuracy(
    model,
    val_loader,
    classes,
    device
):

    model.eval()


    correct = [0] * len(classes)

    total = [0] * len(classes)


    with torch.no_grad():

        for images, labels in val_loader:

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

                label = label.item()

                prediction = prediction.item()


                total[label] += 1


                if label == prediction:

                    correct[label] += 1


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


        class_accuracies[
            class_name
        ] = accuracy


    return class_accuracies


class_accuracies = calculate_class_accuracy(
    trained_model,
    val_loader,
    val_dataset.classes,
    device
)


print(
    "\n--- Per-Class Accuracy ---\n"
)


for class_name, accuracy in class_accuracies.items():

    print(
        f"{class_name:15s}: "
        f"{accuracy:.2f}%"
    )


# ============================================================
# 21. SAVE MODEL
# ============================================================

torch.save(
    trained_model.state_dict(),
    "simple_cnn_cifar100.pth"
)


print(
    "\nModel saved as "
    "simple_cnn_cifar100.pth"
)
