import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
import torchvision.models as models
from torchinfo import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_dir = "../../face_dataset_224_augment"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

mobilenet = models.mobilenet_v2(pretrained=True)
for param in mobilenet.features[:6].parameters():
    param.requires_grad = False

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        se = self.global_avg_pool(x).view(b, c)
        se = self.fc(se).view(b, c, 1, 1)
        return x * se.expand_as(x)

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, expansion=6, stride=1):
        super(InvertedResidual, self).__init__()
        hidden_dim = in_channels * expansion
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.use_residual = stride == 1 and in_channels == out_channels

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)


class ResNetBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

class ResNetBottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, mid_channels, stride=1):
        super(ResNetBottleneck, self).__init__()
        out_channels = mid_channels * self.expansion  # Số kênh đầu ra
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

class ConvNeXtBlock(nn.Module):
    def __init__(self, in_channels):
        super(ConvNeXtBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=7, padding=3, groups=in_channels),
            nn.BatchNorm2d(in_channels),  # Thay vì LayerNorm([in_channels, 1, 1])
            nn.Conv2d(in_channels, 4 * in_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(4 * in_channels, in_channels, kernel_size=1),
        )

    def forward(self, x):
        return x + self.conv(x)


class CustomModel(nn.Module):
    def __init__(self, mobilenet, num_classes=3):
        super(CustomModel, self).__init__()
        self.backbone = mobilenet.features[:6]

        # SEBlock khuếch đại đặc trưng quan trọng
        self.se_block = SEBlock(32)

        # Inverted Residual Blocks tận dụng hiệu suất tính toán
        self.inverted_residual1 = InvertedResidual(32, 128, expansion=6, stride=1)
        self.inverted_residual2 = InvertedResidual(128, 128, expansion=6, stride=1)

        # ResNet Basic Blocks học các đặc trưng phức tạp hơn
        self.resnet_basic1 = ResNetBasicBlock(128, 256, stride=1)
        self.resnet_basic2 = ResNetBasicBlock(256, 256, stride=1)

        # ResNet Bottleneck Blocks tăng độ sâu của mô hình
        self.resnet_bottleneck1 = ResNetBottleneck(256, mid_channels=64, stride=1)
        self.resnet_bottleneck2 = ResNetBottleneck(256, mid_channels=64, stride=1)

        # ConvNeXt Block tinh chỉnh đặc trưng
        self.convnext_block = ConvNeXtBlock(256)

        # Adaptive Pooling và Classifier
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.se_block(x)
        x = self.inverted_residual1(x)
        x = self.inverted_residual2(x)
        x = self.resnet_basic1(x)
        x = self.resnet_basic2(x)
        x = self.resnet_bottleneck1(x)
        x = self.resnet_bottleneck2(x)
        x = self.convnext_block(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x

# Khởi tạo model
model = CustomModel(mobilenet).to(device)
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.00005, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=1)


def train_model(model, train_loader, val_loader, epochs=50):
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total_samples = 0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total_samples += images.size(0)

        train_loss = total_loss / total_samples
        train_acc = 100.0 * correct / total_samples


        model.eval()
        val_loss, val_correct, val_samples = 0, 0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_samples += images.size(0)

        val_loss /= val_samples
        val_acc = 100.0 * val_correct / val_samples

        print(f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        scheduler.step(val_loss)


train_model(model, train_loader, val_loader, epochs=50)

torch.save(model.state_dict(), "model_hybrid_mobilenetV2.pth")

summary(model, input_size=(1, 3, 224, 224))