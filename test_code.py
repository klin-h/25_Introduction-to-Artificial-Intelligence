import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)       
        self.conv2 = nn.Conv2d(20, 20, 3)       
        self.fc1 = nn.Linear(20 * 10 * 10, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        in_size = x.size(0)
        out = self.conv1(x)
        out = F.relu(out)
        out = F.max_pool2d(out, 2, 2)
        out = self.conv2(out)
        out = F.relu(out)
        out = out.view(in_size, -1)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.log_softmax(out, dim=1)
        return out


model = ConvNet()
model.load_state_dict(torch.load("mnist_cnn.pth", map_location=torch.device('cpu')))
model.eval()

# 图像预处理流程（保持一致）
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

def predict_image(image_path):
    image = Image.open(image_path)
    image_processed = transform(image)
    image = image_processed.unsqueeze(0)  # 加 batch 维度

    # 显示处理后的图像
    plt.imshow(image_processed.squeeze(0).squeeze(0).numpy(), cmap='gray')
    plt.title("Processed Image")
    plt.show()

    # 预测
    with torch.no_grad():
        output = model(image)
        pred = output.argmax(dim=1, keepdim=True)
        print(f"Prediction Result: {pred.item()}")

    # 可视化结果
    plt.imshow(image.squeeze(0).squeeze(0).numpy(), cmap='gray')
    plt.title(f"Predicted Digit: {pred.item()}")
    plt.show()

# ✅ 调用预测函数
predict_image("three02.png")  # 替换为你的实际图像路径
