import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1. Hyperparameters 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size = 28 * 28  # MNIST 이미지 크기: 28x28
hidden_size = 512
num_classes = 10  # MNIST 클래스 수: 0-9
num_epochs = 5
batch_size = 100
learning_rate = 0.001

# 2. MNIST 데이터셋 로드 및 전처리
# PyTorch 모델에 맞게 데이터를 Tensor로 변환하고 정규화(Normalize)한다.
transform = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize((0.1307,), (0.3081,))  # MNIST 데이터셋의 평균과 표준편차
])

# 데이터셋 다운로드 및 DataLoader 설정
# 출처: https://pytorch.org/vision/main/generated/torchvision.datasets.MNIST.html
train_dataset = datasets.MNIST(root='./data',
                               train=True,
                               transform=transform,
                               download=True)

test_dataset = datasets.MNIST(root='./data',
                              train=False,
                              transform=transform)

# 출처: https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True)

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=batch_size,
                         shuffle=False)


# 3. Neural Network 모델 정의
class NeuralNet(nn.Module):
	def __init__(self, input_size, hidden_size, num_classes):
		super(NeuralNet, self).__init__()
		self.fc1 = nn.Linear(input_size, hidden_size)
		self.relu = nn.ReLU()
		self.fc2 = nn.Linear(hidden_size, num_classes)

	def forward(self, x):
		# 입력 이미지를 1차원 벡터로 변환
		x = x.reshape(-1, input_size)
		out = self.fc1(x)
		out = self.relu(out)
		out = self.fc2(out)
		return out


model = NeuralNet(input_size, hidden_size, num_classes).to(device)

# 4. Loss 함수 및 Optimizer 정의
# 다중 클래스 분류 문제이므로 CrossEntropyLoss를 사용한다.
criterion = nn.CrossEntropyLoss()
# Adam optimizer를 사용하여 모델의 가중치를 업데이트한다.
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 5. 모델 학습 (Training)
total_steps = len(train_loader)
for epoch in range(num_epochs):
	for i, (images, labels) in enumerate(train_loader):
		# 데이터를 device에 할당
		images = images.to(device)
		labels = labels.to(device)

		# Forward pass
		outputs = model(images)
		loss = criterion(outputs, labels)

		# Backward pass 및 가중치 업데이트
		optimizer.zero_grad()  # 이전 gradient 값을 초기화
		loss.backward()  # backpropagation을 통해 gradient 계산
		optimizer.step()  # 계산된 gradient를 바탕으로 가중치 업데이트

		if (i + 1) % 100 == 0:
			print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{total_steps}], Loss: {loss.item():.4f}')

# 6. 모델 평가 (Evaluation)
# 평가 단계에서는 gradient를 계산할 필요가 없으므로 torch.no_grad()를 사용한다.
model.eval()  # 모델을 평가 모드로 전환
with torch.no_grad():
	correct = 0
	total = 0
	for images, labels in test_loader:
		images = images.to(device)
		labels = labels.to(device)

		outputs = model(images)
		_, predicted = torch.max(outputs.data, 1)  # 가장 높은 확률을 가진 클래스를 예측값으로 선택
		total += labels.size(0)
		correct += (predicted == labels).sum().item()

	print(f'Accuracy of the network on the 10000 test images: {100 * correct / total:.2f} %')