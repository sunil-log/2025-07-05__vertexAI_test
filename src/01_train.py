import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from dataclasses import dataclass
import torch.nn.functional as F


# 1. 설정 (Configuration)
# dataclass를 사용하여 하이퍼파라미터 및 설정을 체계적으로 관리한다.
@dataclass
class TrainingConfig:
	device: str = "cuda" if torch.cuda.is_available() else "cpu"
	input_channels: int = 1
	num_classes: int = 10
	num_epochs: int = 10
	batch_size: int = 128
	learning_rate: float = 0.01
	weight_decay: float = 1e-4  # L2 Regularization을 위한 가중치 감쇠
	dropout_rate: float = 0.5


# 2. 데이터 로더 (Data Loaders)
def get_data_loaders(batch_size: int):
	"""
	Fashion-MNIST 데이터셋을 위한 train_loader와 test_loader를 생성하고 반환한다.
	출처: https://pytorch.org/vision/main/generated/torchvision.datasets.FashionMNIST.html
	"""
	# Fashion-MNIST에 맞는 정규화 값으로 변경이 필요할 수 있다.
	# 우선 MNIST 값으로 유지한다.
	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.1307,), (0.3081,))
	])

	train_dataset = datasets.FashionMNIST(root='./data', train=True, transform=transform, download=True)
	test_dataset = datasets.FashionMNIST(root='./data', train=False, transform=transform)

	# 출처: https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
	train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
	test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

	return train_loader, test_loader


# 3. 모델 정의 (Model Definition)
# CNN(Convolutional Neural Network) 기반의 복잡한 모델을 정의한다.
# Conv2d, BatchNorm2d, Dropout 등의 레이어를 포함한다.
class ComplexCNN(nn.Module):
	def __init__(self, config: TrainingConfig):
		super(ComplexCNN, self).__init__()
		self.conv_block1 = nn.Sequential(
			nn.Conv2d(in_channels=config.input_channels, out_channels=32, kernel_size=3, padding=1),
			nn.BatchNorm2d(32),  # Batch Normalization
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2)
		)
		self.conv_block2 = nn.Sequential(
			nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
			nn.BatchNorm2d(64),  # Batch Normalization
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2)
		)

		# Convolutional 레이어를 거친 후의 feature_dim을 계산한다.
		# Fashion-MNIST 이미지(28x28) -> MaxPool1 -> (14x14) -> MaxPool2 -> (7x7)
		self.flattened_dim = 64 * 7 * 7

		self.fc_block = nn.Sequential(
			nn.Linear(self.flattened_dim, 1024),
			nn.ReLU(),
			nn.Dropout(config.dropout_rate),  # Dropout
			nn.Linear(1024, config.num_classes)
		)

	def forward(self, x):
		x = self.conv_block1(x)
		x = self.conv_block2(x)
		x = x.view(-1, self.flattened_dim)  # Flatten
		logits = self.fc_block(x)
		return logits


# 4. 트레이너 클래스 (Trainer Class)
# 학습 및 평가 로직을 캡슐화한다.
class Trainer:
	def __init__(self, model: nn.Module, train_loader: DataLoader, test_loader: DataLoader, config: TrainingConfig):
		self.model = model.to(config.device)
		self.train_loader = train_loader
		self.test_loader = test_loader
		self.config = config

		self.criterion = nn.CrossEntropyLoss()
		self.optimizer = optim.Adam(
			self.model.parameters(),
			lr=config.learning_rate,
			weight_decay=config.weight_decay  # L2 Regularization
		)
		# CosineAnnealingLR 스케줄러를 사용하여 learning rate를 동적으로 조절한다.
		# 출처: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html
		self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config.num_epochs)

	def _train_epoch(self, epoch: int):
		self.model.train()
		total_steps = len(self.train_loader)
		for i, (images, labels) in enumerate(self.train_loader):
			images, labels = images.to(self.config.device), labels.to(self.config.device)

			# Forward pass
			outputs = self.model(images)
			loss = self.criterion(outputs, labels)

			# Backward pass and optimization
			self.optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()

			if (i + 1) % 100 == 0:
				print(
					f'Epoch [{epoch + 1}/{self.config.num_epochs}], Step [{i + 1}/{total_steps}], Loss: {loss.item():.4f}',
					flush=True
				)

	def _evaluate(self):
		self.model.eval()
		with torch.no_grad():
			correct = 0
			total = 0
			for images, labels in self.test_loader:
				images, labels = images.to(self.config.device), labels.to(self.config.device)
				outputs = self.model(images)
				_, predicted = torch.max(outputs.data, 1)
				total += labels.size(0)
				correct += (predicted == labels).sum().item()

		accuracy = 100 * correct / total
		print(f'\nTest Accuracy: {accuracy:.2f} %', flush=True)
		print(f'Final Learning Rate: {self.scheduler.get_last_lr()[0]:.6f}', flush=True)

	def run(self):
		print(f"Training on {self.config.device}...", flush=True)
		for epoch in range(self.config.num_epochs):
			self._train_epoch(epoch)
			self.scheduler.step()  # 에포크마다 스케줄러 업데이트

		self._evaluate()


# 5. 실행 (Main Execution)
if __name__ == "__main__":
	config = TrainingConfig()
	train_loader, test_loader = get_data_loaders(config.batch_size)
	model = ComplexCNN(config)
	trainer = Trainer(model, train_loader, test_loader, config)
	trainer.run()