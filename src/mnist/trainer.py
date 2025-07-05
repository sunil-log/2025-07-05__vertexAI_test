# src/trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict

# 분리된 모듈 임포트
from mnist.config import TrainingConfig
from mnist.data_loader import get_data_loaders
from mnist.model import ComplexCNN


class Trainer:
	def __init__(self, model: nn.Module, train_loader: DataLoader, test_loader: DataLoader, config: TrainingConfig):
		self.model = model.to(config.device)
		self.train_loader = train_loader
		self.test_loader = test_loader
		self.config = config
		self.criterion = nn.CrossEntropyLoss()

		# config에 따라 Optimizer 동적 생성
		optimizer_class = getattr(optim, config.optimizer_name, None)
		if optimizer_class is None:
			raise ValueError(f"Optimizer {config.optimizer_name} is not supported.")

		self.optimizer = optimizer_class(
			self.model.parameters(),
			lr=config.learning_rate,
			weight_decay=config.weight_decay
		)

		# 스케줄러를 ReduceLROnPlateau로 변경
		self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=3, verbose=True)

	def _train_epoch(self, epoch: int):
		self.model.train()
		total_steps = len(self.train_loader)
		for i, (images, labels) in enumerate(self.train_loader):
			images, labels = images.to(self.config.device), labels.to(self.config.device)
			outputs = self.model(images)
			loss = self.criterion(outputs, labels)
			self.optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()
			if (i + 1) % 100 == 0:
				print(
					f'Epoch [{epoch + 1}/{self.config.num_epochs}], Step [{i + 1}/{total_steps}], Loss: {loss.item():.4f}',
					flush=True
				)

	def _evaluate(self) -> Dict[str, float]:  # 반환 타입을 딕셔너리로 변경
		self.model.eval()
		total_loss = 0
		correct = 0
		total = 0
		with torch.no_grad():
			for images, labels in self.test_loader:
				images, labels = images.to(self.config.device), labels.to(self.config.device)
				outputs = self.model(images)
				loss = self.criterion(outputs, labels)
				total_loss += loss.item()
				_, predicted = torch.max(outputs.data, 1)
				total += labels.size(0)
				correct += (predicted == labels).sum().item()

		accuracy = 100 * correct / total
		avg_loss = total_loss / len(self.test_loader)

		print(f'\nTest Accuracy: {accuracy:.2f} %, Test Avg Loss: {avg_loss:.4f}', flush=True)
		print(f'Current Learning Rate: {self.optimizer.param_groups[0]["lr"]:.6f}', flush=True)

		return {'accuracy': accuracy, 'loss': avg_loss}  # accuracy와 loss 모두 반환

	def run(self) -> float:  # 최종 반환 타입은 loss(float)
		print(f"Training on {self.config.device}...", flush=True)
		final_metrics = {}
		for epoch in range(self.config.num_epochs):
			self._train_epoch(epoch)
			metrics = self._evaluate()
			self.scheduler.step(metrics['loss'])  # loss를 기준으로 scheduler step

		print("Training finished.")
		return metrics['loss']  # HPO를 위해 최종 loss 반환


# 실행부 (수정 없음)
if __name__ == "__main__":
	config = TrainingConfig()
	train_loader, test_loader = get_data_loaders(config.batch_size)
	model = ComplexCNN(config)
	trainer = Trainer(model, train_loader, test_loader, config)
	trainer.run()