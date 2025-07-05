# src/01_train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# 분리된 모듈 임포트
from mnist.config import TrainingConfig
from mnist.data_loader import get_data_loaders
from mnist.model import ComplexCNN

# 1. 설정 (Configuration) -> 삭제 (config.py로 이동)
# 2. 데이터 로더 (Data Loaders) -> 삭제 (data_loader.py로 이동)
# 3. 모델 정의 (Model Definition) -> 삭제 (model.py로 이동)

# 4. 트레이너 클래스 (Trainer Class)
# 학습 및 평가 로직은 그대로 유지한다.
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
            weight_decay=config.weight_decay
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config.num_epochs)

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
            self.scheduler.step()
        self._evaluate()


# 5. 실행 (Main Execution)
# 분리된 컴포넌트들을 조립하여 실행한다.
if __name__ == "__main__":
    config = TrainingConfig()
    train_loader, test_loader = get_data_loaders(config.batch_size)
    model = ComplexCNN(config)
    trainer = Trainer(model, train_loader, test_loader, config)
    trainer.run()