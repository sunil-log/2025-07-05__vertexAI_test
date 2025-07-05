# src/01_optimize_hpo.py (신규 파일)

import optuna
import torch

# 기존 모듈 임포트
from mnist.config import TrainingConfig
from mnist.data_loader import get_data_loaders
from mnist.model import ComplexCNN
from mnist.trainer import Trainer  # 수정된 Trainer 클래스를 임포트

def objective(trial: optuna.trial.Trial) -> float:
	"""Optuna HPO를 위한 objective 함수"""

	# 1. Hyperparameter 탐색 공간 정의
	config = TrainingConfig()
	config.learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)
	config.weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
	config.dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)

	# Optimizer 종류도 HPO 대상으로 추가 가능
	optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop"])

	# 2. 데이터 로더, 모델, 트레이너 인스턴스 생성
	train_loader, test_loader = get_data_loaders(config.batch_size)
	model = ComplexCNN(config)

	# Trainer를 생성하되, optimizer는 HPO 결과에 따라 동적으로 설정
	trainer = Trainer(model, train_loader, test_loader, config)

	# 제안된 optimizer로 교체
	if optimizer_name == "Adam":
		trainer.optimizer = torch.optim.Adam(
			model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
		)
	elif optimizer_name == "RMSprop":
		trainer.optimizer = torch.optim.RMSprop(
			model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
		)

	# Scheduler도 optimizer 변경에 맞춰 다시 설정
	trainer.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(trainer.optimizer, T_max=config.num_epochs)

	# 3. 모델 학습 및 평가
	# 01_train.py에서 수정한 run 메서드는 최종 accuracy를 반환함
	accuracy = trainer.run()

	# 4. 성능 지표 반환
	return accuracy


if __name__ == "__main__":
	# Optuna study 생성 및 최적화 실행
	study = optuna.create_study(direction="maximize")
	study.optimize(objective, n_trials=50)  # 예시로 50회 trial 실행

	# 최적화 결과 출력
	print("Number of finished trials: ", len(study.trials))
	print("Best trial:")
	trial = study.best_trial

	print("  Value: ", trial.value)
	print("  Params: ")
	for key, value in trial.params.items():
		print(f"    {key}: {value}")