# src/01_optimize_hpo.py

import optuna
import torch
import os  # os 모듈 임포트

# 기존 모듈 임포트
from mnist.config import TrainingConfig
from mnist.data_loader import get_data_loaders
from mnist.model import ComplexCNN
from mnist.trainer import Trainer


def objective(trial: optuna.trial.Trial) -> float:
	"""Optuna HPO를 위한 objective 함수 (Loss 최소화 목표)"""

	# 1. Hyperparameter 탐색 공간 정의
	config = TrainingConfig()
	config.learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)
	config.weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
	config.dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
	config.optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop"])

	# 2. 데이터 로더, 모델, 트레이너 인스턴스 생성
	# Trainer가 내부적으로 optimizer와 scheduler를 설정하므로 코드가 간결해짐
	train_loader, test_loader = get_data_loaders(config.batch_size)
	model = ComplexCNN(config)
	trainer = Trainer(model, train_loader, test_loader, config)

	# 3. 모델 학습 및 평가
	# 수정한 run 메서드는 최종 test loss를 반환함
	loss = trainer.run()

	# 4. 성능 지표(loss) 반환
	return loss


if __name__ == "__main__":
	# --- 수정된 부분 시작 ---
	# 결과 저장을 위한 디렉터리 설정
	# 실행 위치가 /sac/src이므로, 상위 디렉터리의 result 폴더를 의미함
	result_dir = "result"
	os.makedirs(result_dir, exist_ok=True)

	# Optuna study 생성 시 DB 저장 경로 지정
	study_name = "mnist-hpo-study"
	storage_path = os.path.join(result_dir, f"{study_name}.db")
	storage_name = f"sqlite:///{storage_path}"
	# --- 수정된 부분 끝 ---

	study = optuna.create_study(
		study_name=study_name,
		storage=storage_name,
		direction="minimize",  # 목표: loss 최소화
		load_if_exists=True  # 동일 이름의 study가 있으면 이어서 실행
	)

	study.optimize(objective, n_trials=50)

	# 최적화 결과 출력
	print("\nOptimization Finished!")
	print(f"Study results are saved in: {storage_path}")
	print("Number of finished trials: ", len(study.trials))
	print("Best trial:")
	trial = study.best_trial

	print(f"  Value (loss): {trial.value:.5f}")
	print("  Params: ")
	for key, value in trial.params.items():
		print(f"    {key}: {value}")