# src/data_loader.py
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data_loaders(batch_size: int):
    """
    Fashion-MNIST 데이터셋을 위한 train_loader와 test_loader를 생성하고 반환한다.
    출처: https://pytorch.org/vision/main/generated/torchvision.datasets.FashionMNIST.html
    """
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