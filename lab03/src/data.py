import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def load_data(data_dir="data"):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])

    train_full = datasets.FashionMNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )

    test_data = datasets.FashionMNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )

    train_data, val_data = random_split(
        train_full,
        [55000, 5000],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_data, batch_size=128, shuffle=True
    )
    val_loader = DataLoader(
        val_data, batch_size=128, shuffle=False
    )
    test_loader = DataLoader(
        test_data, batch_size=128, shuffle=False
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train, val, test = load_data()
    print("Training images:", len(train.dataset))
    print("Validation images:", len(val.dataset))
    print("Test images:", len(test.dataset))