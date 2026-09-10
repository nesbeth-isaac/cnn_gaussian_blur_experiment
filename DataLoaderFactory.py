from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch, torch.utils.data

# Function that accepts seperated data and labels and produces and outputs dataloaders for training, devset
# and test set

def produce_dataloader(dataset, labelset):
    """
        Splits features and labels into training (70%), dev/validation (15%), and test (15%) sets,
        normalizes the feature data using MinMaxScaler, converts arrays to PyTorch Tensors,
        and wraps them inside PyTorch DataLoaders.

        Parameters:
        - dataset (array-like / DataFrame): Feature dataset to be split and scaled.
        - labelset (array-like / Series): Corresponding target labels.

        Returns:
        - train_loader (DataLoader): Batched and shuffled PyTorch DataLoader for training.
        - dev_test_loader (DataLoader): Single-batch DataLoader for validation/tuning.
        - final_test_loader (DataLoader): Single-batch DataLoader for final testing.
        """

    print("Splitting Data")

    # Splitting into training data and test data covering dev testing and final testing data.
    train_data, all_test_data, train_labels, all_test_labels = train_test_split(dataset, labelset, test_size=0.3,
                                                                                shuffle=True)

    # Split testing into dev data/labels and test data/labels.
    dev_test_data, final_test_data, dev_test_labels, final_test_labels = train_test_split(all_test_data,
                                                                                          all_test_labels,
                                                                                          test_size=0.5)

    scaler = MinMaxScaler()

    # train_data = scaler.fit_transform(train_data)
    # dev_test_data = scaler.transform(dev_test_data)
    # final_test_data = scaler.transform(final_test_data)

    train_data = torch.Tensor(train_data)
    dev_test_data = torch.Tensor(dev_test_data)
    final_test_data = torch.Tensor(final_test_data)

    train_dataset = TensorDataset(train_data, train_labels)
    dev_test_dataset = TensorDataset(dev_test_data, dev_test_labels)
    final_test_dataset = TensorDataset(final_test_data, final_test_labels)

    train_loader = DataLoader(train_dataset, batch_size=64, drop_last=True, shuffle=True)
    dev_test_loader = DataLoader(dev_test_dataset, batch_size=len(dev_test_dataset))
    final_test_loader = DataLoader(final_test_dataset, batch_size=len(final_test_dataset))

    return train_loader, dev_test_loader, final_test_loader

