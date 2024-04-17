import torch
from torch import nn
from torch.utils import data
import pandas as pd
import seaborn as sb
import sklearn.model_selection
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from matplotlib import pyplot as plt
import os
import numpy as np

from util.optimization_data import SingleRunData

__all__ = ["data_generator", "LSTM", "get_data_loaders", "nn_train", "nn_test"]


class data_generator(torch.utils.data.Dataset):
    def __init__(self, data_path_list, labels) -> None:
        super().__init__()
        self.data_path_list = data_path_list
        self.labels = labels

    def __len__(self):
        return len(self.data_path_list)

    def __getitem__(self, index):
        run = SingleRunData.import_from_json(self.data_path_list[index])
        pop_metrics = run.get_pop_diversity_metrics_values(normalize=True).to_numpy()
        indiv_metrics = run.get_indiv_diversity_metrics_values(
            normalize=True
        ).to_numpy()

        pca = PCA(n_components=indiv_metrics.shape[1])
        principal_components = pca.fit_transform(indiv_metrics).flatten()
        variance = pca.explained_variance_ratio_

        return (
            torch.from_numpy(pop_metrics).float(),
            torch.from_numpy(principal_components).float(),
            torch.tensor(self.labels.index(run.algorithm_name[0])),
        )


class LSTM(nn.Module):
    def __init__(
        self, input_dim, aux_input_dim, num_labels, hidden_dim=256, num_layers=3
    ) -> None:
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.8,
        )
        self.fc = nn.Linear(hidden_dim + aux_input_dim, num_labels)

    def forward(self, x, aux):
        lstm_out, (h_n, c_n) = self.lstm(x)
        features_0 = lstm_out[:, -1]
        features = torch.concat([features_0, aux], dim=1)
        out = self.fc(features)

        return features, out


def get_data_loaders(
    dataset_path,
    batch_size,
    val_size=0.2,
    test_size=0.2,
    problems: list[str] = None,
    random_state=None,
):
    r"""Get dataloaders for NN training, validation and testing.

    Args:
        dataset_path (str): Path to the root folder containing optimization data.
        batch_size (int): Batch size of the data loaders.
        val_size (Optional[float]): Proportion of the dataset used for validation.
        test_size (Optional[float]): Proportion of the dataset used for testing.
        problems (Optional[list[str]]): Names of the optimization problems to include in the dataset of the loaders. Includes all if not provided.
        random_state (Optional[int]): Random seed for dataset shuffle, provide for reproducible results.

    Returns:
        Dataloader: train data loader
        Dataloader: validation data loader
        Dataloader: test data loader
        Array: class labels
    """
    dataset_paths = []
    labels = []
    for algorithm in os.listdir(dataset_path):
        labels.append(algorithm)
        for problem in os.listdir(os.path.join(dataset_path, algorithm)):
            if problems is not None and not problem in problems:
                continue
            for run in os.listdir(os.path.join(dataset_path, algorithm, problem)):
                dataset_paths.append(
                    os.path.join(dataset_path, algorithm, problem, run)
                )

    if len(dataset_paths) == 0:
        raise ValueError(
            "Provided combination of parameters resulted in an empty dataset. Check your `dataset_path` and `problems`."
        )

    x_train, x_test = sklearn.model_selection.train_test_split(
        dataset_paths, test_size=test_size, shuffle=True, random_state=random_state
    )
    x_train, x_val = sklearn.model_selection.train_test_split(
        x_train,
        test_size=val_size / (1.0 - test_size),
        shuffle=True,
        random_state=random_state,
    )

    train_dataset = data_generator(x_train, labels)
    train_data_loader = data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=os.cpu_count(),
    )

    val_dataset = data_generator(x_val, labels)
    val_data_loader = data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=os.cpu_count(),
    )

    test_dataset = data_generator(x_test, labels)
    test_data_loader = data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=True,
        pin_memory=True,
        num_workers=os.cpu_count(),
    )

    return train_data_loader, val_data_loader, test_data_loader, labels


def nn_train(
    model,
    train_data_loader,
    val_data_loader,
    epochs,
    loss_fn,
    optimizer,
    device,
    model_file_name,
    patience=10,
    verbal=False,
):
    loss_values = []
    val_loss_values = []
    acc_values = []
    val_acc_values = []
    best_acc = 0.0
    trial_counter = 0

    for epoch in range(epochs):
        _loss_values = []
        _val_loss_values = []
        _acc_values = []
        _val_acc_values = []

        model.train()
        for batch in train_data_loader:
            pop_features, indiv_features, target = batch

            target = target.to(device)
            pop_features = pop_features.to(device)
            indiv_features = indiv_features.to(device)

            _, pred = model(pop_features, indiv_features)
            loss = loss_fn(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            y_pred = torch.argmax(pred, dim=1).cpu().numpy()
            y_target = target.cpu().numpy()

            _acc_values.append(accuracy_score(y_target, y_pred))
            _loss_values.append(loss.item())

        model.eval()
        with torch.no_grad():
            for batch in val_data_loader:
                pop_features, indiv_features, target = batch

                target = target.to(device)
                pop_features = pop_features.to(device)
                indiv_features = indiv_features.to(device)

                _, pred = model(pop_features, indiv_features)
                loss = loss_fn(pred, target)

                y_pred = torch.argmax(pred, dim=1).cpu().numpy()
                y_target = target.cpu().numpy()

                _val_acc_values.append(accuracy_score(y_target, y_pred))
                _val_loss_values.append(loss.item())

        acc_values.append(np.mean(_acc_values))
        val_acc_values.append(np.mean(_val_acc_values))
        loss_values.append(np.mean(_loss_values))
        val_loss_values.append(np.mean(_val_loss_values))

        if verbal:
            print(
                f"epoch: {epoch + 1}, loss: {loss_values[-1] :.10f}, val_loss: {val_loss_values[-1] :.10f}, acc: {acc_values[-1] :.10f}, val_acc: {val_acc_values[-1] :.10f}"
            )

        if val_acc_values[-1] > best_acc:
            trial_counter = 0
            best_acc = val_acc_values[-1]
            torch.save(model, model_file_name)
            if verbal:
                print(f"Saving model with accuracy: {best_acc :.10f}")
        else:
            trial_counter += 1
            if trial_counter >= patience:
                if verbal:
                    print(f"Early stopping after {epoch + 1} epochs")
                break

    if verbal:
        x = [*range(1, len(loss_values) + 1)]
        plt.plot(x, loss_values, label="train loss")
        plt.plot(x, val_loss_values, label="val loss")
        plt.legend()
        plt.show()
        plt.plot(x, acc_values, label="train acc")
        plt.plot(x, val_acc_values, label="val acc")
        plt.legend()
        plt.show()


def nn_test(
    model, test_data_loader, device, labels=None, show_classification_report=False
):
    model.eval()

    y_pred = []
    y_target = []

    for batch in test_data_loader:
        pop_features, indiv_features, target = batch

        target = target.to(device)
        pop_features = pop_features.to(device)
        indiv_features = indiv_features.to(device)

        with torch.no_grad():
            _, pred = model(pop_features, indiv_features)

            y_pred.append(torch.argmax(pred).cpu().numpy())
            y_target.append(target.cpu().numpy()[0])

    if labels is not None:
        cf_matrix = confusion_matrix(y_target, y_pred)
        df_cm = pd.DataFrame(
            cf_matrix, index=[i for i in labels], columns=[i for i in labels]
        )
        plt.figure(figsize=(12, 7))
        sb.heatmap(df_cm, annot=True)

    if show_classification_report:
        print(classification_report(y_target, y_pred))

    return accuracy_score(y_target, y_pred)
