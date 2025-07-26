from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    KFold,
)
from sklearn import svm
from matplotlib import pyplot as plt
import os
import numpy as np

from msa.tools.optimization_data import SingleRunData

__all__ = ["svm_and_knn_classification"]


def svm_and_knn_classification(
    dataset_path: str,
    repetitions: int,
    bar_chart_filename: str | None = None,
    box_plot_filename: str | None = None,
):
    r"""Evaluate similarity of metaheuristics with SVM and KNN classifiers based on feature vectors.
    Based on assumption should models perform worse when distinguishing metaheuristics with higher similarity.
    To maximize metaheuristics similarity metric `1-accuracy` is used as similarity metric.

    Args:
        dataset_path (str): Path to the root folder containing optimization data arranged into subsets of comparisons
            of MetaheuristicSimilarityAnalyzer.
        repetitions (int): Number of training repetitions to get the average 1-accuracy score from.
        bar_chart_filename (Optional[str]): Filename of the bar charts showing metric 1-accuracy values of the models
            per MetaheuristicSimilarityAnalyzer comparison.
        box_plot_filename (Optional[str]): Filename of the box plot showing metric 1-accuracy values of the models for
            all MetaheuristicSimilarityAnalyzer comparisons.

    Returns:
        1-accuracy scores (dict[str, numpy.ndarray[float]]): Dictionary containing 1-accuracy scores for train and test
            subsets of both models.
    """
    alg_1_label = ""
    alg_2_label = ""
    _k_svm_scores = []
    _knn_scores = []
    subsets = os.listdir(dataset_path)

    for idx in range(len(subsets)):
        subset = f"{idx}_subset"
        subset_k_svm_scores = []
        subset_knn_scores = []
        feature_vectors = []
        actual_labels = []
        for idx, algorithm in enumerate(os.listdir(os.path.join(dataset_path, subset))):
            if idx == 0 and alg_1_label == "":
                alg_1_label = algorithm
            elif idx == 1 and alg_2_label == "":
                alg_2_label = algorithm

            for problem in os.listdir(os.path.join(dataset_path, subset, algorithm)):
                runs = os.listdir(os.path.join(dataset_path, subset, algorithm, problem))
                runs.sort()
                for run in runs:
                    run_path = os.path.join(dataset_path, subset, algorithm, problem, run)
                    srd = SingleRunData.import_from_json(run_path)
                    feature_vector = srd.get_feature_vector(standard_scale=True)
                    feature_vectors.append(feature_vector)
                    actual_labels.append(idx)

        for _ in range(repetitions):
            # train test split
            X_train, X_test, y_train, y_test = train_test_split(
                feature_vectors, actual_labels, test_size=0.2, shuffle=True
            )

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # K-SVM classifier
            # define the parameter grid
            param_grid = {
                "C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                "gamma": [0.0001, 0.001, 0.01, 1, 10, 100, 1000],
            }

            k_svm = svm.SVC(kernel="rbf")
            kf = KFold(n_splits=5, shuffle=True, random_state=None)

            # perform grid search
            grid_search = GridSearchCV(k_svm, param_grid, cv=kf, n_jobs=-1)
            grid_search.fit(X_train, y_train)

            _C = grid_search.best_params_.get("C")
            _gamma = grid_search.best_params_.get("gamma")
            k_svm = svm.SVC(
                kernel="rbf",
                C=0 if _C is None else _C,
                gamma=0 if _gamma is None else _gamma,
            )
            k_svm.fit(X_train, y_train)
            svm_training_score = k_svm.score(X_train, y_train)
            svm_test_score = k_svm.score(X_test, y_test)
            tmp = []
            tmp.append(1.0 - svm_training_score)
            tmp.append(1.0 - svm_test_score)
            subset_k_svm_scores.append(tmp)

            # kNN classifier
            # define the parameter grid
            param_grid = {"n_neighbors": np.arange(1, min(50, round(len(y_train) / 3))).tolist()}

            knn = KNeighborsClassifier()
            kf = KFold(n_splits=5, shuffle=True, random_state=None)

            # perform grid search
            grid_search = GridSearchCV(knn, param_grid, cv=kf, n_jobs=-1)
            grid_search.fit(X_train, y_train)

            _n_neighbors = grid_search.best_params_.get("n_neighbors")
            if _n_neighbors is None:
                _n_neighbors = 0
            knn = KNeighborsClassifier(n_neighbors=_n_neighbors)
            knn.fit(X_train, y_train)
            knn_training_score = knn.score(X_train, y_train)
            knn_test_score = knn.score(X_test, y_test)
            tmp = []
            tmp.append(1.0 - knn_training_score)
            tmp.append(1.0 - knn_test_score)
            subset_knn_scores.append(tmp)

        _k_svm_scores.append(np.mean(subset_k_svm_scores, axis=0))
        _knn_scores.append(np.mean(subset_knn_scores, axis=0))

    k_svm_scores = np.array(_k_svm_scores)
    knn_scores = np.array(_knn_scores)

    bar_width = 0.35
    (
        fig,
        ax,
    ) = plt.subplots(2, 1, figsize=(15, 10))
    fig.subplots_adjust(hspace=0.5)

    # bar charts
    if bar_chart_filename is not None:
        index = np.arange(1, len(k_svm_scores[:, 0]) + 1)
        low = np.min(k_svm_scores)
        high = np.max(k_svm_scores)
        ax[0].bar(index, k_svm_scores[:, 0], bar_width, label="train")
        ax[0].bar(index + bar_width, k_svm_scores[:, 1], bar_width, label="test")
        ax[0].set_title(f"SVM {alg_1_label} - {alg_2_label}", fontsize=22, pad=10)
        ax[0].set_xlabel("configuration", fontsize=19, labelpad=10)
        ax[0].set_ylabel("1-accuracy", fontsize=19, labelpad=10)
        ax[0].tick_params(axis="x", labelsize=19, rotation=45)
        ax[0].tick_params(axis="y", labelsize=19)
        ax[0].legend(fontsize=15)
        ax[0].xaxis.set_ticks(index + bar_width / 2, index)
        ax[0].set_ylim(low - 0.5 * (high - low), high + 0.5 * (high - low))
        ax[0].set_xlim(
            ax[0].patches[0].get_x() / 2,
            ax[0].patches[-1].get_x() + ax[0].patches[-1].get_width() * 2,
        )
        ax[0].grid(axis="y", color="gray", linestyle="--", linewidth=0.7)
        ax[0].set_axisbelow(True)

        low = np.min(knn_scores)
        high = np.max(knn_scores)
        ax[1].bar(index, knn_scores[:, 0], bar_width, label="train")
        ax[1].bar(index + bar_width, knn_scores[:, 1], bar_width, label="test")
        ax[1].set_title(f"KNN {alg_1_label} - {alg_2_label}", fontsize=22, pad=10)
        ax[1].set_xlabel("configuration", fontsize=19, labelpad=10)
        ax[1].set_ylabel("1-accuracy", fontsize=19, labelpad=10)
        ax[1].tick_params(axis="x", labelsize=19, rotation=45)
        ax[1].tick_params(axis="y", labelsize=19)
        ax[1].legend(fontsize=15)
        ax[1].xaxis.set_ticks(index + bar_width / 2, index)
        ax[1].set_ylim(low - 0.5 * (high - low), high + 0.5 * (high - low))
        ax[1].set_xlim(
            ax[1].patches[0].get_x() / 2,
            ax[1].patches[-1].get_x() + ax[1].patches[-1].get_width() * 2,
        )
        ax[1].grid(axis="y", color="gray", linestyle="--", linewidth=0.7)
        ax[1].set_axisbelow(True)

        fig.savefig(bar_chart_filename, bbox_inches="tight")

    # box plots
    if box_plot_filename is not None:
        (
            fig,
            ax,
        ) = plt.subplots(1, 2, figsize=(15, 5))
        fig.subplots_adjust(wspace=0.3)
        labels = ["train", "test"]

        ax[0].boxplot(k_svm_scores)
        ax[0].set_xticks(ticks=np.arange(1, len(labels) + 1), labels=labels)
        ax[0].tick_params(axis="both", labelsize=19)
        ax[0].set_title(f"SVM  {alg_1_label} - {alg_2_label}", fontsize=22, pad=15)
        ax[0].tick_params(axis="both", labelsize=19)
        ax[0].set_ylabel("1-accuracy", fontsize=19, labelpad=10)

        ax[1].boxplot(knn_scores)
        ax[1].set_xticks(ticks=np.arange(1, len(labels) + 1), labels=labels)
        ax[1].tick_params(axis="both", labelsize=19)
        ax[1].set_title(f"KNN  {alg_1_label} - {alg_2_label}", fontsize=22, pad=15)
        ax[1].tick_params(axis="both", labelsize=19)
        ax[1].set_ylabel("1-accuracy", fontsize=19, labelpad=10)

        fig.savefig(box_plot_filename, bbox_inches="tight")

    accuracy = {
        "svm_train": np.round(k_svm_scores.transpose()[0], 2).tolist(),
        "svm_test": np.round(k_svm_scores.transpose()[1], 2).tolist(),
        "knn_train": np.round(knn_scores.transpose()[0], 2).tolist(),
        "knn_test": np.round(knn_scores.transpose()[1], 2).tolist(),
    }

    return accuracy
