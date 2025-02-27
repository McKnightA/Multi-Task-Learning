import time
from typing import Dict, List, Tuple, Optional, Any
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import Module
from torch.optim import Optimizer
from scipy import stats


class TaskMetrics:
    """Container for tracking training and validation losses."""

    def __init__(self):
        self.name: str
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []


def train_and_evaluate_task(
        training_set: Any,
        validation_set: Any,
        backbone: Module,
        task: Module,
        num_epochs: int,
        batch_size: int,
        use_frozen_backbone: bool = False,
        visualize: bool = True,
        output_dir: str = "plots"
) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
    """
    Train and evaluate a task with performance tracking and visualization.

    This function handles the complete training loop including:
    - Model parameter optimization
    - Loss computation for training and validation
    - Performance metric tracking
    - Visualization of training progress

    Args:
        training_set: Dataset module containing training data
        validation_set: Dataset module containing validation data
        backbone: Feature extraction backbone network
        task: Task module implementing forward() and check_performance()
        num_epochs: Number of training epochs
        batch_size: Number of samples per batch
        use_frozen_backbone: If True, only train task parameters, keeping backbone frozen
        visualize: If True, save visualizations of training metrics
        output_dir: Directory to save performance plots

    Returns:
        Tuple containing:
        - performance_metrics: Dict mapping metric names to lists of training values
        - validation_metrics: Dict mapping metric names to lists of validation values
    """
    # Initialize optimizer with appropriate parameters
    trainable_params = (
        list(task.parameters()) if use_frozen_backbone
        else list(task.parameters()) + list(backbone.parameters())
    )
    optimizer = torch.optim.Adam(trainable_params)

    # Initialize tracking containers
    training_losses: List[float] = []
    validation_losses: List[float] = []
    performance_metrics: Dict[str, List[float]] = {}
    validation_metrics: Dict[str, List[float]] = {}

    # Training loop
    for epoch in range(num_epochs):
        print(f"Epoch: {epoch + 1}/{num_epochs}")

        # Training phase
        train_loss = _run_epoch(
            dataset=training_set,
            task=task,
            backbone=backbone,
            optimizer=optimizer,
            batch_size=batch_size,
            is_training=True
        )
        training_losses.append(train_loss)

        # Validation phase
        val_loss = _run_epoch(
            dataset=validation_set,
            task=task,
            backbone=backbone,
            optimizer=optimizer,
            batch_size=batch_size,
            is_training=False
        )
        validation_losses.append(val_loss)

        # Performance evaluation
        train_perf, val_perf = _evaluate_performance(
            training_set=training_set,
            validation_set=validation_set,
            task=task,
            backbone=backbone,
            batch_size=batch_size
        )

        # Update performance tracking
        for metric, value in train_perf.items():
            if metric not in performance_metrics:
                performance_metrics[metric] = []
            performance_metrics[metric].append(value)

        for metric, value in val_perf.items():
            if metric not in validation_metrics:
                validation_metrics[metric] = []
            validation_metrics[metric].append(value)

    if visualize:
        # Generate visualizations
        _plot_training_curves(
            training_losses=training_losses,
            validation_losses=validation_losses,
            task_name=task.name,
            is_baseline=use_frozen_backbone,
            output_dir=output_dir
        )

        _plot_metric_curves(
            performance_metrics=performance_metrics,
            validation_metrics=validation_metrics,
            task_name=task.name,
            is_baseline=use_frozen_backbone,
            output_dir=output_dir
        )

    performance_metrics["loss"] = training_losses
    validation_metrics["loss"] = validation_losses

    return performance_metrics, validation_metrics


def pretrain_and_finetune(
        pretraining_set: Any,
        finetuning_set: Any,
        backbone: Module,
        pretrain_task: Module,
        finetune_task: Module,
        pretrain_epochs: int,
        finetune_epochs: int,
        batch_size: int,
        output_dir: str = "plots"
) -> List[List[float]]:
    """
    Implement two-stage training: pretraining followed by fine-tuning.

    This function follows the training approach described in:
    https://snorkel.ai/boost-foundation-model-results-with-linear-probing-fine-tuning/

    The process consists of:
    1. Pretraining the backbone network with some task
    2. Fine-tuning both backbone and downstream task for the target objective

    Args:
        pretraining_set: Dataset for pretraining phase
        finetuning_set: Dataset for fine-tuning phase
        backbone: Feature extraction backbone network
        pretrain_task: Pretraining task
        finetune_task: Downstream task for fine-tuning
        pretrain_epochs: Number of pretraining epochs
        finetune_epochs: Number of fine-tuning epochs
        batch_size: Number of samples per batch
        output_dir: Directory to save performance plots

    Returns:
        List containing training history:
        [pretrain_losses, pretrain_val_losses, finetune_losses, finetune_val_losses]
    """
    # Pretraining phase
    pretrain_metrics = _run_pretraining(
        training_set=pretraining_set,
        backbone=backbone,
        pretrain_task=pretrain_task,
        num_epochs=pretrain_epochs,
        batch_size=batch_size
    )

    # Fine-tuning phase
    finetune_metrics = _run_finetuning(
        training_set=finetuning_set,
        backbone=backbone,
        finetune_task=finetune_task,
        num_epochs=finetune_epochs,
        batch_size=batch_size
    )

    # Visualize training progress
    _plot_training_phases(
        pretrain_metrics=pretrain_metrics,
        finetune_metrics=finetune_metrics,
        output_dir=output_dir
    )

    # Combine all metrics for return
    return [
        pretrain_metrics.train_losses,
        pretrain_metrics.val_losses,
        finetune_metrics.train_losses,
        finetune_metrics.val_losses
    ]


def evaluate_with_probing(
        pretraining_set: Any,
        probe_datasets: List[Any],
        backbone: Module,
        pretrain_task: Module,
        probe_tasks: List[Module],
        pretrain_epochs: int,
        probe_epochs: int,
        batch_size: int,
        output_dir: str = "plots"
) -> Dict[str, List[float]]:
    """
    Evaluate feature extraction capabilities using probing.

    This function implements a two-stage evaluation process:
    1. Pretraining the backbone network with a pretraining task
    2. Evaluating the frozen backbone by training probe tasks

    The probing stage keeps the backbone frozen and only trains
    the probe task heads, testing how well the backbone's features
    generalize to different tasks.

    Args:
        pretraining_set: Dataset for pretraining phase
        probe_datasets: List of datasets for probe tasks evaluation
        backbone: Feature extraction backbone network
        pretrain_task: Pretraining task
        probe_tasks: List of probe tasks for evaluation
        pretrain_epochs: Number of pretraining epochs
        probe_epochs: Number of epochs for each probe task
        batch_size: Number of samples per batch
        output_dir: Directory to save performance plots

    Returns:
        Dictionary containing all training metrics:
        - 'pretrain': List of pretraining losses
        - '{probe_task.name}': List of probe task losses
    """
    if len(probe_datasets) != len(probe_tasks):
        raise ValueError(
            f"Number of probe datasets ({len(probe_datasets)}) must match "
            f"number of probe tasks ({len(probe_tasks)})"
        )

    # Run pretraining phase
    pretrain_metrics = _run_pretraining(
        training_set=pretraining_set,
        backbone=backbone,
        pretrain_task=pretrain_task,
        num_epochs=pretrain_epochs,
        batch_size=batch_size
    )

    # Plot pretraining results
    _plot_training_curves(
        training_losses=pretrain_metrics.train_losses,
        validation_losses=pretrain_metrics.val_losses,
        task_name=f"{pretrain_task.name} pretrain training loss",
        is_baseline=False,
        output_dir=output_dir
    )

    # Run probing phase for each task
    probe_metrics = _run_probing(
        probe_datasets=probe_datasets,
        probe_tasks=probe_tasks,
        backbone=backbone,
        num_epochs=probe_epochs,
        batch_size=batch_size,
        pretrain_task_name=pretrain_task.name,
        output_dir=output_dir
    )

    # Combine all results
    all_metrics = {
        'pretrain': pretrain_metrics.train_losses + pretrain_metrics.val_losses
    }
    for probe_metric in probe_metrics:
        all_metrics[probe_metric.name] = (
                probe_metric.train_losses + probe_metric.val_losses
        )

    return all_metrics


def run_statistical_trials(
        training_set: Any,
        validation_set: Any,
        backbone_factory: callable,
        task_factory: callable,
        num_trials: int,
        num_epochs: int,
        batch_size: int,
        use_frozen_backbone: bool = False,
        confidence_level: float = 0.85,
        output_dir: str = "plots"
) -> Dict[str, Dict[str, Any]]:
    """
    Run multiple training trials to compute statistical metrics and confidence intervals.

    This function performs multiple training runs to gather statistical information about:
    - Training and validation loss distributions
    - Performance metric distributions
    - Confidence intervals for learning curves

    Args:
        training_set: Dataset module containing training data
        validation_set: Dataset module containing validation data
        backbone_factory: Function that creates a new backbone network instance
        task_factory: Function that creates a new task module instance
        num_trials: Number of training trials to run
        num_epochs: Number of training epochs per trial
        batch_size: Number of samples per batch
        use_frozen_backbone: If True, only train task parameters, keeping backbone frozen
        confidence_level: Confidence level for interval calculations (0-1)
        output_dir: Directory to save performance plots

    Returns:
        Dictionary containing statistical metrics:
        - 'losses': Dict with training/validation loss statistics
        - 'metrics': Dict with performance metric statistics
        - 'confidence_intervals': Dict with confidence bounds for each epoch
    """
    all_train_losses: List[List[float]] = []
    all_val_losses: List[List[float]] = []
    all_performance_metrics: List[Dict[str, List[float]]] = []
    all_validation_metrics: List[Dict[str, List[float]]] = []

    for trial in range(num_trials):
        print(f"Running trial {trial + 1}/{num_trials}")

        # Create new model instances for each trial
        backbone = backbone_factory()
        task = task_factory()

        # Run training
        perf_metrics, val_metrics = train_and_evaluate_task(
            training_set=training_set,
            validation_set=validation_set,
            backbone=backbone,
            task=task,
            num_epochs=num_epochs,
            batch_size=batch_size,
            use_frozen_backbone=use_frozen_backbone,
            visualize=False,
            output_dir=output_dir
        )

        # Store results
        all_train_losses.append(perf_metrics['loss'])
        all_val_losses.append(val_metrics['loss'])
        all_performance_metrics.append(perf_metrics)
        all_validation_metrics.append(val_metrics)

    # Compute statistics
    statistics = _compute_trial_statistics(
        train_losses=all_train_losses,
        val_losses=all_val_losses,
        confidence_level=confidence_level
    )

    # Generate statistical visualizations
    _plot_statistical_curves(
        train_losses=all_train_losses,
        val_losses=all_val_losses,
        confidence_level=confidence_level,
        task_name=task.name,
        is_baseline=use_frozen_backbone,
        output_dir=output_dir
    )

    return statistics


def _run_epoch(
        dataset: Module,
        task: Module,
        backbone: Module,
        optimizer: Optional[Optimizer],
        batch_size: int,
        is_training: bool
) -> float:
    """
    Run one epoch of training or validation.

    Args:
        dataset: Dataset to process
        task: Task module
        backbone: Feature extraction backbone
        optimizer: Optimizer for parameter updates (only used if is_training=True)
        batch_size: Batch size
        is_training: Whether to perform parameter updates

    Returns:
        Average loss for the epoch
    """
    batch_losses = []
    data_partition = dataset.train_dataset if is_training else dataset.test_dataset
    data_partition.shuffle(seed=42)

    for batch in data_partition.iter(batch_size):
        # Prepare batch data
        inputs, labels = dataset.process_batch(
            batch,
            backbone.expected_input_size
        )

        # Forward pass
        try:
            loss = task.forward(inputs, backbone)
        except TypeError:
            # Handle supervised tasks that require labels
            loss = task.forward(inputs, labels, backbone)

        batch_losses.append(loss.item())

        # Parameter updates for training
        if is_training and optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return np.mean(batch_losses)


def _evaluate_performance(
        training_set: Module,
        validation_set: Module,
        task: Module,
        backbone: Module,
        batch_size: int
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Evaluate model performance on training and validation sets.

    Args:
        training_set: Training dataset
        validation_set: Validation dataset
        task: Task module
        backbone: Feature extraction backbone
        batch_size: Batch size

    Returns:
        Tuple containing:
        - Training performance metrics
        - Validation performance metrics
    """
    train_metrics: Dict[str, List[float]] = {}
    val_metrics: Dict[str, List[float]] = {}

    # Evaluate on training set
    for batch in training_set.train_dataset.iter(batch_size):
        data, labels = training_set.process_batch(
            batch,
            backbone.expected_input_size
        )
        try:
            results = task.check_performance(data, backbone)
        except TypeError:
            results = task.check_performance(data, labels, backbone)

        for metric, value in results.items():
            if metric not in train_metrics:
                train_metrics[metric] = []
            train_metrics[metric].append(value.item())

    # Evaluate on validation set
    for batch in validation_set.test_dataset.iter(batch_size):
        data, labels = validation_set.process_batch(
            batch,
            backbone.expected_input_size
        )
        try:
            results = task.check_performance(data, backbone)
        except TypeError:
            results = task.check_performance(data, labels, backbone)

        for metric, value in results.items():
            if metric not in val_metrics:
                val_metrics[metric] = []
            val_metrics[metric].append(value.item())

    # Calculate mean for each metric
    train_results = {k: np.mean(v) for k, v in train_metrics.items()}
    val_results = {k: np.mean(v) for k, v in val_metrics.items()}

    return train_results, val_results


def _run_pretraining(
        training_set: Module,
        backbone: Module,
        pretrain_task: Module,
        num_epochs: int,
        batch_size: int
) -> TaskMetrics:
    """
    Run the pretraining phase to learn general features.

    Args:
        training_set: Dataset for pretraining
        backbone: Feature extraction backbone
        pretrain_task: Pretraining task
        num_epochs: Number of training epochs
        batch_size: Batch size

    Returns:
        TrainingMetrics containing training and validation losses
    """
    metrics = TaskMetrics()
    metrics.name = pretrain_task.name

    # Initialize optimizer with both task and backbone parameters
    optimizer = torch.optim.Adam(
        list(pretrain_task.parameters()) + list(backbone.parameters())
    )

    # Training loop
    for epoch in range(num_epochs):
        print(f"Pretraining epoch: {epoch + 1}/{num_epochs}")

        # Training phase
        train_loss = _run_epoch(
            dataset=training_set,
            task=pretrain_task,
            backbone=backbone,
            optimizer=optimizer,
            batch_size=batch_size,
            is_training=True
        )
        metrics.train_losses.append(train_loss)

        # Validation phase
        val_loss = _run_epoch(
            dataset=training_set,
            task=pretrain_task,
            backbone=backbone,
            optimizer=optimizer,
            batch_size=batch_size,
            is_training=False
        )
        metrics.val_losses.append(val_loss)

    return metrics


def _run_finetuning(
        training_set: Module,
        backbone: Module,
        finetune_task: Module,
        num_epochs: int,
        batch_size: int
) -> TaskMetrics:
    """
    Run the fine-tuning phase to adapt to the downstream task.

    Args:
        training_set: Dataset for fine-tuning
        backbone: Pretrained feature extraction backbone
        finetune_task: Downstream task for fine-tuning
        num_epochs: Number of training epochs
        batch_size: Batch size

    Returns:
        TrainingMetrics containing training and validation losses
    """
    metrics = TaskMetrics()
    metrics.name = finetune_task.name

    # Initialize optimizer for fine-tuning
    optimizer = torch.optim.Adam(
        list(finetune_task.parameters()) + list(backbone.parameters())
    )

    # Fine-tuning loop
    for epoch in range(num_epochs):
        print(f"Fine-tuning epoch: {epoch + 1}/{num_epochs}")

        # Training phase
        train_loss = _run_epoch(
            dataset=training_set,
            task=finetune_task,
            backbone=backbone,
            optimizer=optimizer,
            batch_size=batch_size,
            is_training=True
        )
        metrics.train_losses.append(train_loss)

        # Validation phase
        val_loss = _run_epoch(
            dataset=training_set,
            task=finetune_task,
            backbone=backbone,
            optimizer=optimizer,
            batch_size=batch_size,
            is_training=False
        )
        metrics.val_losses.append(val_loss)

    return metrics


def _run_probing(
        probe_datasets: List[Any],
        probe_tasks: List[Module],
        backbone: Module,
        num_epochs: int,
        batch_size: int,
        pretrain_task_name: str,
        output_dir: str
) -> List[TaskMetrics]:
    """
    Evaluate the frozen backbone using multiple probe tasks.

    Args:
        probe_datasets: List of datasets for probe tasks
        probe_tasks: List of probe task modules
        backbone: Pretrained feature extraction backbone
        num_epochs: Number of training epochs per probe task
        batch_size: Batch size
        pretrain_task_name: Name of the pretraining task (for plotting)
        output_dir: Directory to save performance plots

    Returns:
        List of ProbeTaskMetrics for each probe task
    """
    # Freeze backbone parameters
    for param in backbone.parameters():
        param.requires_grad = False

    probe_metrics = []

    # Train and evaluate each probe task
    for probe_dataset, probe_task in zip(probe_datasets, probe_tasks):
        # Initialize optimizer with only probe task parameters
        optimizer = torch.optim.Adam(probe_task.parameters())

        task_metrics = TaskMetrics()
        task_metrics.name = probe_task.name

        # Training loop for probe task
        for epoch in range(num_epochs):
            print(f"Probing with {probe_task.name} - Epoch: {epoch + 1}/{num_epochs}")

            # Training phase
            train_loss = _run_epoch(
                dataset=probe_dataset,
                task=probe_task,
                backbone=backbone,
                optimizer=optimizer,
                batch_size=batch_size,
                is_training=True
            )
            task_metrics.train_losses.append(train_loss)

            # Validation phase
            val_loss = _run_epoch(
                dataset=probe_dataset,
                task=probe_task,
                backbone=backbone,
                optimizer=optimizer,
                batch_size=batch_size,
                is_training=False
            )
            task_metrics.val_losses.append(val_loss)

        # Plot probe task results
        _plot_training_curves(
            training_losses=task_metrics.train_losses,
            validation_losses=task_metrics.val_losses,
            task_name=f"{pretrain_task_name} pretraining - probing {probe_task.name}",
            is_baseline=False,
            output_dir=output_dir
        )

        probe_metrics.append(task_metrics)

    # Unfreeze backbone parameters
    for param in backbone.parameters():
        param.requires_grad = True

    return probe_metrics


def _plot_training_curves(
        training_losses: List[float],
        validation_losses: List[float],
        task_name: str,
        is_baseline: bool,
        output_dir: str
) -> None:
    """
    Plot and save training and validation loss curves.

    Args:
        training_losses: List of training losses per epoch
        validation_losses: List of validation losses per epoch
        task_name: Name of the task being evaluated
        is_baseline: Whether this is a baseline model (frozen backbone)
        output_dir: Directory to save plots
    """
    plt.figure()
    plt.plot(training_losses, label="training")
    plt.plot(validation_losses, label="validation")
    plt.xlabel("epochs")
    plt.ylabel("averaged loss")
    plt.legend()

    title = f"{task_name} {'baseline ' if is_baseline else ''}loss"
    plt.title(title)
    plt.savefig(f"{output_dir}/{title}")
    plt.close()


def _plot_training_phases(
        pretrain_metrics: TaskMetrics,
        finetune_metrics: TaskMetrics,
        output_dir: str
) -> None:
    """
    Create and save visualization plots for both training phases.

    Args:
        pretrain_metrics: Metrics from pretraining phase
        finetune_metrics: Metrics from fine-tuning phase
        pretrain_task_name: Name of pretraining task
        finetune_task_name: Name of fine-tuning task
        output_dir: Directory to save plots
    """
    # Plot pretraining losses
    plt.figure()
    plt.plot(pretrain_metrics.train_losses, label="training")
    plt.plot(pretrain_metrics.val_losses, label="validation")
    plt.xlabel("epochs")
    plt.ylabel("averaged loss")
    plt.legend()
    plt.title(f"{pretrain_metrics.name} pretrain training loss")
    plt.savefig(f"{output_dir}/{pretrain_metrics.name} pretrain training loss.png")
    plt.close()

    # Plot fine-tuning losses
    plt.figure()
    plt.plot(finetune_metrics.train_losses, label="training")
    plt.plot(finetune_metrics.val_losses, label="validation")
    plt.xlabel("epochs")
    plt.ylabel("averaged loss")
    plt.legend()
    plt.title(f"{pretrain_metrics.name} pretrain → {finetune_metrics.name} finetune")
    plt.savefig(
        f"{output_dir}/{pretrain_metrics.name} pretraining with fine tuning on {finetune_metrics.name}.png"
    )
    plt.close()


def _plot_metric_curves(
        performance_metrics: Dict[str, List[float]],
        validation_metrics: Dict[str, List[float]],
        task_name: str,
        is_baseline: bool,
        output_dir: str
) -> None:
    """
    Plot and save performance metric curves.

    Args:
        performance_metrics: Dict mapping metric names to lists of training values
        validation_metrics: Dict mapping metric names to lists of validation values
        task_name: Name of the task being evaluated
        is_baseline: Whether this is a baseline model (frozen backbone)
        output_dir: Directory to save plots
    """
    for metric in performance_metrics:
        plt.figure()
        plt.plot(performance_metrics[metric], label="training")
        plt.plot(validation_metrics[metric], label="validation")
        plt.xlabel("epochs")
        plt.ylabel(metric)
        plt.legend()

        title = f"{task_name} {'baseline ' if is_baseline else ''}{metric}"
        plt.title(title)
        plt.savefig(f"{output_dir}/{title}")
        plt.close()


def _plot_statistical_curves(
        train_losses: List[List[float]],
        val_losses: List[List[float]],
        confidence_level: float,
        task_name: str,
        is_baseline: bool,
        output_dir: str
) -> None:
    """
    Plot average learning curves with confidence intervals.

    Args:
        train_losses: List of training loss histories from each trial
        val_losses: List of validation loss histories from each trial
        confidence_level: Confidence level for interval calculations (0-1)
        task_name: Name of the task being evaluated
        is_baseline: Whether this is a baseline model (frozen backbone)
        output_dir: Directory to save plots
    """
    train_array = np.array(train_losses)
    val_array = np.array(val_losses)
    epochs = range(1, len(train_losses[0]) + 1)

    plt.figure(figsize=(10, 6))

    # Plot training data
    train_mean = np.mean(train_array, axis=0)
    train_std = np.std(train_array, axis=0)
    plt.plot(epochs, train_mean, 'b-', label='Training Loss (mean)')
    plt.fill_between(
        epochs,
        train_mean - train_std,
        train_mean + train_std,
        color='b',
        alpha=0.2,
        label=f'{int(confidence_level * 100)}% CI'
    )

    # Plot validation data
    val_mean = np.mean(val_array, axis=0)
    val_std = np.std(val_array, axis=0)
    plt.plot(epochs, val_mean, 'r-', label='Validation Loss (mean)')
    plt.fill_between(
        epochs,
        val_mean - val_std,
        val_mean + val_std,
        color='r',
        alpha=0.2
    )

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    title = f"{task_name} {'Baseline ' if is_baseline else ''}Statistical Learning Curves"
    plt.title(title)
    plt.legend()
    plt.grid(True)

    plt.savefig(f"{output_dir}/{title.lower().replace(' ', '_')}")
    plt.close()


def _compute_trial_statistics(
        train_losses: List[List[float]],
        val_losses: List[List[float]],
        confidence_level: float
) -> Dict[str, Dict[str, Any]]:
    """
    Compute statistical metrics from multiple training trials.

    Args:
        train_losses: List of training loss histories from each trial
        val_losses: List of validation loss histories from each trial
        confidence_level: Confidence level for interval calculations (0-1)

    Returns:
        Dictionary containing:
        - Mean, std, min, max for losses and metrics
        - Confidence intervals for each epoch
    """
    # Convert lists to numpy arrays for easier computation
    train_array = np.array(train_losses)
    val_array = np.array(val_losses)

    # Compute basic statistics
    loss_stats = {
        'train': {
            'mean': np.mean(train_array, axis=0),
            'std': np.std(train_array, axis=0),
            'min': np.min(train_array, axis=0),
            'max': np.max(train_array, axis=0)
        },
        'val': {
            'mean': np.mean(val_array, axis=0),
            'std': np.std(val_array, axis=0),
            'min': np.min(val_array, axis=0),
            'max': np.max(val_array, axis=0)
        }
    }

    # Compute confidence intervals
    train_ci = stats.t.interval(
        confidence_level,
        df=len(train_losses) - 1,
        loc=np.mean(train_array, axis=0),
        scale=stats.sem(train_array, axis=0)
    )
    val_ci = stats.t.interval(
        confidence_level,
        df=len(val_losses) - 1,
        loc=np.mean(val_array, axis=0),
        scale=stats.sem(val_array, axis=0)
    )

    confidence_intervals = {
        'train': {'lower': train_ci[0], 'upper': train_ci[1]},
        'val': {'lower': val_ci[0], 'upper': val_ci[1]}
    }

    return {
        'losses': loss_stats,
        'confidence_intervals': confidence_intervals
    }
