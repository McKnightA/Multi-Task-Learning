import time
import Datasets
import Tasks
import Models
import Tests


# hyper parameters
embedding_size = 1024
batch_size = 64
pre_epochs = 5
inner_loops = 1
fine_epochs = 10
device = 'cuda'

dataset = Datasets.Cifar10(device=device)
# testset = Datasets.Cifar10(device=device)
# testset1 = Datasets.Cifar100(device=device)
# testset2 = Datasets.Beans(device=device)
# testset3 = Datasets.Svhn(device=device)
# testset4 = Datasets.Fmd(device=device)
# testset5 = Datasets.Pokemon(device=device)

# p=4, l=2  [0, 0, 0, 0],  # control = random weights
taguchi_array = [[0, 0, 0, 1],
                 [0, 1, 1, 0],
                 [0, 1, 1, 1],
                 [1, 0, 1, 0],
                 [1, 0, 1, 1],
                 [1, 1, 0, 0],
                 [1, 1, 0, 1]]


def spawn_tasklist():
    return [Tasks.Rotation(embedding_size, Models.BasicNonLinearPredictor, device),
            Tasks.Colorization(embedding_size, Models.MobileViTv3Projector, device),
            Tasks.Contrastive(embedding_size, Models.BasicNonLinearPredictor, device),
            Tasks.MaskedAutoEncoding(embedding_size, Models.MobileViTv3Projector, device)] # ,
#             Tasks.Classification(embedding_size, Models.BasicNonLinearPredictor, testset.num_classes, testset.name, device)]


def spawn_backbone():
    return Models.MobileViTv3Encoder(embedding_size, device)


def spawn_task():
    return Tasks.Classification(embedding_size, Models.BasicNonLinearPredictor, dataset.num_classes, dataset.name, device)


backbone = spawn_backbone()
# tasks = spawn_tasklist()

train_task = spawn_task()
"""
eval_task = Tasks.Classification(embedding_size, Models.BasicNonLinearPredictor, testset.num_classes, testset.name, device)
eval_task1 = Tasks.Classification(embedding_size, Models.BasicNonLinearPredictor, testset1.num_classes, testset1.name, device)
eval_task2 = Tasks.Classification(embedding_size, Models.BasicNonLinearPredictor, testset2.num_classes, testset2.name, device)
eval_task3 = Tasks.Classification(embedding_size, Models.BasicNonLinearPredictor, testset3.num_classes, testset3.name, device)
eval_task4 = Tasks.Classification(embedding_size, Models.BasicNonLinearPredictor, testset4.num_classes, testset4.name, device)
eval_task5 = Tasks.Classification(embedding_size, Models.BasicNonLinearPredictor, testset5.num_classes, testset5.name, device)
"""
# -----------------------------------------------------------------------------------

now = time.time()
Tests.train_and_evaluate_task(dataset, dataset, backbone, train_task, pre_epochs, batch_size, use_frozen_backbone=False)
print(f"total time elapsed during training: {time.time() - now}")
"""
now = time.time()
Tests.run_statistical_trials(
    training_set=dataset,
    validation_set=dataset,
    backbone_factory=spawn_backbone,
    task_factory=spawn_task,
    num_trials=10,
    num_epochs=20,
    batch_size=batch_size,
    confidence_level=0.85
)
print(f"total time elapsed during training: {time.time() - now}")
"""

# Tests.evaluate_with_probing(dataset,
#                             [testset1, testset2, testset3, testset4, testset5],
#                             backbone, train_task,
#                             [eval_task1, eval_task2, eval_task3, eval_task4, eval_task5],
#                             pre_epochs, fine_epochs, batch_size)
# Tests.pretrain_and_finetune(dataset, testset, backbone, train_task, eval_task, pre_epochs, fine_epochs, batch_size)

"""
for trial in taguchi_array:
    trial_tasks = []

    del tasks
    tasks = spawn_tasklist()
    del backbone
    backbone = spawn_backbone()

    for variable in zip(trial, tasks):
        if variable[0]:
            trial_tasks.append(variable[1])

    if len(trial_tasks) == 0:
        baseline = True
    else:
        baseline = False

    train_task = Tasks.AveragedLossMultiTask(trial_tasks, device)
    Tests.evaluate_with_probing(dataset,
                                [testset1, testset2, testset3, testset4, testset5],
                                backbone, train_task,
                                [eval_task1, eval_task2, eval_task3, eval_task4, eval_task5],
                                pre_epochs, fine_epochs, batch_size)
    # Tests.test_finetune(dataset, testset, backbone, train_task, eval_task, pre_epochs, fine_epochs, batch_size)
"""
