# %%

import os
import sys
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules

chapter = "chapter0_fundamentals"
repo = "ARENA_3.0"
branch = "main"

# # Install dependencies
# try:
#     import jaxtyping
# except:
#     %pip install einops jaxtyping torchinfo wandb

# Get root directory, handling 3 different cases: (1) Colab, (2) notebook not in ARENA repo, (3) notebook in ARENA repo
root = (
    "/content"
    if IN_COLAB
    else "/root"
    if repo not in os.getcwd()
    else str(next(p for p in [Path.cwd()] + list(Path.cwd().parents) if p.name == repo))
)

# if Path(root).exists() and not Path(f"{root}/{chapter}").exists():
#     if not IN_COLAB:
#         !sudo apt-get install unzip
#         %pip install jupyter ipython --upgrade
#
#     if not os.path.exists(f"{root}/{chapter}"):
#         !wget -P {root} https://github.com/callummcdougall/ARENA_3.0/archive/refs/heads/{branch}.zip
#         !unzip {root}/{branch}.zip '{repo}-{branch}/{chapter}/exercises/*' -d {root}
#         !mv {root}/{repo}-{branch}/{chapter} {root}/{chapter}
#         !rm {root}/{branch}.zip
#         !rmdir {root}/{repo}-{branch}


assert Path(f"{root}/{chapter}/exercises").exists(), (
    "Unexpected error: please manually clone ARENA repo into `root`"
)

if f"{root}/{chapter}/exercises" not in sys.path:
    sys.path.append(f"{root}/{chapter}/exercises")

os.chdir(f"{root}/{chapter}/exercises")



# %% 



import importlib
import os
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Iterable, Literal

import numpy as np
import torch as t
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import wandb
from IPython.core.display import HTML
from IPython.display import display
from jaxtyping import Float, Int
from torch import Tensor, optim
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms
from tqdm import tqdm

# Make sure exercises are in the path
chapter = "chapter0_fundamentals"
section = "part3_optimization"
root_dir = next(p for p in Path.cwd().parents if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

MAIN = __name__ == "__main__"

import part3_optimization.tests as tests
from part2_cnns.solutions import Linear, ResNet34, get_resnet_for_feature_extraction
from part3_optimization.utils import plot_fn, plot_fn_with_points
from plotly_utils import bar, imshow, line

device = t.device(
    "mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu"
)



# %%

WORLD_SIZE = min(t.cuda.device_count(), 3)

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12345"


def send_receive(rank, world_size):
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

    if rank == 0:
        # Send tensor to rank 1
        sending_tensor = t.zeros(1)
        print(f"{rank=}, sending {sending_tensor=}")
        dist.send(tensor=sending_tensor, dst=1)
    elif rank == 1:
        # Receive tensor from rank 0
        received_tensor = t.ones(1)
        print(f"{rank=}, creating {received_tensor=}")
        dist.recv(received_tensor, src=0)  # this line overwrites the tensor's data with our `sending_tensor`
        print(f"{rank=}, received {received_tensor=}")

    dist.destroy_process_group()


# if MAIN:
#     world_size = 2  # simulate 2 processes
#     mp.spawn(
#         send_receive,
#         args=(world_size,),
#         nprocs=world_size,
#         join=True,
#     )


# %%

assert t.cuda.is_available()
assert t.cuda.device_count() > 1, "This example requires at least 2 GPUs per machine"

# %%

def send_receive_nccl(rank, world_size):
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    device = t.device(f"cuda:{rank}")

    if rank == 0:
        # Create a tensor, send it to rank 1
        sending_tensor = t.tensor([rank], device=device)
        print(f"{rank=}, {device=}, sending {sending_tensor=}")
        dist.send(sending_tensor, dst=1)  # Send tensor to CPU before sending
    elif rank == 1:
        # Receive tensor from rank 0 (it needs to be on the CPU before receiving)
        received_tensor = t.tensor([rank], device=device)
        print(f"{rank=}, {device=}, creating {received_tensor=}")
        dist.recv(received_tensor, src=0)  # this line overwrites the tensor's data with our `sending_tensor`
        print(f"{rank=}, {device=}, received {received_tensor=}")

    dist.destroy_process_group()


# if MAIN:
#     world_size = 2  # simulate 2 processes
#     mp.spawn(
#         send_receive_nccl,
#         args=(world_size,),
#         nprocs=world_size,
#         join=True,
#     )


# %%

def broadcast(tensor: Tensor, rank: int, world_size: int, src: int = 0):
    """
    Broadcast averaged gradients from rank 0 to all other ranks.
    """
    if rank == src:
        for i in range(world_size):
            if i != src:
                dist.send(tensor, dst=i)
    if rank != src:
        received_tensor = tensor
        dist.recv(received_tensor, src=src)
        tensor.copy_(received_tensor)



# if MAIN:
#     tests.test_broadcast(broadcast, WORLD_SIZE)


# %%

def reduce(tensor, rank, world_size, dst=0, op: Literal["sum", "mean"] = "sum"):
    """
    Reduces gradients to rank `dst`, so this process contains the sum or mean of all tensors across
    processes.
    """
    if rank == dst:
        tensors = [tensor]
        for i in range(world_size):
            if i != dst:
                received_tensor = t.zeros_like(tensor)
                dist.recv(received_tensor, src=i)
                tensors.append(received_tensor)
        if op == "sum":
            tensor.copy_(sum(tensors))
        elif op == "mean":
            tensor.copy_(sum(tensors) / len(tensors))
    else:
        dist.send(tensor, dst=dst)


def all_reduce(tensor, rank, world_size, op: Literal["sum", "mean"] = "sum"):
    """
    Allreduce the tensor across all ranks, using 0 as the initial gathering rank.
    """
    dst = 0

    reduce(tensor, rank, world_size, dst=dst, op=op)
    broadcast(tensor, rank, world_size, src=dst)



# if MAIN:
#     tests.test_reduce(reduce, WORLD_SIZE)
#     tests.test_all_reduce(all_reduce, WORLD_SIZE)


# %%

class SimpleModel(t.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.param = t.nn.Parameter(t.tensor([2.0]))

    def forward(self, x: Tensor):
        return x - self.param


def run_simple_model(rank, world_size):
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    device = t.device(f"cuda:{rank}")
    model = SimpleModel().to(device)  # Move the model to the device corresponding to this process
    optimizer = t.optim.SGD(model.parameters(), lr=0.1)

    input = t.tensor([rank], dtype=t.float32, device=device)
    output = model(input)
    loss = output.pow(2).sum()
    loss.backward()  # Each rank has separate gradients at this point

    print(f"Rank {rank}, before all_reduce, grads: {model.param.grad=}")
    all_reduce(model.param.grad, rank, world_size)  # Synchronize gradients
    print(f"Rank {rank}, after all_reduce, synced grads (summed over processes): {model.param.grad=}")

    optimizer.step()  # Step with the optimizer (this will update all models the same way)
    print(f"Rank {rank}, new param: {model.param.data}")

    dist.destroy_process_group()


# if MAIN:
#     world_size = 2
#     mp.spawn(
#         run_simple_model,
#         args=(world_size,),
#         nprocs=world_size,
#         join=True,
#     )


# %%

@dataclass
class ResNetFinetuningArgs:
    n_classes: int = 10
    batch_size: int = 128
    epochs: int = 3
    learning_rate: float = 1e-3
    weight_decay: float = 0.0

@dataclass
class WandbResNetFinetuningArgs(ResNetFinetuningArgs):
    """Contains new params for use in wandb.init, as well as all the ResNetFinetuningArgs params."""

    wandb_project: str | None = "day3-resnet"
    wandb_name: str | None = None


def get_cifar() -> tuple[datasets.CIFAR10, datasets.CIFAR10]:
    """Returns CIFAR-10 train and test sets."""
    cifar_trainset = datasets.CIFAR10(
        exercises_dir / "data", train=True, download=True, transform=IMAGENET_TRANSFORM
    )
    cifar_testset = datasets.CIFAR10(
        exercises_dir / "data", train=False, download=True, transform=IMAGENET_TRANSFORM
    )
    return cifar_trainset, cifar_testset

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

IMAGENET_TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)

# %%

def get_untrained_resnet(n_classes: int) -> ResNet34:
    """
    Gets untrained resnet using code from part2_cnns.solutions (you can replace this with your
    implementation).
    """
    resnet = ResNet34()
    resnet.out_layers[-1] = Linear(resnet.out_features_per_group[-1], n_classes)
    return resnet


@dataclass
class DistResNetTrainingArgs(WandbResNetFinetuningArgs):
    world_size: int = 1
    wandb_project: str | None = "day3-resnet-dist-training"


class DistResNetTrainer:
    args: DistResNetTrainingArgs

    def __init__(self, args: DistResNetTrainingArgs, rank: int):
        self.args = args
        self.rank = rank
        self.device = t.device(f"cuda:{rank}")

    def pre_training_setup(self):
        self.model = get_untrained_resnet(n_classes=self.args.n_classes).to(self.device)

        if self.args.world_size > 1:
            for param in self.model.parameters():
                broadcast(param.data, self.rank, self.args.world_size, src=0)

        self.optimizer = t.optim.AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay
        )

        self.trainset, self.testset = get_cifar()

        if self.args.world_size > 1:
            sampler_shared_kwargs = dict(num_replicas=self.args.world_size, rank=self.rank)
            self.train_sampler = t.utils.data.DistributedSampler(self.trainset, **sampler_shared_kwargs)
            self.test_sampler = t.utils.data.DistributedSampler(self.testset, **sampler_shared_kwargs)
        else:
            self.train_sampler = None
            self.test_sampler = None

        dataloader_shared_kwargs = dict(batch_size=self.args.batch_size, num_workers=2, pin_memory=True)
        self.train_loader = t.utils.data.DataLoader(self.trainset, sampler=self.train_sampler, **dataloader_shared_kwargs)
        self.test_loader = t.utils.data.DataLoader(self.testset, sampler=self.test_sampler, **dataloader_shared_kwargs)

        self.examples_seen = 0

        if self.rank == 0:
            wandb.init(project=self.args.wandb_project, name=self.args.wandb_name, config=self.args)


    def training_step(self, imgs: Tensor, labels: Tensor) -> Tensor:
        t0 = time.time()

        imgs, labels = imgs.to(self.device), labels.to(self.device)
        logits = self.model(imgs)
        t1 = time.time()

        loss = F.cross_entropy(logits, labels)
        loss.backward()
        t2 = time.time()

        if self.args.world_size > 1:
            for param in self.model.parameters():
                all_reduce(param.grad, self.rank, self.args.world_size, op="mean")
        t3 = time.time()


        self.optimizer.step()
        self.optimizer.zero_grad()

        total_seen_tensor = t.tensor([imgs.shape[0]], device=self.device)

        if self.args.world_size > 1:
            all_reduce(total_seen_tensor, self.rank, self.args.world_size, op="sum")

        self.examples_seen += total_seen_tensor.item()
        if self.rank == 0:
            wandb.log({
                "loss": loss.item(),
                "fwd_time": t1 - t0,
                "bwd_time": t2 - t1,
                "dist_time": t3 - t2,
            },
            step=self.examples_seen)

        return loss

    @t.inference_mode()
    def evaluate(self) -> float:
        self.model.eval()
        total_correct, total_samples = 0, 0

        for imgs, labels in tqdm(self.test_loader, desc="Evaluating"):
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            logits = self.model(imgs)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += len(imgs)

        total_correct_tensor = t.tensor([total_correct], device=self.device)
        total_sample_tensor = t.tensor([total_samples], device=self.device)

        if self.args.world_size > 1:
            all_reduce(total_correct_tensor, self.rank, self.args.world_size, op="sum")
            all_reduce(total_sample_tensor, self.rank, self.args.world_size, op="sum")

        accuracy = total_correct_tensor.item() / total_sample_tensor.item()

        if self.rank == 0:
            wandb.log({"accuracy": accuracy}, step=self.examples_seen)

        return accuracy

    def train(self):
        self.pre_training_setup()
        accuracy = self.evaluate()

        for epoch in range(self.args.epochs):
            t_start = time.time()

            if self.args.world_size > 1:
                self.train_sampler.set_epoch(epoch)
                self.test_sampler.set_epoch(epoch)

            self.model.train()

            pbar = tqdm(self.train_loader, desc="Training", disable=(self.rank != 0))
            for imgs, labels in pbar:
                loss = self.training_step(imgs, labels)
                pbar.set_postfix(loss=f"{loss:.3f}", ex_seen=f"{self.examples_seen:06}")

            accuracy = self.evaluate()

            if self.rank == 0:
                wandb.log({"epoch_duration": time.time()-t_start}, step=self.examples_seen),
                pbar.set_postfix(loss=f"{loss:.3f}", accuracy=f"{accuracy:.2f}", ex_seen=f"{self.examples_seen:06}")

        if self.rank == 0:
            wandb.finish()
            t.save(self.model.state_dict(), f"resnet_{self.rank}.pth")


def dist_train_resnet_from_scratch(rank, world_size):
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    args = DistResNetTrainingArgs(world_size=world_size)
    trainer = DistResNetTrainer(args, rank)
    trainer.train()
    dist.destroy_process_group()


if MAIN:
    world_size = t.cuda.device_count()
    mp.spawn(
        dist_train_resnet_from_scratch,
        args=(world_size,),
        nprocs=world_size,
        join=True,
    )