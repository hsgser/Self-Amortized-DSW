import json
import random

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from add_noise_to_data.random_noise import RandomNoiseAdder
from dataset.shapenet_core55 import ShapeNetCore55XyzOnlyDataset
from loss.amortized_functions import (
    Attention_Mapping,
    EfficientAttention_Mapping,
    Generalized_Linear_Mapping,
    Linear_Mapping,
    LinearAttention_Mapping,
    Non_Linear_Mapping,
)
from loss.sw_variants import (
    compute_practical_moments_sw_with_predefined_projections,
    minibatch_rand_projections,
    proj_onto_unit_sphere,
)
from loss.von_mises_fisher import VonMisesFisher
from torch.autograd import Variable
from tqdm import tqdm


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def set_seed(args):
    # Set seed
    random.seed(args["seed"])
    np.random.seed(args["seed"])
    torch.manual_seed(args["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Config parameters
CONFIG = "config.json"
NUM_PROJS = 10
DATA_PATH = "dataset/shapenet_core55/shapenet57448xyzonly.npz"
args = json.load(open(CONFIG))
args["seed"] = 123
args["device"] = "cuda:6"
device = torch.device(args["device"])

# Dataset
print("Start creating dataloader.")
if args["add_noise"]:

    if args["noise_adder"] == "random":
        noise_adder = RandomNoiseAdder(mean=args["mean_noiseadder"], std=args["std_noiseadder"])
    else:
        raise ValueError("Unknown noise_adder type.")

if args["train_set"] == "shapenetcore55":
    # choose test phase to sample wo replacement
    dataset = ShapeNetCore55XyzOnlyDataset(DATA_PATH, num_points=args["num_points"], phase="test")
else:
    raise Exception("Unknown dataset")

set_seed(args)
loader = data.DataLoader(
    dataset,
    batch_size=2,
    pin_memory=True,
    num_workers=args["num_workers"],
    shuffle=True,
    worker_init_fn=seed_worker,
)
print("Finish creating dataloader.")

# Extract toy dataset
NUM_PAIRS = 1000
BATCH_SIZE = 100
num_batches = NUM_PAIRS // BATCH_SIZE
list_X = []
list_Y = []

with torch.no_grad():
    for batch_idx, batch in tqdm(enumerate(loader)):
        if batch_idx >= NUM_PAIRS:
            break
        list_X.append(batch[0])
        list_Y.append(batch[1])

X = torch.stack(list_X).to(device)
Y = torch.stack(list_Y).to(device)

X = X.view(-1, BATCH_SIZE, X.size(1), X.size(2))
Y = Y.view(-1, BATCH_SIZE, Y.size(1), Y.size(2))
print(X.size(), Y.size())


# Find max sliced projection
def find_max_locations(x, y, init_locs=[], kappa=1, num_projs=100, num_iters=50, optimizer="adam", device="cuda"):
    # define projection
    if len(init_locs) > 0:
        # print("Use initialized projections")
        locs = init_locs.detach().clone()
        locs.requires_grad = True
    else:
        # print("Initialize projections")
        locs = Variable(
            minibatch_rand_projections(batchsize=x.size(0), dim=x.size(2), num_projections=1, device=device).squeeze(
                1
            ),
            requires_grad=True,
        )

    scales = torch.full((x.size(0), 1), kappa, device=device)

    # define optimizer
    if optimizer == "adam":
        optimizer = torch.optim.Adam(
            [locs],
            lr=1e-4,
        )
    elif optimizer == "sgd":
        optimizer = torch.optim.SGD(
            [locs],
            lr=1e-1,
        )
    # detach inp
    x_detach, y_detach = x.detach(), y.detach()

    for _ in range(num_iters):
        # compute loss
        vmf = VonMisesFisher(locs, scales)
        projections = vmf.rsample(num_projs).transpose(0, 1)
        first_moment, _ = compute_practical_moments_sw_with_predefined_projections(
            x_detach, y_detach, projections, device=device
        )
        negative_first_moment = (-first_moment).mean()
        # print(negative_first_moment)

        # perform optimization
        optimizer.zero_grad()
        negative_first_moment.backward()
        optimizer.step()
        # project onto unit sphere
        locs.data = proj_onto_unit_sphere(locs.data, dim=1)

    locs_no_grad = locs.detach()
    vmf = VonMisesFisher(locs_no_grad, scales)
    # sample: num_projs x batch_size x dim
    # projections: batch_size x num_projs x dim
    projections = vmf.rsample(num_projs).transpose(0, 1)
    loss, _ = compute_practical_moments_sw_with_predefined_projections(x, y, projections, device=device)
    loss = loss.mean(dim=0)

    return {"loss": loss, "loc": locs_no_grad}


for n_iter in [1, 5, 10, 50, 100]:
    print(f"T = {n_iter}")
    vsw_sgd_loss = 0
    list_vsw_sgd_locs = []
    set_seed(args)

    for bX, bY in tqdm(zip(X, Y)):
        vsw_result = find_max_locations(
            bX, bY, init_locs=[], optimizer="sgd", kappa=1, num_projs=NUM_PROJS, num_iters=n_iter, device=device
        )
        vsw_sgd_loss += vsw_result["loss"]
        list_vsw_sgd_locs.append(vsw_result["loc"])

    vsw_sgd_loss = vsw_sgd_loss / num_batches
    list_vsw_sgd_locs = torch.cat(list_vsw_sgd_locs)
    print(vsw_sgd_loss)
    print(list_vsw_sgd_locs.size())


# Amortized models
def train_amortize(X, Y, f_slice, kappa=1, num_projs=100, num_iters=50, optimizer="adam", device="cuda"):
    # define optimizer
    if optimizer == "adam":
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, f_slice.parameters()),
            lr=1e-4,
        )
    elif optimizer == "sgd":
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, f_slice.parameters()),
            lr=1e-1,
        )
    # detach inp
    X_detach, Y_detach = X.detach(), Y.detach()
    scales = torch.full((X.size(1), 1), kappa, device=device)

    for _ in tqdm(range(num_iters)):
        # compute loss
        for bX, bY in zip(X_detach, Y_detach):
            locs = f_slice(torch.cat([bX, bY], dim=1), detach=True).squeeze(1)
            vmf = VonMisesFisher(locs, scales)
            projections = vmf.rsample(num_projs).transpose(0, 1)
            first_moment, _ = compute_practical_moments_sw_with_predefined_projections(
                bX, bY, projections, device=device
            )
            negative_first_moment = (-first_moment).mean()
            # print(negative_first_moment)

            # perform optimization
            optimizer.zero_grad()
            negative_first_moment.backward()
            optimizer.step()

    # calculate loss + locs
    amort_loss = 0
    list_locs = []

    for bX, bY in zip(X_detach, Y_detach):
        locs = f_slice(torch.cat([bX, bY], dim=1), detach=True).squeeze(1)
        locs_no_grad = locs.detach()
        vmf = VonMisesFisher(locs_no_grad, scales)
        projections = vmf.rsample(num_projs).transpose(0, 1)
        loss, _ = compute_practical_moments_sw_with_predefined_projections(bX, bY, projections, device=device)
        loss = loss.mean(dim=0)
        amort_loss += loss
        list_locs.append(locs_no_grad)

    amort_loss = amort_loss / X_detach.size(0)
    list_locs = torch.cat(list_locs)

    return {"loss": amort_loss, "loc": list_locs}


# Compare amortized models
proj_sharing = False
set_seed(args)
linear = Linear_Mapping(args["num_points"]).to(device)
set_seed(args)
glinear = Generalized_Linear_Mapping(args["num_points"], args["input_channels"]).to(device)
set_seed(args)
nonlinear = Non_Linear_Mapping(args["num_points"], args["input_channels"]).to(device)
set_seed(args)
attn = Attention_Mapping(args["input_channels"], 1).to(device)
set_seed(args)
eff_attn = EfficientAttention_Mapping(args["input_channels"], 1).to(device)
set_seed(args)
lin_attn = LinearAttention_Mapping(args["num_points"], 64, args["input_channels"], 32, sharing=proj_sharing).to(device)

list_slice = [
    linear,
    glinear,
    nonlinear,
    attn,
    eff_attn,
    lin_attn,
]

result_dic = {
    "name": [],
    "loss": [],
}

for f_slice in list_slice:
    set_seed(args)
    slice_result = train_amortize(
        X, Y, f_slice, kappa=1, num_projs=NUM_PROJS, optimizer="sgd", num_iters=1, device=device
    )

    result_dic["name"].append(f_slice.__class__.__name__)
    result_dic["loss"].append(slice_result["loss"].item())

df_result = pd.DataFrame(result_dic)
print(df_result)
print("VSW: ", vsw_sgd_loss.item())
