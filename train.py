import argparse
import datetime
import json
import os
import os.path as osp
import random
import shutil
import time

import numpy as np
import torch
from add_noise_to_data.random_noise import RandomNoiseAdder
from dataset import ShapeNetCore55XyzOnlyDataset
from evaluator import Evaluator
from logger import Logger
from loss import ASW, EMD, SWD, VSW, Amortized_MSW, Amortized_VSW, Chamfer, MaxSW
from loss.amortized_functions import (
    Attention_Mapping,
    EfficientAttention_Mapping,
    Generalized_Linear_Mapping,
    Linear_Mapping,
    LinearAttention_Mapping,
    LinearLrDecay,
    Non_Linear_Mapping,
)
from models import PointCapsNet, PointNetAE
from models.utils import init_weights
from saver import Saver
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import CyclicLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from trainer import AETrainer as Trainer
from utils import get_lr


torch.backends.cudnn.enabled = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="path to json config file")
    parser.add_argument("--logdir", help="path to the log directory")
    parser.add_argument("--data_path", help="path to data")
    parser.add_argument(
        "--loss", default="swd", help="[chamfer, emd, asw, swd, msw, vsw, amortized_msw, amortized_vsw]"
    )
    parser.add_argument(
        "--f_type",
        default="linear",
        help="[linear, glinear, nonlinear, attn, eff_attn, lin_attn]",
    )
    parser.add_argument("--inter_dim", default=64, type=int, help="dimension of keys")
    parser.add_argument("--proj_dim", default=64, type=int, help="projected dimension in linformer")
    parser.add_argument("--proj_sharing", action="store_true", help="sharing projection for key and value")
    parser.add_argument("--kappa", default=1.0, type=float, help="scale of vMF distribution")
    parser.add_argument("--num_projs", default=100, type=int, help="number of projections")
    parser.add_argument("--learning_rate", default=0.001, type=float, help="learning rate to train autoencoder")
    parser.add_argument("--seed", default=1, type=int, help="number of projections")
    parser.add_argument("--autoencoder", default="pointnet", help="[pointnet, pcn]")
    args = parser.parse_args()
    config = args.config
    logdir = args.logdir
    data_path = args.data_path
    loss_type = args.loss
    ae_type = args.autoencoder
    f_type = args.f_type
    inter_dim = args.inter_dim
    proj_dim = args.proj_dim
    num_projs = args.num_projs
    learning_rate = args.learning_rate
    seed = args.seed
    proj_sharing = args.proj_sharing
    kappa = args.kappa
    print("Save checkpoints and logs in: ", logdir)
    args = json.load(open(config))
    args["autoencoder"] = ae_type
    args["loss"] = loss_type
    args["num_projs"] = num_projs
    args["seed"] = seed
    args["learning_rate"] = learning_rate

    # set seed
    torch.manual_seed(args["seed"])
    random.seed(args["seed"])
    np.random.seed(args["seed"])

    if not os.path.exists(logdir):
        os.makedirs(logdir)
        print(">Logdir was created successfully at: ", logdir)
    else:
        print(">Folder {} is existing.".format(logdir))
        print(">Do you want to remove it?")
        answer = None
        while answer not in ("yes", "no"):
            answer = input("Enter 'yes' or 'no': ")
            if answer == "yes":
                shutil.rmtree(logdir)
                os.makedirs(logdir)
            elif answer == "no":
                print("SOME FILES WILL BE OVERWRITTEN OR APPENDED.")
                print("If you do not want this, please stop during next 30s.")
                time.sleep(30)
            else:
                print("Please enter 'yes' or 'no'.")
    fname = os.path.join(logdir, "train_ae_config.json")
    with open(fname, "w") as fp:
        json.dump(args, fp, indent=4)

    # print hyperparameters
    print(">You have 5s to check the hyperparameters below.")
    print(args)
    time.sleep(5)

    # init dic of extra parameters for trainer.train
    dic = {}

    # device
    device = torch.device(args["device"])

    # NoiseAdder
    if args["add_noise"]:
        if args["noise_adder"] == "random":
            noise_adder = RandomNoiseAdder(mean=args["mean_noiseadder"], std=args["std_noiseadder"])
        else:
            raise ValueError("Unknown noise_adder type.")

    # autoencoder architecture
    if args["autoencoder"] == "pointnet":
        autoencoder = PointNetAE(
            args["embedding_size"],
            args["input_channels"],
            args["input_channels"],
            args["num_points"],
            args["normalize"],
        ).to(device)

    elif args["autoencoder"] == "pcn":
        autoencoder = PointCapsNet(
            args["prim_caps_size"],
            args["prim_vec_size"],
            args["latent_caps_size"],
            args["latent_vec_size"],
            args["num_points"],
        ).to(device)

    else:
        raise Exception("Unknown autoencoder.")

    # loss function
    dic["squared_loss"] = args["squared_loss"]
    if args["loss"] == "chamfer":
        loss_func = Chamfer(args["version"])

    elif args["loss"] == "emd":
        loss_func = EMD()

    elif args["loss"] == "swd":
        loss_func = SWD(args["num_projs"], device)

    elif args["loss"] == "asw":
        sample_projs_history = os.path.join(logdir, "sample_projs_history.txt")
        loss_func = ASW(
            args["init_projs"],
            args["step_projs"],
            loop_rate_thresh=args["loop_rate_thresh"],
            projs_history=sample_projs_history,
            max_slices=args["max_slices"],
        )
        dic = {"epsilon": args["init_rec_epsilon"]}
        dic["degree"] = args["degree"]

    elif args["loss"] == "msw":
        loss_func = MaxSW(device=device)
        dic["detach"] = args["detach"]
        dic["max_sw_num_iters"] = args["max_sw_num_iters"]
        dic["max_sw_lr"] = args["max_sw_lr"]
        dic["max_sw_optimizer"] = args["max_sw_optimizer"]

    elif args["loss"] == "vsw":
        loss_func = VSW(num_projs=args["num_projs"], device=device)
        dic["detach"] = args["detach"]
        dic["kappa"] = kappa
        dic["max_sw_num_iters"] = args["max_sw_num_iters"]
        dic["max_sw_lr"] = args["max_sw_lr"]
        dic["max_sw_optimizer"] = args["max_sw_optimizer"]

    elif "amortized" in args["loss"]:
        dic["detach"] = args["detach"]
        dic["finetune"] = args["finetune"]
        dic["kappa"] = kappa
        dic["amortize_num_iters"] = args["amortize_num_iters"]
        dic["amortize_start_epoch"] = args["amortize_start_epoch"]
        dic["amortize_end_epoch"] = args["amortize_end_epoch"]
        dic["max_sw_num_iters"] = args["max_sw_num_iters"]
        dic["max_sw_lr"] = args["max_sw_lr"]
        dic["max_sw_optimizer"] = args["max_sw_optimizer"]

        if f_type == "linear":
            f_slice = Linear_Mapping(args["num_points"]).cuda()
        elif f_type == "glinear":
            f_slice = Generalized_Linear_Mapping(args["num_points"], args["input_channels"]).cuda()
        elif f_type == "nonlinear":
            f_slice = Non_Linear_Mapping(args["num_points"], args["input_channels"]).cuda()
        elif f_type == "attn":
            f_slice = Attention_Mapping(args["num_points"], args["input_channels"], inter_dim).cuda()
        elif f_type == "eff_attn":
            f_slice = EfficientAttention_Mapping(args["num_points"], args["input_channels"], inter_dim).cuda()
        elif f_type == "lin_attn":
            f_slice = LinearAttention_Mapping(
                args["num_points"], proj_dim, args["input_channels"], inter_dim, sharing=proj_sharing
            ).cuda()
        else:
            raise Exception("Unknown amortized functions")

        # Fix optimizer for amortized functions for now
        # TODO: Select optimizer from config
        if args["amortize_optimizer"] == "adam":
            dic["f_optimizer"] = torch.optim.Adam(
                filter(lambda p: p.requires_grad, f_slice.parameters()), args["s_lr"], (args["beta1"], args["beta2"])
            )
        elif args["amortize_optimizer"] == "sgd":
            dic["f_optimizer"] = torch.optim.SGD(
                filter(lambda p: p.requires_grad, f_slice.parameters()),
                args["s_lr"],
            )
        else:
            raise Exception("Unknown optimizer")

        if args["loss"] == "amortized_msw":
            loss_func = Amortized_MSW(f_slice=f_slice, num_projs=args["num_projs"], device=device)
        elif args["loss"] == "amortized_vsw":
            loss_func = Amortized_VSW(f_slice=f_slice, num_projs=args["num_projs"], device=device)

    else:
        raise Exception("Unknown loss function.")

    # dataset
    if args["train_set"] == "shapenetcore55":
        dataset = ShapeNetCore55XyzOnlyDataset(data_path, num_points=args["num_points"], phase="train")

    else:
        raise Exception("Unknown dataset")

    # optimizer
    if args["optimizer"] == "sgd":
        optimizer = SGD(
            autoencoder.parameters(),
            lr=args["learning_rate"],
            momentum=args["momentum"],
            weight_decay=args["weight_decay"],
        )

    elif args["optimizer"] == "adam":
        optimizer = Adam(
            autoencoder.parameters(),
            lr=args["learning_rate"],
            betas=(0.5, 0.999),
            weight_decay=args["weight_decay"],
        )

    else:
        raise Exception("Optimizer has had implementation yet.")

    # init weights
    if osp.isfile(osp.join(logdir, args["checkpoint"])):
        print(">Init weights with {}".format(args["checkpoint"]))
        checkpoint = torch.load(osp.join(logdir, args["checkpoint"]))
        if "autoencoder" in checkpoint.keys():
            autoencoder.load_state_dict(checkpoint["autoencoder"])
        else:
            autoencoder.load_state_dict(checkpoint)
        if "optimizer" in checkpoint.keys():
            try:
                optimizer.load_state_dict(checkpoint["optimizer"])
            except:
                print(">Found no state dict for optimizer.")

    elif osp.isfile(args["checkpoint"]):
        print(">Init weights with {}".format(args["checkpoint"]))
        checkpoint = torch.load(osp.join(args["checkpoint"]))
        if "autoencoder" in checkpoint.keys():
            autoencoder.load_state_dict(checkpoint["autoencoder"])
        else:
            autoencoder.load_state_dict(checkpoint)

    else:
        print(">Init weights with Xavier")
        autoencoder.apply(init_weights)

    # dataloader
    train_loader = DataLoader(
        dataset,
        batch_size=args["batch_size"],
        num_workers=args["num_workers"],
        pin_memory=True,
        shuffle=True,
        worker_init_fn=seed_worker,
    )

    # logger
    tensorboard_dir = osp.join(logdir, "tensorboard")
    if not osp.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)
    tensorboard_logger = Logger(tensorboard_dir)

    # scheduler
    if args["use_scheduler"]:
        if args["scheduler"] == "cyclic_lr":
            scheduler = CyclicLR(optimizer, base_lr=args["base_lr"], max_lr=args["max_lr"])
        else:
            raise Exception("Unknown learning rate scheduler.")

        if "amortized" in args["loss"]:
            f_scheduler = LinearLrDecay(dic["f_optimizer"], args.s_lr, 0.0, 0, args["num_epochs"] * len(train_loader))

    # evaluator
    if args["evaluator"] == "based_on_train_loss":
        args["eval_criteria"] = "loss_func"
        args["have_val_set"] = False

    elif args["evaluator"] == "based_on_val_loss":
        args["eval_criteria"] = "loss_func"
        args["have_val_set"] = True

    else:
        raise ValueError("Unknown evaluator.")

    # val_set and val_loader
    if args["have_val_set"]:
        if args["val_set"] == "shapenetcore55":
            val_set = ShapeNetCore55XyzOnlyDataset(args["val_root"], num_points=args["num_points"], phase="test")

        else:
            raise Exception("Unknown dataset")

        val_loader = DataLoader(
            val_set,
            batch_size=args["val_batch_size"],
            num_workers=args["num_workers"],
            pin_memory=True,
            shuffle=False,
            worker_init_fn=seed_worker,
        )

    # avg_eval_value for model selection
    # init avg_eval_value
    avg_eval_value = args["best_eval_value"]
    best_eval_value = float(args["best_eval_value"])
    best_epoch = int(args["best_epoch"])

    avg_train_loss = args["best_train_loss"]
    best_train_loss = float(args["best_train_loss"])
    best_epoch_based_on_train_loss = int(args["best_epoch_based_on_train_loss"])

    print("best eval value: ", best_eval_value)
    print("best epoch: ", best_epoch)

    # train
    start_epoch = args["start_epoch"]
    num_epochs = args["num_epochs"]

    model_path = os.path.join(logdir, "model.pth")
    best_train_loss_model_path = os.path.join(logdir, "best_train_loss_model.pth")
    f_model_path = os.path.join(logdir, "f_model.pth")
    f_best_train_loss_model_path = os.path.join(logdir, "f_best_train_loss_model.pth")

    rec_train_log_path = os.path.join(logdir, "rec_train.log")
    reg_train_log_path = os.path.join(logdir, "reg_train.log")

    train_log_path = os.path.join(logdir, "train.log")
    eval_log_path = os.path.join(logdir, "eval_when_train.log")

    best_eval_log_path = os.path.join(logdir, "best_eval_when_train.log")
    best_train_log_path = os.path.join(logdir, "best_train.log")

    start_time = time.time()

    dic["iter_id"] = 0
    prev_losses_list = []

    for epoch in tqdm(range(start_epoch, num_epochs)):
        dic["curr_epoch"] = epoch
        # Below optimizer setup as original code of 3D Point Capsule Net https://github.com/yongheng1991/3D-point-capsule-networks/blob/master/apps/AE/train_ae.py
        if args["autoencoder"] == "pcn":
            if epoch < 20:
                optimizer = Adam(autoencoder.parameters(), lr=0.001)
            elif epoch < 50:
                optimizer = Adam(autoencoder.parameters(), lr=0.0001)
            else:
                optimizer = Adam(autoencoder.parameters(), lr=0.00001)

        train_loss_list = []
        rec_train_loss_list = []
        reg_train_loss_list = []

        for batch_id, batch in tqdm(enumerate(train_loader)):
            dic["iter_id"] += 1

            data = batch.to(device)

            if args["add_noise"]:
                if args["train_denoise"]:
                    dic["input"] = data.detach().clone()
                data = noise_adder.add_noise(data)

            # train_on_batch
            result_dic = Trainer.train(autoencoder, loss_func, optimizer, data, **dic)
            autoencoder = result_dic["ae"]
            optimizer = result_dic["optimizer"]
            train_loss = result_dic["loss"]

            # 2 types of losses
            if "rec_loss" in result_dic.keys():
                rec_train_loss_list.append(result_dic["rec_loss"].item())
            if "reg_loss" in result_dic.keys():
                reg_train_loss_list.append(result_dic["reg_loss"].item())

            # append to loss lists
            train_loss_list.append(train_loss.item())

            # update epsilon for adaptive sw
            if "epsilon" in dic.keys():
                if not args["fix_epsilon"]:
                    # updata prev_losses_list
                    assert ("num_prev_losses" in args.keys()) and (args["num_prev_losses"] > 0)
                    if len(prev_losses_list) == args["num_prev_losses"]:
                        prev_losses_list.pop(0)  # pop the first item
                    prev_losses_list.append(train_loss.item())  # add item to the last
                    dic["epsilon"] = min(prev_losses_list) * args["next_epsilon_ratio_rec"]

            if "rec" in dic.keys() and "epsilon" in dic["rec"].keys():
                dic["rec"]["epsilon"] = result_dic["rec_loss"].item() * args["next_epsilon_ratio_rec"]
            if "reg" in dic.keys() and "epsilon" in dic["reg"].keys():
                dic["reg"]["epsilon"] = result_dic["reg_loss"].item() * args["next_epsilon_ratio_reg"]

            # adjust scheduler
            if args["use_scheduler"]:
                scheduler.step()
                if ("amortized" in args["loss"]) and epoch >= dic["amortize_start_epoch"]:
                    f_scheduler.step()

            # write tensorboard log
            info = {"train_loss": train_loss.item(), "learning rate": get_lr(optimizer)}
            if "rec_loss" in result_dic.keys():
                info["rec_train_loss"] = rec_train_loss_list[-1]
            if "reg_loss" in result_dic.keys():
                info["reg_train_loss"] = reg_train_loss_list[-1]
            if "num_slices" in result_dic.keys():
                info["num_slices"] = result_dic["num_slices"]
            for tag, value in info.items():
                tensorboard_logger.scalar_summary(tag, value, len(train_loader) * epoch + batch_id + 1)

            # empty cache
            if ("empty_cache_batch" in args.keys()) and args["empty_cache_batch"]:
                torch.cuda.empty_cache()
        # end for 1 epoch

        # calculate avg_train_loss of the epoch
        if len(rec_train_loss_list) > 0:
            avg_rec_train_loss = sum(rec_train_loss_list) / len(rec_train_loss_list)
        if len(reg_train_loss_list) > 0:
            avg_reg_train_loss = sum(reg_train_loss_list) / len(reg_train_loss_list)
        avg_train_loss = sum(train_loss_list) / len(train_loss_list)

        # evaluate on validation set
        if args["have_val_set"] and (epoch % args["epoch_gap_for_evaluation"] == 0):
            eval_value_list = []
            for batch_id, batch in tqdm(enumerate(val_loader)):
                val_data = batch.to(device)
                result_dic = Evaluator.evaluate(autoencoder, val_data, loss_func, **dic)
                eval_value_list.append(result_dic["evaluation"].item())
                # end for
            avg_eval_value = sum(eval_value_list) / len(eval_value_list)

        if not args["have_val_set"]:
            avg_eval_value = avg_train_loss

        # save checkpoint
        checkpoint_path = osp.join(logdir, "latest.pth")
        f_checkpoint_path = osp.join(logdir, "f_latest.pth")

        if args["use_scheduler"]:
            Saver.save_checkpoint(autoencoder, optimizer, checkpoint_path, scheduler=scheduler)
            if "amortized" in args["loss"]:
                Saver.save_checkpoint(f_slice, dic["f_optimizer"], f_checkpoint_path, f_scheduler=f_scheduler)
        else:
            Saver.save_checkpoint(autoencoder, optimizer, checkpoint_path)
            if "amortized" in args["loss"]:
                Saver.save_checkpoint(f_slice, dic["f_optimizer"], f_checkpoint_path)

        if epoch % args["epoch_gap_for_save"] == 0:
            checkpoint_path = os.path.join(logdir, "epoch_" + str(epoch) + ".pth")
            f_checkpoint_path = os.path.join(logdir, "f_epoch_" + str(epoch) + ".pth")
            Saver.save_best_weights(autoencoder, checkpoint_path)
            if "amortized" in args["loss"]:
                Saver.save_best_weights(f_slice, f_checkpoint_path)

        # save best model based on avg_eval_value
        if args["eval_criteria"] in ["jsd", "loss_func", "mmd"]:
            better = avg_eval_value < best_eval_value
        elif args["eval_criteria"] in ["cov"]:
            better = avg_eval_value > best_eval_value
        else:
            raise Exception("Unknown eval_criteria")
        if better:
            best_eval_value = avg_eval_value
            best_epoch = epoch
            Saver.save_best_weights(autoencoder, model_path)
            if "amortized" in args["loss"]:
                Saver.save_best_weights(f_slice, f_model_path)

        # save best model based on avg_train_loss
        if avg_train_loss < best_train_loss:
            best_train_loss = avg_train_loss
            best_epoch_based_on_train_loss = epoch
        if args["evaluator"] != "based_on_train_loss":
            Saver.save_best_weights(autoencoder, best_train_loss_model_path)
            if "amortized" in args["loss"]:
                Saver.save_best_weights(f_slice, f_best_train_loss_model_path)

        # report
        train_log = "Epoch {}| train_loss : {}\n".format(epoch, avg_train_loss)
        eval_log = "Epoch {}| eval_value : {}\n".format(epoch, avg_eval_value)
        eval_best_log = "Best epoch {}| best eval value: {}\n".format(best_epoch, best_eval_value)
        best_train_loss_log = "Best_train_loss epoch {}| best train loss : {}\n".format(
            best_epoch_based_on_train_loss, best_train_loss
        )
        with open(train_log_path, "a") as fp:
            fp.write(train_log)
        with open(eval_log_path, "a") as fp:
            fp.write(eval_log)
        with open(best_eval_log_path, "w") as fp:
            fp.write(eval_best_log)
        with open(best_train_log_path, "w") as fp:
            fp.write(best_train_loss_log)
        print(train_log)
        print(eval_log)
        print(eval_best_log)
        print(best_train_loss_log)

        if len(rec_train_loss_list) > 0:
            rec_train_log = "Epoch {}| rec_train_loss : {}\n".format(epoch, avg_rec_train_loss)
            with open(rec_train_log_path, "a") as fp:
                fp.write(rec_train_log)
            print(rec_train_log)

        if len(reg_train_loss_list) > 0:
            reg_train_log = "Epoch {}| reg_train_loss : {}\n".format(epoch, avg_reg_train_loss)
            with open(reg_train_log_path, "a") as fp:
                fp.write(reg_train_log)
            print(reg_train_log)

        if ("empty_cache_epoch" in args.keys()) and args["empty_cache_epoch"]:
            torch.cuda.empty_cache()
        print("---------------------------------------------------------------------------------------")
    # end for

    finish_time = time.time()
    total_runtime = finish_time - start_time
    total_runtime = datetime.timedelta(seconds=total_runtime)
    runtime_log = "total runtime: {}".format(str(total_runtime))
    print("total_runtime:", total_runtime)
    with open(train_log_path, "a") as fp:
        fp.write(runtime_log)

    print("Saved checkpoints and logs in: ", logdir)


if __name__ == "__main__":
    main()
