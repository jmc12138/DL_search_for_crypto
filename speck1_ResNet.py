import pandas as pd
import torch.utils.data as data
import torch, os, tqdm
import numpy as np
import torch.nn as nn

# import nni
import argparse
import os

# import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import datasets
import models
import utils
from sklearn.metrics import confusion_matrix


def main():


    max_va = 0
    now_epoch = 1

    writer = SummaryWriter(rf"statics/Summarys/{opt.model}_Summary")
    train_loader,test_loader,weight = utils.get_dataLoader_weight(opt)
    model = models.make(opt.model, **opt.model_args)
    # model = models.make(opt.model)


    model = model.to(opt.device)
    if opt._parallel:
        model = nn.DataParallel(model)

    utils.log("num params: {}".format(utils.compute_n_params(model)))


    criterion = torch.nn.CrossEntropyLoss(
        weight=weight.to(opt.device)
    )
    optimizer, lr_scheduler = utils.make_optimizer(
        model.parameters(), opt.optimizer, opt.lr_scheduler, **opt.optimizer_args
    )
    utils.log(f"optimizer:{opt.optimizer},lr_scheduler:{opt.lr_scheduler}")

    if opt.is_load:
        now_epoch, max_va, model, optimizer, lr_scheduler = utils.load_last_model_state(
            opt.load_path, opt.device, model, optimizer, lr_scheduler
        )
        utils.log(f"加载模型完毕，epoch:{now_epoch}")



    utils.log("开始训练")

    timer_used = utils.Timer()
    timer_epoch = utils.Timer()

    for epoch in range(now_epoch, opt.max_epoch+1):
        train_summary_loss = utils.AverageMeter()
        train_total_score = utils.F1Meter()
        test_summary_loss = utils.AverageMeter()
        test_total_score = utils.F1Meter()
        # train
        model.train()

        for data, label in tqdm(train_loader, desc="train", leave=False):
            data, label = data.to(opt.device), label.to(opt.device)

            optimizer.zero_grad()
            logits = model(data)
            loss = criterion(logits, label)

            loss.backward()

            optimizer.step()
            pred = torch.max(logits.data, dim=1)[1]
            matrix = confusion_matrix(
                label.cpu().detach().numpy(), pred.cpu().detach().numpy(), labels=[1, 0]
            )
            train_total_score.update(matrix)
            train_summary_loss.update(loss.detach().item(), opt.batch_size)

            del data
            del label

        # eval

        model.eval()
        for data, label in tqdm(test_loader, desc="test", leave=False):
            data, label = data.to(opt.device), label.to(opt.device)
            with torch.no_grad():
                logits = model(data)

                loss = criterion(logits, label)

            pred = torch.max(logits.data, dim=1)[1]
            matrix = confusion_matrix(
                label.cpu().detach().numpy(), pred.cpu().detach().numpy(), labels=[1, 0]
            )
            test_total_score.update(matrix)
            test_summary_loss.update(loss.detach().item(), opt.batch_size)

        if lr_scheduler is not None:
            lr_scheduler.step()

        t_epoch = utils.time_str(timer_epoch.t())
        t_used = utils.time_str(timer_used.t())
        t_estimate = utils.time_str(timer_used.t() / epoch * opt.max_epoch)

        if epoch <= opt.max_epoch:
            epoch_str = str(epoch)
        else:
            epoch_str = "ex"

        # 在tensorboard中画图
        _ = ["loss", "acc", "precision", "recall", "f1_score", "tpr", "tnr"]
        train_total_score.cal()
        test_total_score.cal()
        utils.writer_scalars(
            _,
            [
                train_summary_loss.avg,
                train_total_score.accuracy,
                train_total_score.precision,
                train_total_score.recall,
                train_total_score.f1,
                train_total_score.tpr,
                train_total_score.tnr,
                train_total_score.tn, 
                train_total_score.fp, 
                train_total_score.fn, 
                train_total_score.tp
            ],
            "train",
            writer,
            epoch,
        )
        utils.writer_scalars(
            _,
            [
               test_summary_loss.avg,
                test_total_score.accuracy,
                test_total_score.precision,
                test_total_score.recall,
                test_total_score.f1,
                test_total_score.tpr,
                test_total_score.tnr,
                test_total_score.tn, 
                test_total_score.fp, 
                test_total_score.fn, 
                test_total_score.tp
            ],
            "test",
            writer,
            epoch,
        )
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        log_str = "epoch {}, train {:.4f}|{:.4f}|{:.4f}".format(
            epoch_str,
            train_summary_loss.avg,
            train_total_score.accuracy,
            test_total_score.accuracy,
        )

        utils.log(
            log_str + " , " f"t_epoch:{t_epoch},t_used:{t_used},t_estimate:{t_estimate}"
        )

        if opt._parallel:
            model_ = model.module
        else:
            model_ = model
        if opt.is_save:
            save_obj = {
                "file": __file__,
                "epoch": epoch,
                "model": opt.model,
                "model_args": opt.model_args,
                "model_sd": model_.state_dict(),
                "optimizer": opt.optimizer,
                "optimizer_args": opt.optimizer_args,
                "optimizer_sd": optimizer.state_dict(),
                "lr_sd": lr_scheduler.state_dict() if lr_scheduler != None else None,
                "va": max_va
                if max_va > test_total_score.accuracy
                else test_total_score.accuracy,
            }
            if not os.path.exists(opt.save_path):
                os.mkdir(opt.save_path)

            if epoch <= opt.max_epoch:
                torch.save(save_obj, os.path.join(opt.save_path, "epoch-last.pth"))

                if test_total_score.accuracy > max_va:
                    max_va = test_total_score.accuracy
                    torch.save(save_obj, os.path.join(opt.save_path, "max-va.pth"))

                if (opt.save_epoch is not None) and epoch % opt.save_epoch == 0:
                    torch.save(
                        save_obj, os.path.join(opt.save_path, "epoch-{}.pth".format(epoch))
                    )
            else:
                torch.save(save_obj, os.path.join(opt.save_path, "epoch-ex.pth"))

        writer.flush()
        # nni.report_final_result(test_total_score.accuracy)


if __name__ == "__main__":
    model = "BLSTM_speck1"
    args = {
        "dataset_name": "speck1_1_2e5_10",
        "log_path": os.path.abspath("statics/save/"),
        "save_path": os.path.abspath(f"statics/save/{model}_pth"),
        "max_epoch": 150,
        "save_epoch": 10,
        "is_save" : False,
        "batch_size": 64*2,
        "load_path": os.path.abspath(f"statics/save/{model}_pth/epoch-last.pth"),
        "is_load": False,
        "model": model,
        "optimizer": "Adamax",
        "lr": 0.01,
        "lr_scheduler": "None",
        "num_workers": 0,
        "gpu": 2,
        "device": utils.set_device(),
        "_parallel": False,
        "dropout": 0.5,
        "hidden_size": 1000,
        "num_layers": 1
    }

    RECEIVED_PARAMS = {
    "lr": 0.0012368514346040292,
    "hidden_size": 868,
    "optimizer": "AdaMod",
    "num_layers": 2
}
    

    GPU_PARAMS = {
        "gpu": 2,
        "device": "cuda",
        "_parallel": False,
    }

    args.update(RECEIVED_PARAMS)
    args.update(GPU_PARAMS)
    args["model_args"] = dict(dropout=args["dropout"], hidden_size=args["hidden_size"],num_layers=args['num_layers'])
    args["optimizer_args"] = dict(lr=args["lr"], weight_decay=None, milestones=[30, 80])

    opt = argparse.Namespace(**args)
    utils.log("参数如下：")
    utils.log(opt.model_args)
    utils.log(opt.optimizer_args)

    utils.set_gpu(opt.gpu)
    utils.set_log_path(opt.log_path)
    utils.log(f"{opt.device} is used")
    utils.NCCL_log()


    print(utils.get_device_msg())

    main()
