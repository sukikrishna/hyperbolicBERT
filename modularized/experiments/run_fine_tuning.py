"""
Script for fine-tuning BERT with hyperbolic loss functions.

This script implements an experiment to fine-tune BERT models using 
hyperbolic geometry and track how hyperbolicity evolves during training.
"""

import os
import time
import torch
import logging
from transformers import BertTokenizer

from models.probes import HyperbolicDistanceProbe
from data.processors import UDTreebankProcessor, GlueProcessor, create_dataloader
from data.datasets import SyntaxDataset, JointDataset
from fine_tuning import (
    HyperbolicProbeFineTuner,
    HyperbolicBertClassifier,
    save_model
)

# Set up logging
logger = logging.getLogger(__name__)

def setup_fine_tuning(config, args):
    experiment_config = {
        "experiment_name": config.get("experiment_name", "hyperbolic-bert-finetuning"),
        "output_dir": os.path.join(args.output_dir, "fine_tuning"),
        "task_type": config.get("task_type", "classification"),
        "task_name": config.get("task_name", "sst2"),
        "model_name": args.model,
        "num_epochs": config.get("num_epochs", 3),
        "batch_size": config.get("batch_size", 32),
        "learning_rate": config.get("learning_rate", 2e-5),
        "weight_decay": config.get("weight_decay", 0.01),
        "warmup_steps": config.get("warmup_steps", 0),
        "max_seq_length": config.get("max_seq_length", 128),
        "num_labels": config.get("num_labels", 2),
        "probe_hidden_size": config.get("probe_hidden_size", 256),
        "probe_dropout": config.get("probe_dropout", 0.1),
        "hyperbolic_curvature": config.get("hyperbolic_curvature", -1.0),
        "use_riemannian_optimizer": config.get("use_riemannian_optimizer", True),
        "seed": args.seed,
        "device": args.device,
        "max_examples": config.get("max_examples", 1000),
        "analyze_hyperbolicity": config.get("analyze_hyperbolicity", True),
        "save_model": config.get("save_model", True),
        "data_dir": args.data_dir,
        "glue_task": config.get("glue_task", "sst2")
    }

    os.makedirs(experiment_config["output_dir"], exist_ok=True)
    return experiment_config

def load_dataset(config):
    task_type = config["task_type"]
    task_name = config["task_name"]
    model_name = config["model_name"]
    max_seq_length = config["max_seq_length"]
    max_examples = config["max_examples"]
    data_dir = config["data_dir"]

    tokenizer = BertTokenizer.from_pretrained(model_name)

    if task_type in ["probe", "joint"]:
        ud_path = os.path.join(data_dir, "ud_treebanks")
        language = task_name.split("-")[1] if "-" in task_name else "en"
        ud_processor = UDTreebankProcessor(base_path=ud_path, language=language, split="train")
        ud_data = ud_processor.get_dataset(max_sentences=max_examples)

        train_sentences = ud_data["sentences"][:int(0.8 * len(ud_data["sentences"]))]
        train_trees = ud_data["dependency_trees"][:int(0.8 * len(ud_data["dependency_trees"]))]
        val_sentences = ud_data["sentences"][int(0.8 * len(ud_data["sentences"])):]
        val_trees = ud_data["dependency_trees"][int(0.8 * len(ud_data["dependency_trees"])):]

        if task_type == "probe":
            train_dataset = SyntaxDataset(train_sentences, train_trees, tokenizer, max_length=max_seq_length)
            val_dataset = SyntaxDataset(val_sentences, val_trees, tokenizer, max_length=max_seq_length)
        else:
            glue_path = os.path.join(data_dir, "glue")
            glue_processor = GlueProcessor(base_path=glue_path, task_name=config["glue_task"], split="train")
            glue_data = glue_processor.get_dataset(max_examples=max_examples)

            train_sem_sentences = glue_data["sentences"][:len(train_sentences)]
            train_sem_labels = glue_data["labels"][:len(train_sentences)]
            val_sem_sentences = glue_data["sentences"][:len(val_sentences)]
            val_sem_labels = glue_data["labels"][:len(val_sentences)]

            train_dataset = JointDataset(
                sentences=train_sentences + train_sem_sentences,
                dependency_trees=train_trees + [None] * len(train_sem_sentences),
                semantic_labels=[-1] * len(train_sentences) + train_sem_labels,
                tokenizer=tokenizer,
                max_length=max_seq_length
            )

            val_dataset = JointDataset(
                sentences=val_sentences + val_sem_sentences,
                dependency_trees=val_trees + [None] * len(val_sem_sentences),
                semantic_labels=[-1] * len(val_sentences) + val_sem_labels,
                tokenizer=tokenizer,
                max_length=max_seq_length
            )
    else:
        raise NotImplementedError("Only 'probe' and 'joint' task types are currently supported.")

    return train_dataset, val_dataset, tokenizer

def run_fine_tuning(model, tokenizer, config, args):
    config = setup_fine_tuning(config, args)
    train_dataset, val_dataset, tokenizer = load_dataset(config)

    train_loader = create_dataloader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = create_dataloader(val_dataset, batch_size=config["batch_size"], shuffle=False)

    if config["task_type"] == "probe":
        probe = HyperbolicDistanceProbe(
            input_dim=model.config.hidden_size,
            probe_rank=config["probe_hidden_size"],
            curvature=config["hyperbolic_curvature"],
            device=config["device"]
        ).to(config["device"])

        optimizer = torch.optim.Adam(probe.parameters(), lr=config["learning_rate"])

        trainer = HyperbolicProbeFineTuner(
            model=model,
            probe=probe,
            optimizer=optimizer,
            scheduler=None,
            device=config["device"],
            curvature=config["hyperbolic_curvature"]
        )

        trainer.train(train_loader, val_loader, num_epochs=config["num_epochs"])

        if config["save_model"]:
            model_path = os.path.join(config["output_dir"], f"{config['experiment_name']}_probe.pt")
            torch.save(probe.state_dict(), model_path)
            print(f"Saved trained probe to {model_path}")

    else:
        raise NotImplementedError("Classification and joint training not implemented yet.")
