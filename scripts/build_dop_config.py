#!/usr/bin/env python3
"""Build a richer-batch DOP config directly from native observation stores."""

from __future__ import annotations

import argparse
import copy
from importlib.resources import files
from pathlib import Path

import zarr
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict

SOURCES = {
    "microwave": {
        "filename": "mars-odb-radiances-tabular-20200101-20220306-3h.zarr",
        "prognostics": ["brightness_temperature"],
        "forcings": [
            "channel",
            "reportype",
            "satellite_identifier",
            "satellite_instrument",
            "scanline",
            "scanpos",
            "lsm",
            "seaice",
            "zenith",
            "azimuth",
            "datum_status",
        ],
    },
    "iasi": {
        "filename": "metop-iasi-radiances-tabular-20200101-20220108-3h.zarr",
        "prognostics": ["wavenumber_radiance"],
        "forcings": ["channel", "wavenumber_cm1"],
    },
    "ascat": {
        "filename": "metop-ascat-tabular-20200101-20220108-3h.zarr",
        "prognostics": ["wind_speed", "wind_dir"],
        "forcings": [
            "wvc_quality_flag",
            "ice_prob",
            "ice_age",
            "bs_distance",
            "wvc_index",
        ],
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--core",
        type=Path,
        help="optional anemoi-core checkout; defaults to the installed anemoi.training package",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("../../datasets"),
        help="directory containing the native MARS-ODB, IASI, and ASCAT stores",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("training_configs/dop.yaml")
    )
    parser.add_argument("--max-rows-per-window", type=int, default=32768)
    parser.add_argument(
        "--iasi-channel",
        type=int,
        action="append",
        help="optional repeatable IASI channel filter for a smaller proof run",
    )
    parser.add_argument("--hidden-resolution", type=int, default=5)
    parser.add_argument("--num-channels", type=int, default=128)
    parser.add_argument("--processor-layers", type=int, default=4)
    args = parser.parse_args()
    if args.max_rows_per_window <= 0:
        parser.error("--max-rows-per-window must be positive")

    data_root = args.data_root.resolve()
    for name, source in SOURCES.items():
        path = data_root / source["filename"]
        if not path.exists():
            raise FileNotFoundError(f"Missing {name} dataset: {path}")
        root = zarr.open(path, mode="r")
        expected = set(source["prognostics"] + source["forcings"])
        missing = expected - set(root.attrs["variables"])
        if missing:
            raise ValueError(
                f"{path} is missing variables: {', '.join(sorted(missing))}"
            )

    if args.core is None:
        config_dir = Path(str(files("anemoi.training") / "config")).resolve()
    else:
        config_dir = (args.core / "training/src/anemoi/training/config").resolve()
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        config = compose(config_name="multi", overrides=["model=gnn"])

    data_configs = {}
    loaders = {}
    scalers = {}
    losses = {}
    metrics = {}
    groups = {}
    metric_variables = {}
    for name, source in SOURCES.items():
        variables = source["prognostics"] + source["forcings"]
        data_configs[name] = {
            "forcing": source["forcings"],
            "diagnostic": [],
            "processors": {
                "imputer": {
                    "_target_": "anemoi.models.preprocessing.imputer.InputImputer",
                    "config": {"default": "mean"},
                },
                "normalizer": {
                    "_target_": "anemoi.models.preprocessing.normalizer.InputNormalizer",
                    "config": {"default": "mean-std"},
                },
            },
        }
        loaders[name] = {
            "dataset_config": {
                "dataset": f"${{system.input.{name}}}",
                "frequency": "${data.frequency}",
                "window": "(-3h,+0h]",
                "select": variables,
            },
            "start": "2020-01-01T00:00:00",
            "end": "2020-01-05T12:00:00",
            "max_rows_per_window": args.max_rows_per_window,
            "trajectory": None,
        }
        if name == "iasi" and args.iasi_channel:
            loaders[name]["row_filters"] = {"channel": args.iasi_channel}
        scalers[name] = {
            "general_variable": {
                "_target_": "anemoi.training.losses.scalers.GeneralVariableLossScaler",
                "weights": {"default": 1.0},
            },
        }
        losses[name] = {
            "_target_": "anemoi.training.losses.MSELoss",
            "scalers": ["general_variable"],
            "ignore_nans": True,
        }
        metrics[name] = {
            "mse": {
                "_target_": "anemoi.training.losses.MSELoss",
                "scalers": [],
                "ignore_nans": True,
            },
        }
        groups[name] = {"default": "sfc", "sfc": {"param": source["prognostics"]}}
        metric_variables[name] = source["prognostics"]

    config.data.frequency = "12h"
    config.data.resolution = "native-observations"
    config.data.datasets = OmegaConf.create(data_configs)
    for split in ("training", "validation", "test"):
        split_loaders = copy.deepcopy(loaders)
        for loader in split_loaders.values():
            if split == "validation":
                loader["start"] = "2020-01-06T00:00:00"
                loader["end"] = "2020-01-07T12:00:00"
            elif split == "test":
                loader["start"] = "2020-01-08T00:00:00"
                loader["end"] = "2020-01-09T12:00:00"
        config.dataloader[split].datasets = OmegaConf.create(split_loaders)
    config.dataloader.prefetch_factor = 1
    config.dataloader.persistent_workers = False
    config.dataloader.batch_size.training = 1
    config.dataloader.batch_size.validation = 1
    config.dataloader.batch_size.test = 1
    config.dataloader.limit_batches.training = None
    config.dataloader.limit_batches.validation = 2
    config.dataloader.limit_batches.test = 2
    config.dataloader.num_workers.training = 1
    config.dataloader.num_workers.validation = 1
    config.dataloader.num_workers.test = 1

    config.task = OmegaConf.create(
        {
            "_target_": "anemoi.training.tasks.OffsetForecaster",
            "input_offsets": ["0h"],
            "output_offsets": ["12h"],
            "rollout_shift": "12h",
            "rollout": {"start": 1, "epoch_increment": 0, "maximum": 1},
            "validation_rollout": 1,
        },
    )
    config.training.scalers = OmegaConf.create({"datasets": scalers})
    config.training.training_loss = OmegaConf.create({"datasets": losses})
    config.training.validation_metrics = OmegaConf.create({"datasets": metrics})
    config.training.variable_groups = OmegaConf.create({"datasets": groups})
    config.training.metrics = OmegaConf.create({"datasets": metric_variables})
    config.training.max_epochs = 1
    config.training.num_sanity_val_steps = 0
    config.training.precision = "bf16-mixed"

    config.model.num_channels = args.num_channels
    config.model.processor.num_layers = args.processor_layers
    config.model.processor.num_chunks = 4
    config.model.processor.gradient_checkpointing = True
    encoder_template = OmegaConf.to_container(config.model.encoders["0"], resolve=False)
    decoder_template = OmegaConf.to_container(config.model.decoders["0"], resolve=False)
    encoders = {}
    decoders = {}
    for name in SOURCES:
        encoder = copy.deepcopy(encoder_template)
        encoder["source_datasets"] = [name]
        encoder["dataset_fusing_strategy"] = "none"
        encoder["mapper"]["num_chunks"] = 4
        encoders[name] = encoder
        decoder = copy.deepcopy(decoder_template)
        decoder["target_datasets"] = [name]
        decoder["target_node_features"] = ["coordinates", "target_forcings"]
        decoder["mapper"]["num_chunks"] = 4
        decoders[name] = decoder
    config.model.encoders = OmegaConf.create(encoders)
    config.model.decoders = OmegaConf.create(decoders)
    config.model.residual = OmegaConf.create({"datasets": {}})
    config.model.output_mask = OmegaConf.create({"datasets": {}})
    config.model.node_trainable_parameters = OmegaConf.create({"hidden": 8})
    config.model.edge_trainable_parameters.data2hidden = 0
    config.model.edge_trainable_parameters.hidden2data = 0
    config.model.bounding = OmegaConf.create({"datasets": {}})

    graph_nodes = {
        name: {
            "node_builder": {
                "_target_": "anemoi.graphs.nodes.AnemoiDatasetNodes",
                "dataset": f"${{dataloader.training.datasets.{name}.dataset_config}}",
            },
            "attributes": {
                "timedelta": {
                    "_target_": "anemoi.graphs.nodes.attributes.Timedeltas",
                    "scale_seconds": 10800.0,
                },
            },
        }
        for name in SOURCES
    }
    graph_nodes["hidden"] = {
        "node_builder": {
            "_target_": "anemoi.graphs.nodes.TriNodes",
            "resolution": args.hidden_resolution,
        },
    }
    graph_edges = []
    for name in SOURCES:
        graph_edges.append(
            {
                "source_name": name,
                "target_name": "hidden",
                "edge_builders": [
                    {
                        "_target_": "anemoi.graphs.edges.ReversedKNNEdges",
                        "num_nearest_neighbours": 8,
                        "source_mask_attr_name": None,
                        "target_mask_attr_name": None,
                    },
                ],
                "attributes": "${graph.attributes.edges}",
            },
        )
    graph_edges.append(
        {
            "source_name": "hidden",
            "target_name": "hidden",
            "edge_builders": [
                {
                    "_target_": "anemoi.graphs.edges.MultiScaleEdges",
                    "x_hops": 1,
                    "scale_resolutions": "${graph.nodes.hidden.node_builder.resolution}",
                    "source_mask_attr_name": None,
                    "target_mask_attr_name": None,
                },
            ],
            "attributes": "${graph.attributes.edges}",
        },
    )
    for name in SOURCES:
        graph_edges.append(
            {
                "source_name": "hidden",
                "target_name": name,
                "edge_builders": [
                    {
                        "_target_": "anemoi.graphs.edges.KNNEdges",
                        "num_nearest_neighbours": 3,
                        "source_mask_attr_name": None,
                        "target_mask_attr_name": None,
                    },
                ],
                "attributes": {
                    "edge_length": "${graph.attributes.edges.edge_length}",
                    "edge_dirs": "${graph.attributes.edges.edge_dirs}",
                    "target_timedelta": {
                        "_target_": "anemoi.graphs.edges.attributes.Timedeltas",
                        "node_axis": "target",
                        "scale_seconds": 10800.0,
                    },
                },
            },
        )
    config.graph.overwrite = True
    config.graph.nodes = OmegaConf.create(graph_nodes)
    config.graph.edges = OmegaConf.create(graph_edges)
    config.graph.attributes.nodes = OmegaConf.create({})
    config.graph.attributes.edges = OmegaConf.create(
        {
            "edge_length": {
                "_target_": "anemoi.graphs.edges.attributes.EdgeLength",
                "norm": "unit-std",
            },
            "edge_dirs": {
                "_target_": "anemoi.graphs.edges.attributes.EdgeDirection",
                "norm": "unit-std",
            },
        },
    )

    config.diagnostics.plot.callbacks = []
    config.diagnostics.plot.datasets_to_plot = list(SOURCES)
    config.diagnostics.plot.parameters = []
    config.diagnostics.callbacks = []
    config.diagnostics.log.mlflow.enabled = False
    config.diagnostics.benchmark_profiler.memory.enabled = False
    config.diagnostics.benchmark_profiler.time.enabled = False
    config.diagnostics.benchmark_profiler.speed.enabled = False
    config.diagnostics.benchmark_profiler.system.enabled = False
    config.diagnostics.benchmark_profiler.model_summary.enabled = False
    config.diagnostics.benchmark_profiler.snapshot.enabled = False

    with open_dict(config.system.input):
        for name, source in SOURCES.items():
            config.system.input[name] = (
                f"${{oc.env:DOP_DATA_ROOT,../../datasets}}/{source['filename']}"
            )
        config.system.input.graph = "${system.output.root}/graph.pt"
    config.system.output.root = "${oc.env:SCRATCH,.}/cherri-dop/native-observations/"
    config.system.hardware.accelerator = "gpu"
    config.system.hardware.num_gpus_per_node = (
        "${oc.decode:${oc.env:SLURM_GPUS_PER_NODE,1}}"
    )
    config.system.hardware.num_nodes = "${oc.decode:${oc.env:SLURM_NNODES,1}}"
    config.system.hardware.num_gpus_per_model = "${system.hardware.num_gpus_per_node}"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, args.output, resolve=False)


if __name__ == "__main__":
    main()
