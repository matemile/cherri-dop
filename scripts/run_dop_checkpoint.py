"""Run one DOP prediction and export the runtime dynamic graphs."""

from __future__ import annotations

import argparse
import os
from contextlib import nullcontext
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import zarr
from matplotlib.collections import LineCollection

plt.switch_backend("Agg")


DATASET_FILES = {
    "microwave": "mars-odb-radiances-tabular-20200101-20220306-3h.zarr",
    "iasi": "metop-iasi-radiances-tabular-20200101-20220108-3h.zarr",
    "ascat": "metop-ascat-tabular-20200101-20220108-3h.zarr",
    "frost": "metno-frost-dwh-hourly-2020-2025-v2-tabular-1h-observations-lat-gt-20.zarr",
}


def checkpoint_file(path: Path) -> Path:
    if path.is_file():
        return path
    preferred = path / "inference-last.ckpt"
    if preferred.is_file():
        return preferred
    candidates = list(path.glob("inference-*.ckpt"))
    if not candidates:
        raise FileNotFoundError(f"No inference-*.ckpt found under {path}")
    return max(candidates, key=lambda item: item.stat().st_mtime)


def read_window(path: Path, epoch: int, variables: list[str], max_rows: int):
    store = zarr.open_group(str(path), mode="r")
    ranges = np.asarray(store["date_index_ranges"][:])
    selected = ranges[(ranges[:, 0] > epoch - 3 * 3600) & (ranges[:, 0] <= epoch)]
    if not len(selected):
        raise ValueError(f"No rows at {np.datetime64(epoch, 's')} in {path}")
    start, stop = int(selected[0, 1]), int(selected[-1, 1] + selected[-1, 2])
    rows = np.asarray(store["data"][start:stop])
    timedeltas = np.repeat(selected[:, 0] - epoch, selected[:, 2]).astype("float32")
    if len(rows) > max_rows:
        keep = np.arange(max_rows) * len(rows) // max_rows
        rows, timedeltas = rows[keep], timedeltas[keep]

    meta = list(store.attrs["meta_variables"])
    stored_variables = list(store.attrs["variables"])
    columns = [len(meta) + stored_variables.index(name) for name in variables]
    payload = {
        "data": torch.from_numpy(rows[:, columns].astype("float32", copy=False)),
        "latitudes": rows[:, meta.index("__latitude")],
        "longitudes": rows[:, meta.index("__longitude")],
        "timedeltas": timedeltas,
        "boundaries": [(0, len(rows))],
        "layout": ("grid", "variables"),
    }
    return payload, rows[:, len(meta) :], stored_variables


def plot_graph(graph, path: Path) -> None:
    edge_name = graph.edge_types[0]
    source_name, _, target_name = edge_name
    source = np.rad2deg(graph[source_name].x.numpy())
    target = np.rad2deg(graph[target_name].x.numpy())
    source[:, 1] = (source[:, 1] + 180) % 360 - 180
    target[:, 1] = (target[:, 1] + 180) % 360 - 180
    edge_index = graph[edge_name].edge_index
    edge_count = min(50_000, edge_index.shape[1])
    edge_sample = np.linspace(0, edge_index.shape[1] - 1, edge_count, dtype=int)
    edges = edge_index[:, edge_sample].numpy()
    segments = np.stack(
        [source[edges[0]][:, [1, 0]], target[edges[1]][:, [1, 0]]], axis=1
    )
    segments = segments[np.abs(segments[:, 0, 0] - segments[:, 1, 0]) <= 180]
    figure, axis = plt.subplots(figsize=(9, 7))
    axis.add_collection(
        LineCollection(segments, colors="0.5", alpha=0.04, linewidths=0.2)
    )
    axis.scatter(source[:, 1], source[:, 0], s=2, label=source_name)
    axis.scatter(target[:, 1], target[:, 0], s=5, label=target_name)
    axis.autoscale()
    axis.legend()
    axis.set(xlabel="Longitude", ylabel="Latitude", title=str(edge_name))
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--date", default="2020-01-01T12:00:00")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=os.environ.get("DOP_DATA_ROOT", "../../datasets"),
    )
    parser.add_argument("--output", type=Path, default=Path("dop_prediction"))
    parser.add_argument("--max-rows", type=int, default=32768)
    parser.add_argument("--lead-hours", type=int, default=12)
    args = parser.parse_args()

    checkpoint = checkpoint_file(args.checkpoint.expanduser())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if isinstance(model, dict):
        raise TypeError(f"{checkpoint} is a training checkpoint; use inference-*.ckpt")
    model = model.eval().to(device)
    core = model.model
    for providers in (core.encoder_graph_provider, core.decoder_graph_provider):
        for provider in providers.values():
            for attribute in getattr(provider, "attributes_config", {}).values():
                attribute.device = device
    epoch = int(np.datetime64(args.date).astype("datetime64[s]").astype(np.int64))
    output_epoch = epoch + args.lead_hours * 3600

    inputs, targets, truth = {}, {}, {}
    for name in model.data_indices:
        dataset_path = args.data_root / DATASET_FILES[name]
        input_names = list(model.data_indices[name].model.input.ordered_names)
        forcing_names = list(model._target_forcing_names(name))
        inputs[name], _, _ = read_window(
            dataset_path, epoch, input_names, args.max_rows
        )
        targets[name], values, stored_names = read_window(
            dataset_path, output_epoch, forcing_names, args.max_rows
        )
        truth[name] = (values, stored_names)
        inputs[name]["data"] = inputs[name]["data"].to(device)
        targets[name]["data"] = targets[name]["data"].to(device)

    hidden = core._graph_name_hidden
    providers = []
    for name, provider in core.encoder_graph_provider.items():
        if not hasattr(provider, "capture_next_graph"):
            continue
        if not hasattr(provider, "_capture_request"):
            provider._capture_request = None
            provider._captured_graph = None
        provider.capture_next_graph(f"{name}_input", hidden)
        providers.append((name, "encoder", provider))
    for name, provider in core.decoder_graph_provider.items():
        if not hasattr(provider, "capture_next_graph"):
            continue
        if not hasattr(provider, "_capture_request"):
            provider._capture_request = None
            provider._captured_graph = None
        provider.capture_next_graph(hidden, f"{name}_output")
        providers.append((name, "decoder", provider))

    autocast = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )
    with torch.inference_mode(), autocast:
        predictions = model.predict_step(inputs, targets)

    args.output.mkdir(parents=True, exist_ok=True)
    torch.save(predictions, args.output / "predictions.pt")
    torch.save(core._graph_data.cpu(), args.output / "static_graph.pt")
    for name, role, provider in providers:
        graph = provider.consume_captured_graph()
        if graph is None:
            print(f"No graph captured for {name}:{role}")
            continue
        torch.save(graph, args.output / f"{name}_{role}_graph.pt")
        plot_graph(graph, args.output / f"{name}_{role}_graph.png")

    for name, payload in predictions.items():
        values = payload["data"].detach().float().cpu().numpy()
        latitude = payload["latitudes"].detach().cpu().numpy()
        longitude = payload["longitudes"].detach().cpu().numpy()
        longitude = (longitude + 180) % 360 - 180
        for variable_index, variable in enumerate(payload["variables"]):
            expected = truth[name][0][:, truth[name][1].index(variable)]
            figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
            limits = np.nanpercentile(
                np.concatenate([expected, values[:, variable_index]]), [1, 99]
            )
            for axis, field, title in zip(
                axes,
                (expected, values[:, variable_index]),
                ("Truth", "Prediction"),
            ):
                points = axis.scatter(
                    longitude,
                    latitude,
                    c=field,
                    s=2,
                    vmin=limits[0],
                    vmax=limits[1],
                )
                figure.colorbar(points, ax=axis)
                axis.set(
                    title=f"{name}: {variable} — {title}",
                    xlabel="Longitude",
                    ylabel="Latitude",
                )
            figure.savefig(args.output / f"{name}_{variable}.png", dpi=150)
            plt.close(figure)

    print(f"Checkpoint: {checkpoint}")
    print(f"Prediction and runtime graphs: {args.output.resolve()}")


if __name__ == "__main__":
    main()
