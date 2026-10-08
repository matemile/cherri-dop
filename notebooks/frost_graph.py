import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # FROST tabular observation graph

    Build and plot the graph for one FROST observation window using the graph
    settings in `training_configs/dop.yaml`: observation nodes, a triangular
    hidden mesh, reversed-KNN encoder edges, multiscale processor edges, and
    KNN decoder edges.

    The selected window is read directly from the tabular Zarr arrays. This
    avoids loading Earthkit, which requires Python's optional `sqlite3` module
    and is not available in this repository's HPC `.venv`. The observations
    come from the strict latitude greater than 20 degrees store created by
    `scripts/crop_tabular_zarr.py`, matching the training graph snippet.
    """)


@app.cell
def _():
    import os
    from pathlib import Path

    os.environ["ANEMOI_GRAPHS_FORCE_CPU"] = "1"

    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import zarr
    from anemoi.graphs.create import GraphCreator
    from anemoi.graphs.nodes import AnemoiDatasetNodes
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from torch_geometric.data import HeteroData

    repo_root = next(
        path
        for path in (Path.cwd(), *Path.cwd().parents)
        if (path / "training_configs/dop.yaml").is_file()
    )
    dop = OmegaConf.load(repo_root / "training_configs/dop.yaml")
    return (
        AnemoiDatasetNodes,
        GraphCreator,
        HeteroData,
        OmegaConf,
        Path,
        dop,
        instantiate,
        np,
        os,
        plt,
        repo_root,
        torch,
        zarr,
    )


@app.cell
def _(Path, dop, os, repo_root):
    frost_path = (
        Path(
            os.environ.get(
                "FROST_DATASET",
                repo_root
                / "../../datasets/metno-frost-dwh-hourly-2020-2025-v2-tabular-1h-observations-lat-gt-20.zarr",
            )
        )
        .expanduser()
        .resolve()
    )
    frequency = str(dop.data.frequency)
    window = "(-3h,+0h]"
    window_index = 1
    minimum_latitude = 20.0
    hidden_resolution = int(dop.graph.nodes.hidden.node_builder.resolution)
    graph_path = repo_root / "notebooks/frost_graph.pt"
    plot_path = repo_root / "notebooks/frost_graph.png"
    return (
        frequency,
        frost_path,
        graph_path,
        hidden_resolution,
        minimum_latitude,
        plot_path,
        window,
        window_index,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Read one observation window

    `frequency` sets the spacing between reference times. The default reference
    is 12 hours after the first timestamp, and `window` selects the three hourly
    groups in $(-3\,\mathrm{h}, 0\,\mathrm{h}]$. Only their one contiguous data
    slice is read from the pre-cropped 44-million-row store. The strict
    latitude condition is checked again before graph construction.
    """)


@app.cell
def _(frequency, frost_path, minimum_latitude, np, window, window_index, zarr):
    if not frost_path.is_dir():
        raise FileNotFoundError(
            f"FROST tabular Zarr store not found: {frost_path}. "
            "Set FROST_DATASET to override the default."
        )

    store = zarr.open(str(frost_path), mode="r")
    if store.attrs.get("layout") != "tabular":
        raise ValueError(f"Expected a tabular FROST store: {frost_path}")

    meta_variables = list(store.attrs["meta_variables"])
    latitude_index = meta_variables.index("__latitude")
    longitude_index = meta_variables.index("__longitude")
    date_index_ranges = np.asarray(store["date_index_ranges"][:])

    if not frequency.endswith("h"):
        raise ValueError(
            f"This notebook expects an hourly frequency, got {frequency!r}."
        )
    reference_epoch = (
        int(date_index_ranges[0, 0]) + window_index * int(frequency[:-1]) * 3600
    )
    selected_ranges = date_index_ranges[
        (date_index_ranges[:, 0] > reference_epoch - 3 * 3600)
        & (date_index_ranges[:, 0] <= reference_epoch)
    ]
    if len(selected_ranges) == 0:
        raise ValueError(
            f"No observations found for reference time {np.datetime64(reference_epoch, 's')}."
        )

    row_start = int(selected_ranges[0, 1])
    row_stop = int(selected_ranges[-1, 1] + selected_ranges[-1, 2])
    observations = np.asarray(store["data"][row_start:row_stop])
    timedeltas_array = np.repeat(
        selected_ranges[:, 0] - reference_epoch,
        selected_ranges[:, 2],
    )
    if observations.shape[0] != timedeltas_array.shape[0]:
        raise ValueError(
            "The selected FROST date ranges are not a contiguous data slice."
        )

    area_mask = np.isfinite(observations[:, latitude_index]) & (
        observations[:, latitude_index] > minimum_latitude
    )
    if not np.all(area_mask):
        raise ValueError(
            f"The pre-cropped store contains observations at or south of "
            f"{minimum_latitude:g} degrees."
        )
    observations = observations[area_mask]
    timedeltas_array = timedeltas_array[area_mask]
    latitudes = observations[:, latitude_index]
    longitudes = observations[:, longitude_index]
    reference_date = np.datetime64(reference_epoch, "s")
    print(f"FROST store: {frost_path}")
    print(f"Reference time: {reference_date}; window: {window}")
    print(
        f"Observation nodes north of {minimum_latitude:g} degrees: {len(observations):,}"
    )
    print(f"Variables: {store.attrs['variables']}")
    return latitudes, longitudes, reference_date, timedeltas_array


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Reuse the DOP graph settings

    The microwave branch supplies the same observation-to-hidden and
    hidden-to-observation edge declarations used by training. It is renamed to
    `frost`; the hidden mesh and hidden-to-hidden processor edges are retained.
    """)


@app.cell
def _(OmegaConf, dop, frequency, frost_path, hidden_resolution, window):
    template = OmegaConf.to_container(dop.graph, resolve=True)
    frost_node = template["nodes"]["microwave"]
    frost_node["node_builder"]["dataset"] = {
        "dataset": str(frost_path),
        "frequency": frequency,
        "window": window,
    }
    hidden_node = template["nodes"]["hidden"]
    hidden_node["node_builder"]["resolution"] = hidden_resolution
    edges = []
    for edge in template["edges"]:
        if {edge["source_name"], edge["target_name"]} <= {"microwave", "hidden"}:
            for key in ("source_name", "target_name"):
                if edge[key] == "microwave":
                    edge[key] = "frost"
            for builder in edge["edge_builders"]:
                if "scale_resolutions" in builder:
                    builder["scale_resolutions"] = hidden_resolution
            edges.append(edge)

    graph_config = OmegaConf.create(
        {
            "nodes": {"frost": frost_node, "hidden": hidden_node},
            "edges": edges,
        }
    )
    print(OmegaConf.to_yaml(graph_config))
    return (graph_config,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Build the graph

    Tabular observation coordinates vary by window, so the FROST nodes are
    registered from this window before `GraphCreator` adds the static hidden
    mesh and all spatial edges. Time offsets are stored in seconds and encoded
    with the configured Anemoi `Timedeltas` attribute.
    """)


@app.cell
def _(
    AnemoiDatasetNodes,
    GraphCreator,
    HeteroData,
    OmegaConf,
    graph_config,
    graph_path,
    instantiate,
    latitudes,
    longitudes,
    timedeltas_array,
    torch,
):
    node_builder = instantiate(graph_config.nodes.frost.node_builder, name="frost")
    assert isinstance(node_builder, AnemoiDatasetNodes)

    graph = HeteroData()
    graph["frost"].x = node_builder.reshape_coords(latitudes, longitudes).to(
        dtype=torch.float32,
        device="cpu",
    )
    graph["frost"].node_type = type(node_builder).__name__
    if not torch.isfinite(graph["frost"].x).all():
        raise ValueError("The selected window contains invalid coordinates.")

    timedeltas = torch.as_tensor(timedeltas_array, dtype=torch.float32)
    graph["frost"].timedeltas = timedeltas
    graph["frost"].timedelta = instantiate(
        graph_config.nodes.frost.attributes.timedelta
    ).compute(timedeltas)

    spatial_config = OmegaConf.create(
        OmegaConf.to_container(graph_config, resolve=True)
    )
    del spatial_config.nodes.frost
    creator = GraphCreator(spatial_config)
    graph = creator.update_graph(graph)
    graph = creator.clean(graph)
    graph.validate(raise_on_error=True)
    creator.save(graph, graph_path, overwrite=True)

    print(graph)
    print(f"Saved graph: {graph_path}")
    return graph, timedeltas


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plot the observation and hidden nodes

    FROST observations are coloured by their offset from the reference time.
    The hidden mesh is shown over the same geographic extent. Longitudes are
    wrapped to $[-180, 180)$ for plotting; graph coordinates retain Anemoi's
    radian convention.
    """)


@app.cell
def _(
    graph,
    hidden_resolution,
    minimum_latitude,
    np,
    plot_path,
    plt,
    reference_date,
    timedeltas,
    window,
):
    frost_coords = np.rad2deg(graph["frost"].x.cpu().numpy())
    hidden_coords = np.rad2deg(graph["hidden"].x.cpu().numpy())
    frost_lon = (frost_coords[:, 1] + 180) % 360 - 180
    hidden_lon = (hidden_coords[:, 1] + 180) % 360 - 180
    lat_min, lat_max = frost_coords[:, 0].min(), frost_coords[:, 0].max()
    lon_min, lon_max = frost_lon.min(), frost_lon.max()
    padding = 2.0

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), constrained_layout=True)
    points = axes[0].scatter(
        frost_lon,
        frost_coords[:, 0],
        c=timedeltas.numpy() / 3600,
        s=10,
        cmap="viridis",
        alpha=0.7,
        rasterized=True,
    )
    fig.colorbar(points, ax=axes[0], label="Time offset from reference (hours)")
    axes[0].set_title(
        f"FROST: {graph['frost'].num_nodes:,} observation nodes (latitude > {minimum_latitude:g}°)"
    )
    axes[1].scatter(hidden_lon, hidden_coords[:, 0], s=10, color="tab:orange")
    axes[1].scatter(frost_lon, frost_coords[:, 0], s=3, color="tab:blue", alpha=0.3)
    axes[1].set_title(f"Hidden mesh (resolution {hidden_resolution}) with FROST nodes")
    for ax in axes:
        ax.set_xlim(lon_min - padding, lon_max + padding)
        ax.set_ylim(max(-90, lat_min - padding), min(90, lat_max + padding))
        ax.set_xlabel("Longitude (degrees)")
        ax.set_ylabel("Latitude (degrees)")
        ax.grid(alpha=0.25)
    fig.suptitle(f"FROST observation graph — {reference_date} — {window}")
    fig.savefig(plot_path, dpi=150)
    print(f"Saved plot: {plot_path}")
    plt.show()


if __name__ == "__main__":
    app.run()
