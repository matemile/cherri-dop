"""Create a latitude-cropped copy of an Anemoi tabular Zarr dataset."""

from __future__ import annotations

import argparse
import datetime
import os
import shutil
import uuid
from pathlib import Path

import numpy as np
import zarr


def create_cropped_store(
    source_path: Path, target_path: Path, minimum_latitude: float
) -> int:
    source = zarr.open_group(str(source_path), mode="r")
    if source.attrs.get("layout") != "tabular":
        raise ValueError(f"source is not an Anemoi tabular dataset: {source_path}")

    meta_variables = list(source.attrs["meta_variables"])
    variables = list(source.attrs["variables"])
    latitude_column = meta_variables.index("__latitude")
    variable_start = len(meta_variables)
    source_data = source["data"]
    date_index_ranges = np.asarray(source["date_index_ranges"][:])
    range_starts = date_index_ranges[:, 1]
    range_ends = range_starts + date_index_ranges[:, 2]
    if (
        range_starts[0] != 0
        or range_ends[-1] != source_data.shape[0]
        or not np.array_equal(range_starts[1:], range_ends[:-1])
    ):
        raise ValueError("date_index_ranges does not describe contiguous rows")

    statistics_start = source.attrs.get("statistics_start_date")
    statistics_end = source.attrs.get("statistics_end_date")
    statistics_start_epoch = (
        int(np.datetime64(statistics_start).astype("datetime64[s]").astype(np.int64))
        if statistics_start is not None
        else int(date_index_ranges[0, 0])
    )
    statistics_end_epoch = (
        int(np.datetime64(statistics_end).astype("datetime64[s]").astype(np.int64))
        if statistics_end is not None
        else int(date_index_ranges[-1, 0])
    )
    statistics_ranges = np.flatnonzero(
        (date_index_ranges[:, 0] >= statistics_start_epoch)
        & (date_index_ranges[:, 0] <= statistics_end_epoch)
    )
    if statistics_ranges.size == 0:
        raise ValueError("the configured statistics period has no rows")
    statistics_row_start = int(range_starts[statistics_ranges[0]])
    statistics_row_stop = int(range_ends[statistics_ranges[-1]])

    target = zarr.open_group(str(target_path), mode="w")
    target.attrs.update(dict(source.attrs))
    target.attrs["_tabular_finalise_complete"] = False

    target_data = target.create_dataset(
        "data",
        shape=(0, source_data.shape[1]),
        chunks=source_data.chunks,
        dtype=source_data.dtype,
        compressor=source_data.compressor,
        filters=source_data.filters,
        fill_value=source_data.fill_value,
        order=source_data.order,
    )
    target_data.attrs.update(dict(source_data.attrs))

    retained_counts = np.zeros(len(date_index_ranges), dtype=np.int64)
    statistics_count = np.zeros(len(variables), dtype=np.int64)
    statistics_mean = np.zeros(len(variables), dtype=np.float64)
    statistics_m2 = np.zeros(len(variables), dtype=np.float64)
    statistics_minimum = np.full(len(variables), np.inf, dtype=np.float64)
    statistics_maximum = np.full(len(variables), -np.inf, dtype=np.float64)

    target_offset = 0
    block_rows = source_data.chunks[0]
    threshold = np.float32(minimum_latitude)
    for row_start in range(0, source_data.shape[0], block_rows):
        row_stop = min(row_start + block_rows, source_data.shape[0])
        block = np.asarray(source_data[row_start:row_stop])
        latitudes = block[:, latitude_column]
        keep = np.isfinite(latitudes) & (latitudes > threshold)
        retained = block[keep]
        next_target_offset = target_offset + len(retained)
        target_data.resize((next_target_offset, source_data.shape[1]))
        if len(retained):
            target_data[target_offset:next_target_offset] = retained
        del retained

        first_range = int(np.searchsorted(range_ends, row_start, side="right"))
        last_range = int(np.searchsorted(range_starts, row_stop, side="left"))
        for range_index in range(first_range, last_range):
            local_start = max(int(range_starts[range_index]), row_start) - row_start
            local_stop = min(int(range_ends[range_index]), row_stop) - row_start
            retained_counts[range_index] += np.count_nonzero(
                keep[local_start:local_stop]
            )

        stats_start = max(row_start, statistics_row_start)
        stats_stop = min(row_stop, statistics_row_stop)
        if stats_start < stats_stop:
            local_start = stats_start - row_start
            local_stop = stats_stop - row_start
            statistics_keep = keep[local_start:local_stop]
            for variable_index in range(len(variables)):
                valid_values = block[
                    local_start:local_stop, variable_start + variable_index
                ][statistics_keep].astype(np.float64)
                valid_values = valid_values[~np.isnan(valid_values)]
                if valid_values.size == 0:
                    continue
                batch_count = valid_values.size
                batch_mean = np.mean(valid_values)
                batch_m2 = np.sum((valid_values - batch_mean) ** 2)
                previous_count = statistics_count[variable_index]
                combined_count = previous_count + batch_count
                delta = batch_mean - statistics_mean[variable_index]
                statistics_mean[variable_index] += delta * batch_count / combined_count
                statistics_m2[variable_index] += (
                    batch_m2 + delta**2 * previous_count * batch_count / combined_count
                )
                statistics_count[variable_index] = combined_count
                statistics_minimum[variable_index] = min(
                    statistics_minimum[variable_index], np.min(valid_values)
                )
                statistics_maximum[variable_index] = max(
                    statistics_maximum[variable_index], np.max(valid_values)
                )

        target_offset = next_target_offset
        print(
            f"Read {row_stop:,}/{source_data.shape[0]:,} rows; retained {target_offset:,}",
            flush=True,
        )

    nonempty_ranges = retained_counts > 0
    cropped_ranges = date_index_ranges[nonempty_ranges].copy()
    cropped_counts = retained_counts[nonempty_ranges]
    if not len(cropped_ranges):
        raise ValueError(f"no rows have latitude > {minimum_latitude:g}")
    cropped_ranges[:, 1] = 0
    cropped_ranges[1:, 1] = np.cumsum(cropped_counts[:-1])
    cropped_ranges[:, 2] = cropped_counts
    source_ranges = source["date_index_ranges"]
    target_ranges = target.create_dataset(
        "date_index_ranges",
        data=cropped_ranges,
        chunks=source_ranges.chunks,
        compressor=source_ranges.compressor,
        filters=source_ranges.filters,
        fill_value=source_ranges.fill_value,
        order=source_ranges.order,
    )
    target_ranges.attrs.update(dict(source_ranges.attrs))

    if np.any(statistics_count == 0):
        missing = [variables[index] for index in np.flatnonzero(statistics_count == 0)]
        raise ValueError(f"no non-NaN values were retained for statistics: {missing}")
    statistics = {
        "mean": statistics_mean,
        "minimum": statistics_minimum,
        "maximum": statistics_maximum,
        "stdev": np.sqrt(statistics_m2 / statistics_count),
    }
    for name, values in statistics.items():
        source_array = source[name]
        target_array = target.create_dataset(
            name,
            data=values.astype(source_array.dtype, copy=False),
            chunks=source_array.chunks,
            compressor=source_array.compressor,
            filters=source_array.filters,
            fill_value=source_array.fill_value,
            order=source_array.order,
        )
        target_array.attrs.update(dict(source_array.attrs))

    replaced_arrays = {"data", "date_index_ranges", *statistics}
    for name in source.array_keys():
        if name in replaced_arrays:
            continue
        source_array = source[name]
        target_array = target.create_dataset(
            name,
            data=source_array[:],
            chunks=source_array.chunks,
            compressor=source_array.compressor,
            filters=source_array.filters,
            fill_value=source_array.fill_value,
            order=source_array.order,
        )
        target_array.attrs.update(dict(source_array.attrs))

    first_date = datetime.datetime.fromtimestamp(
        int(cropped_ranges[0, 0]), datetime.UTC
    )
    last_date = datetime.datetime.fromtimestamp(
        int(cropped_ranges[-1, 0]), datetime.UTC
    )
    previous_uuids = list(source.attrs.get("previous_uuids") or [])
    source_uuid = source.attrs.get("uuid")
    if source_uuid is not None and source_uuid not in previous_uuids:
        previous_uuids.append(source_uuid)
    target.attrs.update(
        {
            "chunks": list(target_data.chunks),
            "crop": {"latitude": {"operator": ">", "value": minimum_latitude}},
            "cropped_from": {"path": str(source_path), "uuid": source_uuid},
            "description": (
                f"{source.attrs.get('description', '')} "
                f"Cropped to latitude > {minimum_latitude:g} degrees."
            ).strip(),
            "index_end_date": last_date.strftime("%Y-%m-%dT%H:%M:%S"),
            "index_length": len(cropped_ranges),
            "index_start_date": first_date.strftime("%Y-%m-%dT%H:%M:%S"),
            "last_date": last_date.strftime("%Y-%m-%d %H:%M:%S"),
            "latest_write_timestamp": datetime.datetime.now(datetime.UTC)
            .replace(tzinfo=None)
            .isoformat(),
            "previous_uuids": previous_uuids,
            "shape": list(target_data.shape),
            "start_date": first_date.strftime("%Y-%m-%d %H:%M:%S"),
            "uuid": str(uuid.uuid4()),
        }
    )

    if int(cropped_ranges[:, 2].sum()) != target_data.shape[0]:
        raise RuntimeError("cropped date index does not match the data length")
    if not np.all(np.diff(cropped_ranges[:, 0]) > 0):
        raise RuntimeError("cropped date index epochs are not increasing")
    if not np.all(cropped_ranges[:, 2] > 0):
        raise RuntimeError("cropped date index contains an empty range")

    from anemoi.datasets.create.tabular.validate import validate_date_ranges

    validate_date_ranges(target_data, cropped_ranges)

    target.attrs["_tabular_finalise_complete"] = True
    target.attrs["total_number_of_files"] = sum(
        len(filenames) for _, _, filenames in os.walk(target_path)
    )
    for _ in range(3):
        target.attrs["total_size"] = sum(
            Path(directory, filename).stat().st_size
            for directory, _, filenames in os.walk(target_path)
            for filename in filenames
        )
    return target_data.shape[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument("--minimum-latitude", type=float, default=20.0)
    args = parser.parse_args()

    source_path = args.source.expanduser().resolve()
    target_path = args.target.expanduser().resolve()
    if not source_path.is_dir():
        parser.error(f"source is not a Zarr directory: {source_path}")
    if target_path.exists():
        parser.error(f"target already exists: {target_path}")

    target_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = target_path.with_name(
        f".{target_path.name}.{uuid.uuid4().hex}.tmp"
    )
    try:
        row_count = create_cropped_store(
            source_path, temporary_path, args.minimum_latitude
        )
        temporary_path.rename(target_path)
    except BaseException:
        shutil.rmtree(temporary_path, ignore_errors=True)
        raise

    print(f"Created {target_path} with {row_count:,} rows")


if __name__ == "__main__":
    main()
