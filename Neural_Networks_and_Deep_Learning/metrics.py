import pypianoroll
import matplotlib.pyplot as plt
from pypianoroll.visualization import plot_pianoroll
from pypianoroll import Multitrack, Track, BinaryTrack
from math import nan
import math
from typing import Sequence, Tuple, List, Any, Union
import numpy as np
from numpy import ndarray
import pandas as pd
import matplotlib.ticker as mticker


def polyphony(pianoroll: ndarray) -> float:
    denominator = np.count_nonzero(pianoroll.sum(1) > 0)
    if denominator < 1:
        return nan
    return pianoroll.sum() / denominator


def _to_chroma(pianoroll: ndarray) -> ndarray:
    """Return the unnormalized chroma features."""
    reshaped = pianoroll[:, :120].reshape(-1, 10, 12)
    # reshaped[..., :8] += pianoroll[:, 120:].reshape(-1, 1, 8) useless since we do not consider notes above 120, it's just padding
    return np.sum(reshaped, 1)


def _get_scale(root: int, mode: str, wght: bool = False) -> ndarray:
    """Return the scale mask for a specific root."""
    if not wght:
        if mode == "major":
            a_scale_mask = np.array([0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1], bool)
        else:
            a_scale_mask = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1], bool)
    else:
        if mode == "major":
            a_scale_mask = np.array([0, 1.0, 0.5, 0, 1.0, 0, 0.5, 0, 0.5, 1.0, 0, 0.5], bool)
        else:
            a_scale_mask = np.array([1.0, 0, 0.5, 0, 1.0, 0.5, 0, 0.5, 0, 1.0, 0, 0.5], bool)
    return np.roll(a_scale_mask, root)


def in_scale_rate(pianoroll: ndarray, root: int, mode: str = "major", wght: bool = False) -> float:
    chroma = _to_chroma(pianoroll)
    scale_mask = _get_scale(root, mode, wght)
    if np.count_nonzero(pianoroll) < 1:
        return nan
    n_in_scale = np.sum(scale_mask.reshape(-1, 12) * chroma)
    return n_in_scale / np.count_nonzero(pianoroll)


def in_scale_rate_1(pianoroll: ndarray, root: int, mode: str = "major", wght: bool = False) -> float:
    scale = _get_scale(root, mode, wght)
    note_count = 0
    in_scale_count = 0

    pianoroll = pianoroll.astype(int)
    nonz = []

    for idx in range(pianoroll.shape[0]):
        nonz.append(np.where(pianoroll[idx] != 0)[0])

    nonz = np.concatenate(nonz, axis=0)

    for note in range(len(nonz)):
        note_count += 1
        note_temp = scale[nonz[note] % 12]
        in_scale_count += note_temp
    if note_count < 1:
        return nan
    return in_scale_count / note_count


def n_pitch_classes_used(pianoroll: ndarray) -> int:
    return np.count_nonzero(_to_chroma(pianoroll).any(0))


def _entropy(prob):
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.nansum(prob * np.log2(prob))


def pitch_class_entropy_1(pianoroll: ndarray) -> float:
    chroma = _to_chroma(pianoroll)
    counts = np.sum(chroma, 0)
    denominator = counts.sum()
    if denominator < 1:
        return nan
    prob = counts / denominator
    return _entropy(prob)


def pitch_class_entropy(pianoroll: ndarray) -> float:
    pianoroll = pianoroll.astype(int)
    nonz = []
    counter = np.zeros(12)
    for idx in range(pianoroll.shape[0]):
        nonz.append(np.where(pianoroll[idx] != 0)[0])
    nonz = np.concatenate(nonz, axis=0)
    for note in range(len(nonz)):
        counter[nonz[note] % 12] += 1
    denominator = counter.sum()
    if denominator < 1:
        return nan
    prob = counter / denominator
    return _entropy(prob)


def pitch_entropy(pianoroll: ndarray) -> float:
    pianoroll = pianoroll.astype(int)
    nonz = []
    counter = np.zeros(128)
    for idx in range(pianoroll.shape[0]):
        nonz.append(np.where(pianoroll[idx] != 0)[0])
    nonz = np.concatenate(nonz, axis=0)
    for note in range(len(nonz)):
        counter[nonz[note]] += 1
    # print("counter", counter)
    denominator = counter.sum()
    # print(denominator)
    if denominator < 1:
        return nan
    prob = counter / denominator
    return _entropy(prob)


def scale_consistency(pianoroll: ndarray, wght: bool = False) -> List[Union[float, int, str]]:
    max_in_scale_rate = 0.0
    max_root = 0
    max_mode = 'none'
    for mode in ("major", "minor"):
        for root in range(12):
            rate = in_scale_rate(pianoroll, root, mode, wght)
            if math.isnan(rate):
                return [nan, nan, nan]
            if rate > max_in_scale_rate:
                max_in_scale_rate = rate
                max_root = root
                max_mode = mode
    return [max_in_scale_rate, max_root, max_mode]


def drum_in_pattern_rate(pianoroll: ndarray, resolution: int, tolerance: float = 0.1) -> float:
    if resolution not in (4, 6, 8, 9, 12, 16, 18, 24):
        raise ValueError("Unsupported beat resolution. Expect 4, 6, 8 ,9, 12, 16, 18 or 24.")

    def _drum_pattern_mask(res, tol):
        """Return a drum pattern mask with the given tolerance."""
        if res == 24:
            drum_pattern_mask = np.tile([1.0, tol, 0.0, 0.0, 0.0, tol], 4)
        elif res == 12:
            drum_pattern_mask = np.tile([1.0, tol, tol], 4)
        elif res == 6:
            drum_pattern_mask = np.tile([1.0, tol, tol], 2)
        elif res == 18:
            drum_pattern_mask = np.tile([1.0, tol, 0.0, 0.0, 0.0, tol], 3)
        elif res == 9:
            drum_pattern_mask = np.tile([1.0, tol, tol], 3)
        elif res == 16:
            drum_pattern_mask = np.tile([1.0, tol, 0.0, tol], 4)
        elif res == 8:
            drum_pattern_mask = np.tile([1.0, tol], 4)
        elif res == 4:
            drum_pattern_mask = np.tile([1.0, tol], 2)
        return drum_pattern_mask

    drum_pattern_mask = _drum_pattern_mask(resolution, tolerance)
    n_in_pattern = np.sum(pianoroll * np.tile(drum_pattern_mask, 16).reshape(-1, 1))
    note_count = np.count_nonzero(pianoroll)
    if note_count < 1:
        return nan
    return n_in_pattern / note_count


def elementwise_subtract(a, b):
    if isinstance(a, (int, float, np.int64, np.float64)):
        return float(a - b)
    elif isinstance(a, tuple) and isinstance(b, tuple):
        return tuple(ai - bi for ai, bi in zip(a, b))
    elif isinstance(a, list) and isinstance(b, list):
        return [ai - bi if isinstance(ai, (int, float, np.int64, np.float64)) and isinstance(bi, (int, float, np.int64, np.float64)) else ai for ai, bi in zip(a, b)]
    else:
        return a


def multitrack_metrics_all(samples, metrics_params, wght=False, diff=False, samples2=None):
    programs = metrics_params['programs']
    is_drums = metrics_params['is drums']
    track_names = metrics_params['track names']
    lowest_pitch = metrics_params['lowest pitch']
    n_pitches = metrics_params['n pitches']
    beat_resolution = metrics_params['beat resolution']

    tr = []
    for idx, (program, is_drum, track_name) in enumerate(zip(programs, is_drums, track_names)):
        pianoroll = np.pad(
            samples[idx],
            ((0, 0), (lowest_pitch, 128 - lowest_pitch - n_pitches))
        )
        metrics = [track_name]
        # Calculate metrics based on the first dataset
        #print(scale_consistency(pianoroll))
        #print(scale_consistency(pianoroll)[:-1])
        mt = [
            pypianoroll.empty_beat_rate(pianoroll, beat_resolution),
            pypianoroll.n_pitches_used(pianoroll),
            n_pitch_classes_used(pianoroll),
            pypianoroll.pitch_range_tuple(pianoroll),
            pypianoroll.pitch_range(pianoroll),
            pypianoroll.polyphonic_rate(pianoroll=pianoroll, threshold=2),
            polyphony(pianoroll),
            scale_consistency(pianoroll) if not diff else scale_consistency(pianoroll)[:-1],
            pitch_class_entropy_1(pianoroll),
            pitch_entropy(pianoroll)
        ]

        if track_name == "Drums":
            mt.append(drum_in_pattern_rate(pianoroll, beat_resolution))

        if diff and samples2 is not None:
            # Calculate metrics based on the second dataset
            pianoroll2 = np.pad(
                samples2[idx],
                ((0, 0), (lowest_pitch, 128 - lowest_pitch - n_pitches))
            )

            mt2 = [
                pypianoroll.empty_beat_rate(pianoroll2, beat_resolution),
                pypianoroll.n_pitches_used(pianoroll2),
                n_pitch_classes_used(pianoroll2),
                pypianoroll.pitch_range_tuple(pianoroll2),
                pypianoroll.pitch_range(pianoroll2),
                pypianoroll.polyphonic_rate(pianoroll=pianoroll2, threshold=2),
                polyphony(pianoroll2),
                scale_consistency(pianoroll2)[:-1],
                pitch_class_entropy_1(pianoroll2),
                pitch_entropy(pianoroll2)
            ]

            if track_name == "Drums":
                mt2.append(drum_in_pattern_rate(pianoroll2, beat_resolution))

            # Compute differences between metrics
            mt1 = mt
            mt = [elementwise_subtract(metric1, metric2) for metric1, metric2 in zip(mt1, mt2)]

        metrics.extend(mt)
        tr.append(metrics)

    return tr


"""
def create_list_dfs(data, metrics_params, wght=False, diff=False, data2=None):
    if data2 is not None and np.all([diff, data.shape[0] == data2.shape[0]]):
        metric_lists = [multitrack_metrics_all(data[i], metrics_params, wght, diff, data2[i]) for i in
                        range(data.shape[0])]
    else:
        metric_lists = [multitrack_metrics_all(data[i], metrics_params, wght) for i in range(data.shape[0])]

    Drums, Piano, Guitar, Bass, Strings = ([m[0] for m in metric_lists], [m[1] for m in metric_lists],
                                           [m[2] for m in metric_lists], [m[3] for m in metric_lists],
                                           [m[4] for m in metric_lists])

    return [pd.DataFrame(Drums), pd.DataFrame(Piano), pd.DataFrame(Guitar), pd.DataFrame(Bass), pd.DataFrame(Strings)]
"""


def create_list_dfs(data, metrics_params, wght=False, diff=False, data2=None):
    
    n_tracks = metrics_params['n tracks']
    track_names = metrics_params['track names']

    if data2 is not None and np.all([diff, data.shape[0] == data2.shape[0]]):
        metric_lists = [multitrack_metrics_all(data[i], metrics_params, wght, diff, data2[i]) for i in
                        range(data.shape[0])]
    else:
        metric_lists = [multitrack_metrics_all(data[i], metrics_params, wght) for i in range(data.shape[0])]

    dfs = []

    for idx, track_name in enumerate(track_names):
        dfs.append(pd.DataFrame([m[idx] for m in metric_lists]))

    return dfs


def plot_histogram(data, data2, ax, title, xlabel, data2_provided, xticklabels=None, ints=False):
    # width = (max(data) - min(data) + 1) / 10 if not ints else 1.0
    
    ### new limits
    min_data = min(min(data), min(data2)) if data2_provided else min(data)
    max_data = max(max(data), max(data2)) if data2_provided else max(data)
    
    if not ints:
        ax.hist(data, bins=np.linspace(min_data - 0.5, max_data + 0.5, 10), alpha=0.5, label="Set", density = True)
    else:
        ax.hist(data, bins=np.linspace(min_data - 0.5, max_data + 1.5, int(max_data - min_data + 2 + 1)), alpha=0.5,
                label="Set", density = True)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("frequency")
    print("\n\n" + title + "\n", data.describe(), "\nmode ", data.mode()[0])

    if data2_provided:
        if not ints:
            ax.hist(data2, bins=np.linspace(min_data - 0.5, max_data + 0.5, 10), alpha=0.5, label="Set2", density = True)
        else:
            ax.hist(data2, bins=np.linspace(min_data - 0.5, max_data + 1.5, int(max_data - min_data + 2 + 1)),
                    alpha=0.5, label="Set2", density = True)
            print("\n\n" + title + ", generated \n", data2.describe(), "\nmode ", data2.mode()[0])

    if xticklabels:
        print("min and max:", min(data), max(data))
        ax.xaxis.set_major_locator(
            #mticker.FixedLocator(np.linspace(min_data, max_data + 1, int(max_data - min_data + 1 + 1))))
            
            ### new
            mticker.FixedLocator(np.arange(13)))
        
        # ax.set_xticks(np.linspace(min_data, max_data + 1, int(max_data - min_data + 1 + 1)))
        ax.set_xticklabels(xticklabels, minor=False, rotation=45)

    ax.legend()
    ax.grid()


def multitrack_metrics_plot(data, metrics_params, wght=False, data2=None, diff=False, data_to_diff=None):
    if data_to_diff is not None and np.all([diff, data.shape[0] == data_to_diff.shape[0]]):
        list_dfs = create_list_dfs(data, metrics_params, diff=True, data2=data_to_diff)
    else:
        list_dfs = create_list_dfs(data, metrics_params, wght)

    data2_provided = data2 is not None and data2.any()
    data2_list_dfs = create_list_dfs(data2, metrics_params, wght) if data2_provided else [None for _ in list_dfs]

    metrics_names = metrics_params['track names']
    metrics_indices = {name: idx for idx, name in enumerate(metrics_names)}

    for df, data2_df in zip(list_dfs, data2_list_dfs):
        metric_name = df[0][0]
        metric_idx = metrics_indices.get(metric_name)
        fig, ax = plt.subplots(4, 3, figsize=(16, 12))

        #if metric_idx == metrics_indices["Drums"]:
        if metric_name == "Drums":
            plot_histogram(df[11], data2_df[11] if data2_provided else None, ax[0, 2], "drums pattern",
                           "rate of drums in fixed pattern", data2_provided)

        plot_histogram(df[1], data2_df[1] if data2_provided else None, ax[0, 0], "non empty beat rate", "rate",
                       data2_provided)

        dff = df[df[1] != 0][1]
        data2_dff = data2_df[data2_df[1] != 0][1] if data2_provided else None

        plot_histogram(dff, data2_dff, ax[0, 1], "non empty beat rate, only non empty tracks", "rate", data2_provided)

        df = df[df.notna().all(axis=1)]
        data2_df = data2_df[data2_df.notna().all(axis=1)] if data2_provided else None

        plot_histogram(df[2], data2_df[2] if data2_provided else None, ax[1, 0], "pitches used",
                       "number of different pitches used", data2_provided)
        plot_histogram(df[3], data2_df[3] if data2_provided else None, ax[1, 1], "number of pitch classes used",
                       "number of pitch classes used", data2_provided, ints=True)
        plot_histogram(df[5], data2_df[5] if data2_provided else None, ax[1, 2], "pitch range", "pitch range",
                       data2_provided)
        plot_histogram(df[6], data2_df[6] if data2_provided else None, ax[2, 0],
                       "poliphony rate, threshold = 2",
                       "#number of timesteps in which threshold+1 \nor more pitches are playing together",
                       data2_provided)
        plot_histogram(df[7], data2_df[7] if data2_provided else None, ax[2, 1], "poliphony",
                       "poliphony, average number of pitches being played concurrently", data2_provided, ints=True)

        scales_data = pd.DataFrame({"scales": [ll[1] for ll in df[8]]})
        consistency_data = pd.DataFrame({"consistency": [ll[0] for ll in df[8]]})
        data2_scales_data = pd.DataFrame({"gen_scales": [ll[1] for ll in data2_df[8]]}) if data2_provided else None
        data2_consistency_data = pd.DataFrame(
            {"gen_consistency": [ll[0] for ll in data2_df[8]]}) if data2_provided else None

        scale_labels = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A"]

        if data_to_diff is not None and np.all([diff, data.shape[0] == data_to_diff.shape[0]]):
            plot_histogram(scales_data["scales"], data2_scales_data["gen_scales"] if data2_provided else None, ax[3, 0],
                           "scale consistency, scales", "key distance", data2_provided, ints=True)
        else:
            plot_histogram(scales_data["scales"], data2_scales_data["gen_scales"] if data2_provided else None, ax[3, 0],
                           "scale consistency, scales", "scale", data2_provided, xticklabels=scale_labels, ints=True)

        # plot_histogram(scales_data["scales"], data2_scales_data["gen_scales"] if data2_provided else None, ax[3, 0],
        # "scale consistency, scales", "scale", data2_provided, xticklabels=scale_labels, ints = True)

        plot_histogram(consistency_data["consistency"],
                       data2_consistency_data["gen_consistency"] if data2_provided else None, ax[2, 2],
                       "scale consistency, in scale rate", "in scale rate of notes for the best fitting scale",
                       data2_provided)

        plot_histogram(df[9], data2_df[9] if data2_provided else None, ax[3, 1], "pitch class entropy",
                       "Shannon entropy related to the number of pitch classes used", data2_provided)
        plot_histogram(df[10], data2_df[10] if data2_provided else None, ax[3, 2], "pitch entropy",
                       "Shannon entropy related to the number of pitches used", data2_provided)

        plt.tight_layout()
        plt.show()
        

def superimposed_metrics_plot(list_dfs, metrics_params, data2_list_dfs):

    metrics_names = metrics_params['track names']
    metrics_indices = {name: idx for idx, name in enumerate(metrics_names)}

    for df, data2_df in zip(list_dfs, data2_list_dfs):
        metric_name = df[0][0]
        metric_idx = metrics_indices.get(metric_name)
        fig, ax = plt.subplots(4, 3, figsize=(16, 12))

        if metric_idx == metrics_indices["Drums"]:
            plot_histogram(df[11], data2_df[11], ax[0, 2], "drums pattern",
                           "rate of drums in fixed pattern", True)

        plot_histogram(df[1], data2_df[1], ax[0, 0], "non empty beat rate", "rate",
                       True)

        dff = df[df[1] != 0][1]
        data2_dff = data2_df[data2_df[1] != 0][1]

        plot_histogram(dff, data2_dff, ax[0, 1], "non empty beat rate, only non empty tracks", "rate", True)

        df = df[df.notna().all(axis=1)]
        data2_df = data2_df[data2_df.notna().all(axis=1)]

        plot_histogram(df[2], data2_df[2], ax[1, 0], "pitches used",
                       "number of different pitches used", True)
        plot_histogram(df[3], data2_df[3], ax[1, 1], "number of pitch classes used",
                       "number of pitch classes used", True, ints=True)
        plot_histogram(df[5], data2_df[5], ax[1, 2], "pitch range", "pitch range",
                       True)
        plot_histogram(df[6], data2_df[6], ax[2, 0],
                       "polyphony rate, threshold = 2",
                       "#number of timesteps in which threshold+1 \nor more pitches are played together",
                       True)
        plot_histogram(df[7], data2_df[7], ax[2, 1], "polyphony",
                       "polyphony, average number of pitches being played concurrently", True, ints=True)

        scales_data = pd.DataFrame({"scales": [ll[1] for ll in df[8]]})
        consistency_data = pd.DataFrame({"consistency": [ll[0] for ll in df[8]]})
        data2_scales_data = pd.DataFrame({"gen_scales": [ll[1] for ll in data2_df[8]]})
        data2_consistency_data = pd.DataFrame(
            {"gen_consistency": [ll[0] for ll in data2_df[8]]})

        scale_labels = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A"]

        if data_to_diff is not None and np.all([diff, data.shape[0] == data_to_diff.shape[0]]):
            plot_histogram(scales_data["scales"], data2_scales_data["gen_scales"], ax[3, 0],
                           "scale consistency, scales", "key distance", True, ints=True)
        else:
            plot_histogram(scales_data["scales"], data2_scales_data["gen_scales"], ax[3, 0],
                           "scale consistency, scales", "scale", True, xticklabels=scale_labels, ints=True)

        # plot_histogram(scales_data["scales"], data2_scales_data["gen_scales"] if data2_provided else None, ax[3, 0],
        # "scale consistency, scales", "scale", data2_provided, xticklabels=scale_labels, ints = True)

        plot_histogram(consistency_data["consistency"],
                       data2_consistency_data["gen_consistency"], ax[2, 2],
                       "scale consistency, in scale rate", "in scale rate of notes for the best fitting scale",
                       True)

        plot_histogram(df[9], data2_df[9], ax[3, 1], "pitch class entropy",
                       "Shannon entropy related to the number of pitch classes used", True)
        plot_histogram(df[10], data2_df[10], ax[3, 2], "pitch entropy",
                       "Shannon entropy related to the number of pitches used", True)

        plt.tight_layout()
        plt.show()
