import numpy as np
import mir_eval
import os
from .song_data_structure import McpaMusic
import json

def _cum_time_to_time_dur(d):
    d_cumsum = np.cumsum(d)
    # it's just fuck insert 0 at first and take the same length as d_cumsum
    starts = np.insert(d_cumsum, 0, 0)[0: -1]
    return starts

'''
def _parse_phrase_label(phrase_string):
    # input: i4A4B8A4A4b4B8A4A4b4b4A4A4b4A4o3
    # output: (['i4', 'A4', 'B8', 'A4', 'A4', 'b4', 'B8', 'A4', 'A4', 'b4', 'b4', 'A4', 'A4', 'b4', 'A4', 'o3'], ['i', 'A', 'B', 'A', 'A', 'b', 'B', 'A', 'A', 'b', 'b', 'A', 'A', 'b', 'A', 'o'], array([4, 4, 8, 4, 4, 4, 8, 4, 4, 4, 4, 4, 4, 4, 4, 3]))
    phrase_starts = [i for i, s in enumerate(phrase_string) if s.isalpha()] + [len(phrase_string)]
    phrase_names = [phrase_string[phrase_starts[i]: phrase_starts[i + 1]] for i in range(len(phrase_starts) - 1)]
    phrase_types = [pn[0] for pn in phrase_names]
    phrase_lgths = np.array([int(pn[1:]) for pn in phrase_names])
    return phrase_names, phrase_types, phrase_lgths


def read_label(label_fn):
    """label_fn is human_label1.txt or human_label2.txt in the song folder of the dataset."""
    with open(label_fn) as f:
        phrase_label = f.readlines()[0].strip()
    phrase_names, phrase_types, phrase_lgths = _parse_phrase_label(phrase_label)

    phrase_starts = _cum_time_to_time_dur(phrase_lgths)

    phrases = [{'name': pn, 'type': pt, 'lgth': pl, 'start': ps}
               for pn, pt, pl, ps in zip(phrase_names, phrase_types, phrase_lgths, phrase_starts)]

    return phrases


def read_melody(melody_fn):
    """melody_fn is melody.txt in the song folder of the dataset."""

    # convert txt file to numpy array of (pitch, duration)
    with open(melody_fn) as f:
        melody = f.readlines()
    melody = [m.strip().split(' ') for m in melody]
    melody = np.array([[int(m[0]), int(m[1])] for m in melody])

    # convert numpy array of (pitch, duration) to (onset, pitch, dur)
    starts = _cum_time_to_time_dur(melody[:, 1])
    durs = melody[:, 1]
    pitches = melody[:, 0]
    is_pitch = melody[:, 0] != 0
    note_mat = np.stack([starts, pitches, durs], -1)[is_pitch]

    return note_mat


def _read_chord_string(c):
    """
    "\x01"s are replaced with " ". ")"s are added to "sus4(b7".
    (E.g., check with if c_name[-7:] == 'sus4(b7'). And there may have other annotation problems.)
    The files in this repo have been cleaned.
    """
    c = c.strip().split(' ')
    c_name = c[0]
    c_dur = int(c[-1])

    # cleaning the chord symbol
    c_name = c_name.replace('\x01', '')

    # convert to chroma representation
    root, chroma, bass = mir_eval.chord.encode(c_name)
    chroma = np.roll(chroma, shift=root)
    return np.concatenate([np.array([root]), chroma, np.array([bass]), np.array([c_dur])])


def read_chord(chord_fn):
    """chord_fn is finalized_chord.txt in the song folder of the dataset."""
    with open(chord_fn) as f:
        chords = f.readlines()

    # convert chord text label to chroma representation
    chords = np.stack([_read_chord_string(c) for c in chords], 0)

    # convert chord to output chord matrix
    starts = _cum_time_to_time_dur(chords[:, -1])
    chord_mat = np.concatenate([starts[:, np.newaxis], chords], -1)

    return chord_mat

'''
def process_multi_channel_matrix(acc_mats):
    track_names = sorted(acc_mats.keys(), key=lambda x: int(x))  # 名稱是數字字串
    tracks = [acc_mats[name] for name in track_names]
    acc = np.concatenate(tracks, axis=0)
    acc = acc[acc[:, 0].argsort()]
    return acc

def read_data(data_fn, song_name=None):
    infojsonname = f"{song_name}_info.json"
    matnpzname = f"{song_name}_mat.npz"

    infojsonpath = os.path.join(data_fn, infojsonname)
    matnpzpath = os.path.join(data_fn, matnpzname)
    #print("Info path: ", infojsonpath, "Mat path: ", matnpzpath)
    with open(infojsonpath, 'r', encoding='utf-8') as f:
        info = json.load(f)

    '''
        McpaMusic's form
        def __init__(self, mat,
            num_beat_per_measure=4, num_step_per_beat=4,
            song_name=None, clean_chord_unit=4, phrase_label=None, melody=None, chord=None, acc=None):
    '''

    mat = np.load(matnpzpath, allow_pickle=True)
    mat = process_multi_channel_matrix(mat)
    numerator, denominator = map(int, info["time_signature"].split("/"))
    num_beat_per_measure = numerator
    num_step_per_beat = 4
    clean_chord_unit = num_beat_per_measure
    song = McpaMusic(mat, num_beat_per_measure, num_step_per_beat, song_name, clean_chord_unit)
    return song
    
'''
def read_data(data_fn, acc_fn, num_beat_per_measure=4, num_step_per_beat=4,
              clean_chord_unit=None, song_name=None, label=1):
    if label == 1:
        label_fn = os.path.join(data_fn, 'human_label1.txt')
    elif label == 2:
        label_fn = os.path.join(data_fn, 'human_label2.txt')
    else:
        raise NotImplementedError

    label = read_label(label_fn)

    melody_fn = os.path.join(data_fn, 'melody.txt')
    melody = read_melody(melody_fn)

    chord_fn = os.path.join(data_fn, 'finalized_chord.txt')
    chord = read_chord(chord_fn)

    clean_chord_unit = num_beat_per_measure if clean_chord_unit is None else clean_chord_unit

    acc_mats = np.load(os.path.join(acc_fn, 'acc_mat.npz'))
    bridge_track, piano_track = acc_mats['bridge'], acc_mats['piano']
    acc = np.concatenate([bridge_track, piano_track], 0)
    acc = acc[acc[:, 0].argsort()]

    song = McpaMusic(melody, chord, acc, label, num_beat_per_measure,
                     num_step_per_beat, song_name, clean_chord_unit)

    return song
'''
