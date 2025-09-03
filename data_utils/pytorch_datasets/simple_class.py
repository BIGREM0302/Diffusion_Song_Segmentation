import numpy as np
from torch.utils.data import Dataset

def compute_pitch_shift_value(shift, min_pitch, max_pitch):
    min_pitch += shift
    max_pitch += shift
    min_allowed_thresh = 21
    max_allowed_thresh = 108

    down_shift_range = (min_allowed_thresh - min_pitch)
    if down_shift_range >= 0:
        down_shift_range = 0
    up_shift_range = (max_allowed_thresh - max_pitch)
    if up_shift_range <= 0:
        up_shift_range = 0
    possible_shifts = np.arange(down_shift_range, up_shift_range+1)
    probs = np.array([5 if po == 0 else 1 for po in possible_shifts], dtype=np.float64)
    probs /= probs.sum()
    legal_shift = np.random.choice(possible_shifts, p=probs)

    return shift+legal_shift

def compute_octave_pitch_shift_value(shift, min_pitch, max_pitch):
    min_pitch += shift
    max_pitch += shift
    min_allowed_thresh = 21
    max_allowed_thresh = 108

    down_octave = (min_allowed_thresh - min_pitch) // 12 + 1
    up_octave = (max_allowed_thresh - max_pitch) // 12

    possible_octaves = np.arange(down_octave, up_octave + 1)
    probs = np.array([5 if po == 0 else 1 for po in possible_octaves], dtype=np.float64)
    probs /= probs.sum()
    octave = np.random.choice(possible_octaves, p=probs)
    return shift + octave * 12

def compute_octave_pitch_shift_melody_value(shift, min_pitch, max_pitch):
    min_pitch += shift
    max_pitch += shift
    min_allowed_thresh = 48
    max_allowed_thresh = 98

    down_octave = (min_allowed_thresh - min_pitch) // 12 + 1
    up_octave = (max_allowed_thresh - max_pitch) // 12

    possible_octaves = np.arange(down_octave, up_octave + 1)
    probs = np.array([5 if po == 0 else 1 for po in possible_octaves], dtype=np.float64)
    probs /= probs.sum()
    octave = np.random.choice(possible_octaves, p=probs)
    return shift + octave * 12

def select_prev_slices_nearest(t_m, tgt_lgth_m, n_seg):
    """
    從過去挑選距離 t_m 最近的 n_seg 個片段（長度為 tgt_lgth_m 小節）
    確保不超出 [0, t_m) 範圍
    """
    slices = []
    if t_m <= 0:
        return slices  # 沒有過去可用

    # 能夠開始的最大位置
    max_start = max(0, t_m - tgt_lgth_m)

    # 如果過去不足 tgt_lgth_m，就縮短長度
    if max_start == 0 and t_m < tgt_lgth_m:
        tgt_lgth_m = t_m  # 用到開頭為止

    # 所有可能的開始位置
    possible_starts = list(range(0, max_start + 1))

    if not possible_starts:
        return slices

    # **取距離 t_m 最近的 n_seg 個開始位置**
    # 排序方式：大的在前 (越接近 t_m)
    chosen_starts = sorted(possible_starts, reverse=True)[:n_seg]

    for start in chosen_starts:
        end = min(t_m, start + tgt_lgth_m)
        if start < end:
            slices.append((start, end))

    return slices


def select_prev_slices_random(t_m, tgt_lgth_m, n_seg):
    """
    從過去隨機挑選 n_seg 個片段（長度為 tgt_lgth_m 小節）
    確保不超出 [0, t_m) 範圍

    t_m: current position (unit = measure)
    tgt_lgth_m: length of each segment (unit = measure)
    n_seg: number of segments to be picked
    """
    slices = []
    if t_m <= 0:
        return slices  # 沒有過去可用

    # 能夠開始的最大位置
    max_start = max(0, t_m - tgt_lgth_m)

    # 如果過去不足 tgt_lgth_m，就縮短長度
    if max_start == 0 and t_m < tgt_lgth_m:
        tgt_lgth_m = t_m  # 用到開頭為止

    # 所有可能的開始位置
    possible_starts = list(range(0, max_start + 1))

    if not possible_starts:
        return slices

    # 挑 n_seg 個不同的開始位置（少於 n_seg 就全挑）
    chosen_starts = np.random.choice(
        possible_starts,
        size=min(n_seg, len(possible_starts)),
        replace=False
    )

    for start in chosen_starts:
        end = min(t_m, start + tgt_lgth_m)
        if start < end:
            slices.append((start, end))
    # will return [[start, end], [start, end], ..., [start end]]
    return slices

class SimpleMusicDataset(Dataset):
    '''
    train_set, valid_set = create_unsup_datasets(
            train_analyses, valid_analyses, autoreg, external, multi_label, random_pitch_aug
        )
    '''
    def __init__(self, analyses, shift_high = 6, shift_low = 5, max_l=128, h=128, n_channels=None,
                 autoreg_seg_lgth=None,  max_n_autoreg=None, n_autoreg_prob=None,
                 seg_pad_unit=None, autoreg_max_l=None,
                 use_autoreg_cond=False, use_external_cond=False, multi_label=False,
                 random_pitch_aug=True):
        super(SimpleMusicDataset, self).__init__()

        self.multi_label = multi_label
        self.random_pitch_aug = random_pitch_aug  # sample a pitch shift (True) or extend the dataset 12 times (False).
        self.use_autoreg_cond = use_autoreg_cond
        self.use_external_cond = use_external_cond

        self.music_rolls = None
        self._music_roll = None # after pitch augmentation

        self.shift_high = shift_high
        self.shift_low = shift_low
        self.max_l = max_l
        self.h = h
        self.n_channels = n_channels

        self.min_mel_pitches = [analysis['min_mel_pitch'] for analysis in analyses]
        self.max_mel_pitches = [analysis['max_mel_pitch'] for analysis in analyses]

        self.nbpms = [analysis['nbpm'] for analysis in analyses]
        self.nspbs = [analysis['nspb'] for analysis in analyses]
        self.song_names = [analysis['name'] for analysis in analyses]

        ####
        self.lengths = None  # an array: [num_time_step_song_0, num_time_step_song_1, num_time_step_song_2, ...]
        self.start_ids_per_song = None  # a list: [[seg0_start, seg1_start, ...], [seg0_start, seg1_start, ...], ...]
        self.indices = None  # a list of all possible (song_id, segment_id) pairs)
        ####

        if self.use_autoreg_cond:
            assert max_n_autoreg + 1 == len(n_autoreg_prob), "max_n_autoreg + 1 == len(n_autoreg_prob)."
        self.autoreg_seg_lgth = autoreg_seg_lgth # the length of autoregressiv segments

        self.max_n_autoreg = max_n_autoreg # max number of auto regressive condition segments
        self.n_autoreg_prob = n_autoreg_prob # from 0, 1, 2.... max_n_autoreg

        self.seg_pad_unit = seg_pad_unit
        self.autoreg_max_l = autoreg_max_l # specify the max length of auto regressive condition
    


    def _song_id_to_indices(self):
        assert self.start_ids_per_song is not None, "The attribute start_ids_per_song must be filled first."
        '''
        output will be like:
        array([
            [0, 0], [0, 16], [0, 32],
            [1, 0], [1, 24], [1, 48], [1, 72],
            [2, 0], [2, 12]
        ])
        '''
        # song_id starts from 0
        return np.concatenate([
            np.stack([np.ones(len(start_ids), dtype=np.int64) * song_id, start_ids], -1)
            for song_id, start_ids in enumerate(self.start_ids_per_song)
        ], 0)

    def __len__(self):
        '''
        # number of segments = length of self indices, where self indices = [[song_id, start_id]]
        if self.multi_phrase_label:
            assert len(self.indices) % 2 == 0, "len(self.indices) must be even when self.random_label=True."
            num_segment = len(self.indices) // 2
        else:
        '''
        num_segment = len(self.indices)
        # number of pitch augmentation = 1 if random, else = shift high - shift low + 1
        num_aug_pitch = 1 if self.random_pitch_aug else self.shift_high - self.shift_low + 1

        return num_segment * num_aug_pitch

    def get_data_sample(self, song_id, start_id, shift):
        raise NotImplementedError

    def __getitem__(self, item):
        '''
        if self.multi_phrase_label:
            if np.random.random() > 0.5:
                item = len(self) + item
        '''
        # if random pitch augmention then input item = segment id
        #print(f"shift_high={self.shift_high}, shift_low={self.shift_low}")
        if self.random_pitch_aug:
            segment_id = item
            pitch_shift = np.random.randint(self.shift_high - self.shift_low + 1) + self.shift_low
        else:
            segment_id = item // (self.shift_high - self.shift_low + 1)
            pitch_shift = item % (self.shift_high - self.shift_low + 1) + self.shift_low
        #print(f"pitch_shift={pitch_shift}")
        song_id, start_id = self.indices[segment_id]
        #print(f"song_id={song_id},start_id={start_id}")
        return self.get_data_sample(song_id, start_id, pitch_shift)
    
    def select_autoreg_slices(self, start_id, scale_unit):
        t_m = start_id // scale_unit  # 當前小節位置
        # random choose the number of previous segments
        # if p - None, random choice will assume equal probability
        #print(f"max autoregression number={self.max_n_autoreg}")
        n_seg = np.random.choice(
            np.arange(0, self.max_n_autoreg + 1),
            p=self.n_autoreg_prob
        )
        # randomly choose the number of auto condition segments
        #print(f"n_seg={n_seg}")
        #print(f"current measure number={t_m}")
        '''
        autoreg_slices = select_prev_slices_random(
            t_m,
            self.autoreg_seg_lgth,
            n_seg
        )
        '''
        autoreg_slices = select_prev_slices_nearest(
            t_m,
            self.autoreg_seg_lgth,
            n_seg
        )
        #print("Select conditoin slices:")
        #print(autoreg_slices)
        # return [[start, end],[start, end]...], unit = measure
        return autoreg_slices

    def lang_to_img(self, song_id, start_id, end_id, tgt_lgth):
        # which should be specified afterwards
        raise NotImplementedError

    def get_autoreg_cond(self, song_id, start_id, scale_unit):
        # scale unit is number of step / measure
        # pass the scale unit, which convert between start_id(littler unit) and t_m(measure position)
        # the height is equals to 128, max
        #print(f"autoreg_max_l: {self.autoreg_max_l}; n_channels: {self.n_channels}")
        cond_img = -np.ones((self.n_channels, self.autoreg_max_l, self.h), dtype=np.float32)
        # autoreg_slice is [[start, end],...], in unit = measure
        autoreg_slices = self.select_autoreg_slices(start_id, scale_unit)

        # scale_unit_: the min 2^x s.t. 2^x >= scale_unit
        scale_unit_ = int(2 ** np.ceil(np.log2(scale_unit)))
        seg_lgth_unit = self.autoreg_seg_lgth * scale_unit_ + self.seg_pad_unit
                    
        for i, slc in enumerate(autoreg_slices):
            #print(f"i:{i}")
            # slc[0]: start measure, slc[1]: end measure
            seg_start, seg_end = slc[0] * scale_unit, slc[1] * scale_unit
            #print(f"seg_start={seg_start}, seg_end={seg_end}")
            tgt_l = self.autoreg_seg_lgth * scale_unit
            #print(f"tgt_l:{tgt_l}")
            autoreg_img = self.lang_to_img(song_id, seg_start, seg_end, tgt_l)

            cond_img[:, i * seg_lgth_unit: i * seg_lgth_unit + tgt_l] = autoreg_img

        return cond_img
    
    def store_music(self, song_id, shift):
        # only store "one" song_id's roll to _music_roll
        if self.music_rolls is not None:
            music_roll = self.music_rolls[song_id]
            self._music_roll = np.roll(music_roll, shift=shift, axis=-1)