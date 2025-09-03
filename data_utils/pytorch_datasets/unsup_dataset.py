import numpy as np
import matplotlib.pyplot as plt
from .simple_class import *
from .const import LANGUAGE_DATASET_PARAMS, AUTOREG_PARAMS, SHIFT_HIGH_T, SHIFT_LOW_T, SHIFT_HIGH_V, SHIFT_LOW_V

'''

Some constant values:

SHIFT_HIGH_T = 5
SHIFT_LOW_T = -6
SHIFT_HIGH_V = 0
SHIFT_LOW_V = 0

LANGUAGE_DATASET_PARAMS = {
    'unsup': {'max_l': 128, 'h':128, 'n_channel': 2, 'cur_channel': 2}
}

AUTOREG_PARAMS = {
    
    'unsup':{
        'autoreg_seg_lgth': 4, 'max_n_autoreg': 2, 'n_autoreg_prob': np.array([0.1, 0.2, 0.7]),
        'seg_pad_unit': 4, 'autoreg_max_l': 136
    }
}

'''
'''
train_set = UnSupDataset(
        train_analyses,
        SHIFT_HIGH_T, SHIFT_LOW_T, lang_params['max_l'], lang_params['h'], lang_params['n_channel'],
        autoreg_seg_lgth=autoreg_params['autoreg_seg_lgth'], max_n_autoreg=autoreg_params['max_n_autoreg'],
        n_autoreg_prob=autoreg_params['n_autoreg_prob'], seg_pad_unit=autoreg_params['seg_pad_unit'],
        autoreg_max_l=autoreg_params['autoreg_max_l'], use_autoreg_cond=autoreg,
        use_external_cond=external, multi_label=multi_label, random_pitch_aug=random_pitch_aug
    )
'''

class UnSupDataset(SimpleMusicDataset):
    '''
    def __init__(self, analyses, shift_high=0, shift_low=0, max_l=128, h=128, n_channels=10,
                 autoreg_seg_lgth=8, max_n_autoreg=3, n_autoreg_prob=np.array([0.1, 0.1, 0.2, 0.6]),
                 seg_pad_unit=4, autoreg_max_l=108,
                 use_autoreg_cond=True, use_external_cond=False, multi_phrase_label=False,
                 random_pitch_aug=True, mask_background=False):
    '''
    def __init__(self, analyses, shift_high=0, shift_low=0, max_l=128, h=128, n_channels=2,
                 autoreg_seg_lgth=9, max_n_autoreg=3, n_autoreg_prob=np.array([0.1,0.2,0.6]),
                 seg_pad_unit=4, autoreg_max_l=108, use_autoreg_cond=False, use_external_cond=False,
                 multi_label=False, random_pitch_aug=True):
        
        '''
        SimpleMusicDataset constructor:
        def __init__(self, analyses, shift_high = 6, shift_low = 5, max_l=128, h=128, n_channels=None,
                 autoreg_seg_lgth=None,  max_n_autoreg=None, n_autoreg_prob=None,
                 seg_pad_unit=None, autoreg_max_l=None,
                 use_autoreg_cond=False, use_external_cond=False, multi_label=False,
                 random_pitch_aug=True):
        '''
        super(UnSupDataset, self).__init__(
            analyses, shift_high, shift_low, max_l, h, n_channels, autoreg_seg_lgth, max_n_autoreg, n_autoreg_prob,
            seg_pad_unit, autoreg_max_l, use_autoreg_cond, use_external_cond, multi_label, random_pitch_aug
        )
        # roll contains shape = (2, l, 128) roll np array, l's unit = step
        self.music_langs = [analysis['language'] for analysis in analyses]
        self.music_rolls = [ml['roll'] for ml in self.music_langs]

        # self.lengths unit = step
        self.lengths = [mus.shape[1] for mus in self.music_rolls]
        
        # number step per beat * number beat per measure = number of step per measure\
        # hop every measure as start ids
        self.start_ids_per_song = [np.arange(0, lgth - self.max_l // 2, nbpm * nspb * 8, dtype=np.int64)
                                   for lgth, nbpm, nspb in zip(self.lengths, self.nbpms, self.nspbs)]
        
        # 3. construct indicies
        self.indices = self._song_id_to_indices()
    
    def get_data_sample(self, song_id, start_id, shift):
        nbpm, nspb = self.nbpms[song_id], self.nspbs[song_id]
        #print(f"npbm={nbpm}, nspb={nspb}")
        pitch_shift = shift
        #pitch_shift = compute_octave_pitch_shift_value(shift, self.min_mel_pitches[song_id], self.max_mel_pitches[song_id])
        #print(f"pitch_shift={pitch_shift}")
        
        self.store_music(song_id, pitch_shift)

        img = self.lang_to_img(song_id, start_id, end_id=start_id + self.max_l, tgt_lgth=self.max_l)
        
        # prepare for the autoreg condition
        if self.use_autoreg_cond:
            autoreg_cond = self.get_autoreg_cond(song_id, start_id, nbpm * nspb)
        else:
            autoreg_cond = None

        return img, autoreg_cond, None
    
    def lang_to_img(self, song_id, start_id, end_id, tgt_lgth=None):
        #print(f'start_id:{start_id}, end_id:{end_id}')
        # end_id > the last one, won't report error
        music_roll = self._music_roll[:,start_id: end_id, :]
        actual_l = music_roll.shape[1]
        #print(f'acutal_length:{actual_l}')
        # to output image
        if tgt_lgth is None:
            tgt_lgth = end_id - start_id
        #print(f'tgt_length: {tgt_lgth}')
        # if length isn't enough for tgt_lgth: pad zero
        img = np.zeros((self.n_channels, tgt_lgth, 128), dtype=np.float32)
        img[0:2, 0:actual_l, 0:128] = music_roll

        return img[:, :, 0: self.h]
    
    def show(self, item, show_img=True):
        print(self[item])
        data, autoreg, _ = self[item]

        titles = ['music roll']

        if show_img:
            fig, axs = plt.subplots(1, 2, figsize=(20, 40))
            for i in range(1):
                img = data[2 * i: 2 * i + 2]
                img = np.pad(img, pad_width=((0, 1), (0, 0), (0, 0)), mode='constant')
                img[2][img[0] < 0] = 1
                img[img < 0] = 0
                img = img.transpose((2, 1, 0))
                print("img shape:", img.shape)
                print("img min:", img.min(), "max:", img.max(), "dtype:", img.dtype)
                axs[0].imshow(img, origin='lower', aspect='auto')
                axs[0].title.set_text(titles[i])

                autoreg_img = autoreg[2 * i: 2 * i + 2]
                autoreg_img = np.pad(autoreg_img, pad_width=((0, 1), (0, 0), (0, 0)), mode='constant')
                autoreg_img[2][autoreg_img[0] < 0] = 1
                autoreg_img[autoreg_img < 0] = 0
                autoreg_img = autoreg_img.transpose((2, 1, 0))

                axs[1].imshow(autoreg_img, origin='lower', aspect='auto')

            plt.savefig(f"{item}.png")

def create_unsup_datasets(train_analyses, valid_analyses, autoreg=True, external=False,
                           multi_label=False, random_pitch_aug=True, segment_length=128):
    """
    建立訓練集和驗證集的 MelodyDataset。
    
    train_analyses / valid_analyses: list of analysis dicts
    autoreg: 是否使用自回歸條件
    external: 目前沒用到
    multi_label: 目前沒用到
    random_pitch_aug: 是否使用隨機 pitch shift
    segment_length: 每個 segment 的長度
    """
    lang_params = LANGUAGE_DATASET_PARAMS['unsup']
    autoreg_params = AUTOREG_PARAMS['unsup']


    train_set = UnSupDataset(
        train_analyses,
        SHIFT_HIGH_T, SHIFT_LOW_T, lang_params['max_l'], lang_params['h'], lang_params['n_channel'],
        autoreg_seg_lgth=autoreg_params['autoreg_seg_lgth'], max_n_autoreg=autoreg_params['max_n_autoreg'],
        n_autoreg_prob=autoreg_params['n_autoreg_prob'], seg_pad_unit=autoreg_params['seg_pad_unit'],
        autoreg_max_l=autoreg_params['autoreg_max_l'], use_autoreg_cond=autoreg,
        use_external_cond=external, multi_label=multi_label, random_pitch_aug=random_pitch_aug
    )
    
    valid_set = UnSupDataset(
        valid_analyses,
        SHIFT_HIGH_T, SHIFT_LOW_T, lang_params['max_l'], lang_params['h'], lang_params['n_channel'],
        autoreg_seg_lgth=autoreg_params['autoreg_seg_lgth'], max_n_autoreg=autoreg_params['max_n_autoreg'],
        n_autoreg_prob=autoreg_params['n_autoreg_prob'], seg_pad_unit=autoreg_params['seg_pad_unit'],
        autoreg_max_l=autoreg_params['autoreg_max_l'], use_autoreg_cond=autoreg,
        use_external_cond=external, multi_label=multi_label, random_pitch_aug=random_pitch_aug
    )
    
    return train_set, valid_set