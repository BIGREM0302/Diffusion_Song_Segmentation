from . import *
from data_utils import load_datasets, create_train_valid_dataloaders
from model import init_ldm_model, init_diff_pro_sdf

class LdmTrainConfig(TrainConfig):
    
    def __init__(self, params, output_dir, mode, use1k7, autoreg, external,
                 multi_label, random_pitch_aug) -> None:
        super().__init__(params, None, output_dir)
        self.use_autoreg_cond = autoreg
        self.use_external_cond = external
        self.multi_phrase_label = multi_label
        self.use1k7 = use1k7
        self.random_pitch_aug = random_pitch_aug

        self.ldm_model = init_ldm_model(mode, autoreg, external, params, False)
        self.model = init_diff_pro_sdf(self.ldm_model, params, self.device)

        train_set, valid_set = load_datasets(
            mode, use1k7, multi_label, random_pitch_aug, autoreg, external
        )
        self.train_dl, self.val_dl = create_train_valid_dataloaders(params.batch_size, train_set, valid_set)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=params.learning_rate
        )

if __name__ == "__main__":
    train_set, valid_set = load_datasets('unsup', True, False, False, True, False)
    print(train_set.autoreg_max_l)
    print(train_set.nbpms)
    print(train_set.indices)
    #img, autoreg_cond = train_set[66]
    train_set.show(101)
    train_set.show(102)
    train_set.show(103)
    train_set.show(104)
    #print(len(train_set))
    '''
    print(img)
    print(autoreg_cond)
    print(train_set.start_ids_per_song[0])
    print(train_set.lengths[0])
    '''
    train_dl, valid_dl = create_train_valid_dataloaders(batch_size=4, train_set=train_set, valid_set=valid_set)
    
    #print(train_set.music_rolls)
    #print(train_set.max_mel_pitches)
    #print(train_set.min_mel_pitches)
    #print(train_set.start_ids_per_song)