from .read_pop909_data import load_train_and_valid_data, analyze_train_and_valid_datasets
from .pytorch_datasets import create_form_datasets, create_counterpoint_datasets, create_leadsheet_datasets, \
    create_accompaniment_datasets, create_unsup_datasets
from .pytorch_datasets.dataloaders import create_train_valid_dataloaders
from .pytorch_datasets.const import LANGUAGE_DATASET_PARAMS, AUTOREG_PARAMS


'''
train_set, valid_set = load_datasets(
            mode, use1k7, multi_label, random_pitch_aug, autoreg, external
)
'''

def load_datasets(mode, use1k7, multi_label, random_pitch_aug, autoreg, external):
    train_data, valid_data = load_train_and_valid_data(use1k7, multi_label)
    train_analyses, valid_analyses = analyze_train_and_valid_datasets(train_data, valid_data)
    #PARAMS_DICTS = {'unsup': params_unsup, 'sup': params_sup, 'mel': params_mel}
    if mode == 'unsup':
        train_set, valid_set = create_unsup_datasets(
            train_analyses, valid_analyses, autoreg, external, multi_label, random_pitch_aug
        )
    else:
        raise NotImplementedError
    '''
    elif mode == 'sup':
        train_set, valid_set = create_sup_datasets(
            train_analyses, valid_analyses, autoreg, external, multi_label, random_pitch_aug
        )
    elif mode == 'mel':
        train_set, valid_set = create_mel_datasets(
            train_analyses, valid_analyses, autoreg, external, multi_label, random_pitch_aug
        )
    else:
        raise NotImplementedError
    '''
    return train_set, valid_set

if __name__ == "__main__":
    train_set, valid_set = load_datasets('unsup', True, False, True, True, False)