import numpy as np
import os


def create_train_valid_test_split(output_fn, n_sample, train_factor=8, valid_factor=1, test_factor=1, seed=1234):
    """
    隨機切分資料成 train / valid / test
    :param output_fn: 儲存 .npz 檔路徑
    :param n_sample: 總樣本數
    :param train_factor: 訓練比例分子
    :param valid_factor: 驗證比例分子
    :param test_factor: 測試比例分子
    :param seed: 隨機種子
    """
    np.random.seed(seed)

    data_ids = np.arange(0, n_sample)
    total_factor = train_factor + valid_factor + test_factor

    n_train = int(n_sample * train_factor / total_factor)
    n_valid = int(n_sample * valid_factor / total_factor)
    n_test = n_sample - n_train - n_valid  # 保證總數正好等於 n_sample

    # 打散順序
    shuffled = np.random.permutation(data_ids)

    train_inds = np.sort(shuffled[:n_train])
    valid_inds = np.sort(shuffled[n_train:n_train + n_valid])
    test_inds = np.sort(shuffled[n_train + n_valid:])

    np.savez(output_fn, train_inds=train_inds, valid_inds=valid_inds, test_inds=test_inds)


def load_split_file(split_fn):
    split_data = np.load(split_fn)
    train_inds = split_data['train_inds']
    valid_inds = split_data['valid_inds']
    test_inds = split_data['test_inds']
    return train_inds, valid_inds, test_inds


if __name__ == '__main__':

    output_fn0 = os.path.join('data', 'split2.npz')
    output_fn1 = os.path.join('data', 'split3.npz')

    # 80% train, 10% valid, 10% test
    create_train_valid_test_split(output_fn0, 909, train_factor=8, valid_factor=1, test_factor=1, seed=1234)
    create_train_valid_test_split(output_fn1, 2656, train_factor=8, valid_factor=1, test_factor=1, seed=1234)

    t_id, v_id, te_id = load_split_file(output_fn0)
    print(f'split0 -> n_train={len(t_id)}, n_valid={len(v_id)}, n_test={len(te_id)}')

    t_id, v_id, te_id = load_split_file(output_fn1)
    print(f'split1 -> n_train={len(t_id)}, n_valid={len(v_id)}, n_test={len(te_id)}')
