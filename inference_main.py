import torch
import os
import re
import numpy as np
from model import init_ldm_model, Diffpro_SDF, get_model_path
from model.stable_diffusion.sampler.sampler_sdf import SDFSampler
from params import PARAMS_DICTS
from data_utils.pytorch_datasets.const import LANGUAGE_DATASET_PARAMS, AUTOREG_PARAMS, SHIFT_HIGH_T, SHIFT_LOW_T, SHIFT_HIGH_V, SHIFT_LOW_V
import torch.nn.functional as F
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import json
from scipy.signal import find_peaks

PROJECT_PATH = os.path.join(os.path.dirname(__file__))
DATASET_PATH = os.path.join(PROJECT_PATH, 'datasets', 'data')
LABEL_PATH = os.path.join(PROJECT_PATH, 'datasets', 'pop909data', 'pop909_w_structure_label')
MODEL_OUTPUT_PATH = os.path.join(PROJECT_PATH, 'results')

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def note_end_matrix_to_piano_roll(note_mat, total_length=None):
    total_length = total_length if total_length is not None else max(note_mat[:, 2])
    #print("total_length:", total_length)
    piano_roll = np.zeros((2, total_length, 128), dtype=np.int64)
    for note in note_mat:
        onset, pitch, end = note
        if onset >= end:
            continue   
        #print(f"{onset},{pitch},{end}")
        piano_roll[0, onset, pitch] = 1
        piano_roll[1, onset + 1: end, pitch] = 1
    return piano_roll

def note_matrix_to_piano_roll(note_mat, total_length=None):
    total_length = total_length if total_length is not None else max(note_mat[:, 0] + note_mat[:, 2])

    piano_roll = np.zeros((2, total_length, 128), dtype=np.int64)
    for note in note_mat:
        onset, pitch, duration = note
        piano_roll[0, onset, pitch] = 1
        piano_roll[1, onset + 1: onset + duration, pitch] = 1
    return piano_roll

def process_multi_channel_matrix(acc_mats):
    track_names = sorted(acc_mats.keys(), key=lambda x: int(x))  # 名稱是數字字串
    tracks = [acc_mats[name] for name in track_names]
    acc = np.concatenate(tracks, axis=0)
    acc = acc[acc[:, 0].argsort()]
    return acc

def boundary_score(predicted, reference, tol=1):
    predicted = sorted(predicted)
    reference = sorted(reference)

    matched_ref = set()
    tp = 0  # true positives

    for p in predicted:
        for r in reference:
            if r in matched_ref:
                continue
            if abs(p - r) <= tol:
                tp += 1
                matched_ref.add(r)
                break
    
    fp = len(predicted) - tp  # false positives
    fn = len(reference) - tp  # false negatives
    print("TP:", tp)
    print("FN:", fn)
    print("FP:", fp)
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}

def plot_metrics_with_values(list1, list2):
    metrics = ["precision", "recall", "f1"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    for row, data in enumerate([list1, list2]):
        for col, metric in enumerate(metrics):
            values = [d[metric] for d in data]
            axes[row, col].plot(values, marker="o", linestyle="-")
            axes[row, col].set_title(f"List{row+1} - {metric}")
            axes[row, col].set_xlabel("Index")
            axes[row, col].set_ylabel(metric)
            axes[row, col].grid(True, linestyle="--", alpha=0.5)

            for i, v in enumerate(values):
                axes[row, col].text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig("results.png")
    plt.close()

'''
mat = np.load(matnpzpath, allow_pickle=True)
mat = process_multi_channel_matrix(mat)
'''

# [START] For embedding/loss analysis ===============

def pairwise_l2_matrix(X: np.ndarray) -> np.ndarray:
    """
    Compute pairwise L2 distances for a set of vectors X (N, D).
    Returns S with shape (N, N), S[i, j] = ||X[i] - X[j]||_2.
    """
    # Efficient computation using (x - y)^2 = ||x||^2 + ||y||^2 - 2 x·y
    X = np.asarray(X, dtype=float)
    norms2 = np.sum(X**2, axis=1, keepdims=True)  # (N, 1)
    G = X @ X.T                                   # Gram matrix (N, N)
    D2 = norms2 + norms2.T - 2.0 * G              # Squared distances
    # Numerical guard for negatives due to floating error
    D2[D2 < 0] = 0.0
    return np.sqrt(D2)

def median_filter_2d(A: np.ndarray, k: int) -> np.ndarray:
    """
    Simple 2D median filter with odd window size (2k+1).
    Zero-pads the borders.
    """
    A = np.asarray(A, dtype=float)
    N, M = A.shape
    w = 2 * k + 1
    pad = k
    Ap = np.pad(A, pad_width=pad, mode='edge')
    out = np.empty_like(Ap, dtype=float)
    # Compute only the valid region into an array of shape (N, M)
    out_valid = np.empty((N, M), dtype=float)
    for i in range(N):
        for j in range(M):
            window = Ap[i:i+w, j:j+w]
            out_valid[i, j] = np.median(window)
    return out_valid

def checkerboard_kernel(kappa: int, sigma = 1.0) -> np.ndarray:
    """
    Construct g[i, j] = sgn(i) * sgn(j) * exp(-(i - j)^2),
    where i, j in [-kappa, ..., 0, ..., kappa].
    """
    r = np.arange(-kappa, kappa + 1)
    I, J = np.meshgrid(r, r, indexing='ij')
    #print(I)
    #print(J)
    g = -np.sign(I) * np.sign(J) * np.exp(-(((I - J) ** 2)/sigma).astype(float))
    #print("g.mean = ",g.mean())
    '''
    for i in np.arange(-1, 2):
        for j in np.arange(-1, 2):
            g[i+kappa][j+kappa] = 0
    '''
    return g


def diagonal_convolution_novelty(Sbar: np.ndarray, kappa: int, sigma: float) -> np.ndarray:
    """
    Compute novelty function eta[nu] by convolving Sbar with checkerboard
    kernel g centered along the main diagonal. Only positions where the
    (2kappa+1)x(2kappa+1) window fully fits are computed.
    Returns eta with length N (zeros at borders that cannot be computed).
    """
    N = Sbar.shape[0]
    assert Sbar.shape[0] == Sbar.shape[1], "Sbar must be square."
    g = checkerboard_kernel(kappa, sigma)
    #print("g:",g)
    w = 2 * kappa + 1
    eta = np.zeros(N, dtype=float)
    for nu in range(kappa, N - kappa):
        patch = Sbar[nu - kappa: nu + kappa + 1, nu - kappa: nu + kappa + 1]
        eta[nu] = np.sum(patch * g)
    return eta

def peak_pick_plateau(eta: np.ndarray, threshold: float, visualize=False):
    """
    Detect peaks in plateau / flat regions of a novelty function.

    Parameters:
    - eta: 1D novelty function (np.ndarray)
    - threshold: values above this are considered part of a plateau
    - visualize: whether to plot eta and detected peaks

    Returns:
    - peaks: np.ndarray of peak indices
    """
    N = len(eta)
    peaks = []

    in_plateau = False
    start = 0

    for i in range(N):
        if eta[i] >= threshold and not in_plateau:
            # Enter a new plateau region
            start = i
            in_plateau = True
        elif eta[i] < threshold and in_plateau:
            # Leave the plateau region
            end = i
            plateau_idx = start + np.argmax(eta[start:end])
            peaks.append(plateau_idx)
            in_plateau = False

    # handle case plateau goes till the last element
    if in_plateau:
        plateau_idx = start + np.argmax(eta[start:N])
        peaks.append(plateau_idx)

    peaks = np.array(peaks, dtype=int)

    return peaks

def peak_pick_threshold(eta: np.ndarray, T: int, tau: float) -> np.ndarray:
    """
    Peak picking with contrast threshold:
    (2T + 1) * eta[nu] / sum_{t=-T..T} eta[nu + t] > tau,
    and eta[nu] is a local maximum within [-T, T].
    Returns an array of peak indices.
    """
    N = len(eta)
    peaks = []
    for nu in range(T, N - T):
        window = eta[nu - T: nu + T + 1]
        denom = np.sum(window)
        if denom <= 0:
            continue
        contrast = (2 * T + 1) * eta[nu] / denom
        # Local-maximum criterion
        if eta[nu] >= np.max(window) and contrast > tau:
            peaks.append(nu)
    return np.array(peaks, dtype=int)

def shift(arr):
    nonzero_min = arr[arr != 0].min()
    #print("None zero min =", nonzero_min)

    shifted = np.where(arr != 0, arr-nonzero_min, 0)
    #print("shifted:", shifted)
    return shifted

def normalize_losses(losses, a=1.0, b=1.0):
    losses = np.array(losses)
    nlosses = np.zeros_like(losses)

    # 從 j=2 開始才有 j-2
    for j in range(2, len(losses)-1):
        nlosses[j] = max(0, a*losses[j+1] + losses[j] - losses[j-1] - b*losses[j-2])
    return nlosses

def analyze_generation_result(x, start_time, split_border = None):
    batch_size = x.shape[0]
    output_folder = os.path.join(PROJECT_PATH, "analyze_generation_result")
    os.makedirs(output_folder, exist_ok=True)
    for id in range(batch_size):
        if split_border is not None:
            analyze_generation_segment(x[id,:,0:split_border,:], f"{id}-{start_time}-1", output_folder)
            analyze_generation_segment(x[id,:,split_border:,:], f"{id}-{start_time}-2", output_folder)
        else:
            analyze_generation_segment(x[id], f"{id}-{start_time}", output_folder)

def analyze_generation_segment(x, name, output_folder):
    print("Number of > 0.5 elements:", count_greater_than_half(x))
    print("Number of < 0.9 elements:", count_less_than_threshold(x, threshold=0.9))
    flat = x.flatten()
    bins = np.arange(0.0, 1.01, 0.1)
    flat = flat[flat != 0]
    counts, _ = np.histogram(flat, bins=bins)

    file_path = os.path.join(output_folder, f"{name}.png")
    fig, ax = plt.subplots()
    # 畫直方圖
    # 畫 histogram 並加上數字標籤
    for i in range(len(counts)):
        width = bins[i+1] - bins[i]
        ax.bar(bins[i], counts[i], width=width, align='edge', edgecolor='black')
        ax.text(bins[i] + width/2, counts[i], str(counts[i]),
                ha='center', va='bottom', fontsize=8)

    ax.set_xticks(bins)
    ax.set_xlabel("Value Range")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Values (0.0 ~ 1.0, excluding zeros)")

    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()

def analyze_loss(losses, bpm=90, ts=0, a=1.0, b=1.0, prominence=0.01, ref_peaks=None, song_name=""):
    plt.figure(figsize=(10,2))
    plt.plot(losses)
    plt.title(f"losses, t={ts}")
    plt.xlabel("i")
    plt.ylabel("loss")
    plt.savefig(f"loss_{ts}.png")

    score = None

    nlosses = normalize_losses(losses, a, b)
    peaks, properties = find_peaks(nlosses, prominence=prominence)
    
    print("ref_peaks:", ref_peaks)
    print("loss length:", len(losses))

    plt.figure(figsize=(10,2))
    plt.plot(nlosses)
    plt.title(f"n_losses, t={ts}")
    plt.xlabel("i")
    plt.ylabel("n_loss")
    plt.savefig(f"{song_name}n_loss_{ts}.png")

    plt.figure(figsize=(12,4))
    plt.plot(losses, label="Raw Loss")
    plt.plot(nlosses, label="Normalized Loss", alpha=0.8)
    plt.scatter(peaks, nlosses[peaks], color='red', marker='x', label="Peaks")
    #ref_peaks = None
    if ref_peaks is not None:
        score = boundary_score(peaks, ref_peaks, tol=4)
        print("Temporal loss:", score)
        for rp in ref_peaks:
            plt.axvline(rp, linestyle='--', color='red')
    plt.title("Losses with Peak Detection")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"{song_name}loss_with_peaks.png")
    plt.close()

    return score


def analyze_pipeline(X_list, bpm=90, tick_of_segment=8 ,ref_peaks=None, song_name=""):
    k_median = 3 # smaller, then more smooth but will lose details
    
    kappa = 10 # 10*2 + 1
    sigma = 18
    
    T = 3
    tau = 1.05
    threshold = 6.5

    score = None

    X_array = np.stack(X_list, axis=0) 
    S = pairwise_l2_matrix(X_array)
    S_bar = median_filter_2d(S, k_median)
    eta = diagonal_convolution_novelty(S_bar, kappa=kappa, sigma=sigma) / 1000.0
    #print(eta)
    eta = shift(eta)
    peaks = peak_pick_threshold(eta, T=T, tau=tau)
    #peaks = peak_pick_plateau(eta, 6.5)
    # save image
    # S
    plt.figure(figsize=(6, 5))
    plt.imshow(S, origin='lower', aspect='auto')
    plt.title("S: Pairwise L2 distance matrix")
    plt.xlabel("j")
    plt.ylabel("i")
    plt.colorbar()
    plt.savefig(f"{song_name}S.png")
    # S bar
    plt.figure(figsize=(6, 5))
    plt.imshow(S_bar, origin='lower', aspect='auto')
    plt.title(f"Sbar: Median-filtered S (window={2*k_median+1})")
    plt.xlabel("j")
    plt.ylabel("i")
    plt.colorbar()
    plt.savefig(f"{song_name}S_bar.png")
    
    # 3) Novelty function
    hop_time = ((tick_of_segment / 4) * 60) / bpm # 1 beat 4 ticks

    times = np.arange(len(eta)) * hop_time

    plt.figure(figsize=(9, 3))
    plt.plot(eta)
    if len(peaks) > 0:
        plt.scatter(peaks, eta[peaks], marker='o')
    plt.title(f"Novelty function (kappa={kappa}) with peaks (T={T}, tau={tau})")
    plt.xlabel("Index (nu)")
    plt.ylabel("eta[nu]")
    plt.grid(True)
    plt.savefig(f"{song_name}eta.png")

    # 4) For reference, show peaks as vertical lines on novelty
    plt.figure(figsize=(9, 3))
    plt.plot(times, eta, label="Novelty Function")
    
    def format_mmss(x, pos):
        minutes = int(x // 60)
        seconds = int(x % 60)
        return f"{minutes:02d}:{seconds:02d}"
    
    def format(x):
        minutes = int(x // 60)
        seconds = int(x % 60)
        return f"{minutes:02d}:{seconds:02d}"
    
    print("Peaks:")
    for p in peaks:
        plt.axvline(times[p], linestyle='--', color='blue', label="Detected" if p == peaks[0] else "")
        print(format(times[p]))
    
    if ref_peaks is not None:
        print("Reference Peaks:")
        for rp in ref_peaks:
            plt.axvline(times[rp], linestyle='--', color='red', label="Reference" if rp == ref_peaks[0] else "")
            print(format(times[rp]))
        score = boundary_score(peaks, ref_peaks, tol=4)
        print(score)

    plt.gca().xaxis.set_major_formatter(FuncFormatter(format_mmss))

    plt.title("Novelty function with detected boundaries")
    plt.xlabel("Time (mm:ss)")
    plt.ylabel("eta[nu]")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{song_name}etaref.png")
    plt.close()

    return score
# [END] For embedding analysis ===============

class GenOpBase:

    mode = None
    data_params = None

    def __init__(self, params, model_path, device, use_autoreg_cond=False, use_external_cond=False, debug_mode=False,
                 is_autocast_fp16=True):
        
        self.data_params = {"max_l":128, "h":128}
        self.mode = 'unsup'
        self.autoreg_params = AUTOREG_PARAMS['unsup']
        
        self.params = params
        self.device = device
        self.use_autoreg_cond = use_autoreg_cond
        self.use_external_cond = use_external_cond
        self.debug_mode = debug_mode
        self.is_autocast_fp16 = is_autocast_fp16

        self.sampler = self.load_sampler(model_path, False)
        # self._consistency_check(model_path)
    def load_sampler(self, model_path, show_image):
        # model ready
        ldm_model = init_ldm_model(self.mode, self.use_autoreg_cond, self.use_external_cond, self.params, self.debug_mode)

        model = Diffpro_SDF.load_trained(ldm_model, model_path).to(self.device)

        sampler = SDFSampler(model.ldm, self.data_params['max_l'],
                             self.data_params['h'], is_autocast=self.is_autocast_fp16, device=self.device,
                             debug_mode=self.debug_mode)

        return sampler

    def show_embed(self, clean_x, ts, autoreg_cond=None):
        
        if autoreg_cond is not None:
            autoreg_cond = torch.from_numpy(autoreg_cond).float().to(self.device)

        if clean_x is not None:
            clean_x = torch.from_numpy(clean_x).float().to(self.device)

        noise = torch.randn_like(clean_x, device=self.device)

        predict_noise, embedding = self.sampler.findembed(clean_x, ts, autoreg_cond=autoreg_cond, noise=noise)

        loss = F.mse_loss(noise, predict_noise)
        loss = loss.item()
        time_emb = embedding["time_emb"].cpu().numpy()
        skip_features = [f.detach().cpu().numpy() for f in embedding["skip_features"]]
        embed = embedding["embed"].cpu().numpy()
        '''
        print(time_emb.shape)
        for feature in skip_features:
            print(feature.shape)
        print(embed.shape)
        '''
        # embedding is a dictionary like:
        '''
        {
            "time_emb": t_emb,
            "skip_features": skip_features,
            "embed": embedding
        }
        '''
        return loss, time_emb, skip_features, embed

    def predict(self, background_cond, autoreg_cond, external_cond, orig_x, mask, 
                n_sample=None, uncond_scale=None,):
        if background_cond is not None:
            background_cond = torch.from_numpy(background_cond).float().to(self.device)

        if autoreg_cond is not None:
            autoreg_cond = torch.from_numpy(autoreg_cond).float().to(self.device)

        if external_cond is not None:
            external_cond = torch.from_numpy(external_cond).float().to(self.device)

        if mask is not None:
            mask = torch.from_numpy(mask).float().to(self.device)
        
        if orig_x is not None:
            orig_x = torch.from_numpy(orig_x).float().to(self.device)
        
        if n_sample is not None:
            batch_size = n_sample

        self.sampler.model.eval()

        output_x = self.sampler.generate(background_cond, autoreg_cond, external_cond, orig_x, mask,
                                         n_sample, uncond_scale)

        output_x = torch.clamp(output_x, min=0, max=1)

        output_x = output_x.cpu().numpy()

        return output_x

def count_greater_than_half(arr: np.ndarray) -> int:
    return np.sum(arr > 0.5)

def count_less_than_threshold(arr: np.ndarray, threshold) -> int:
    return np.sum(arr < threshold) - np.sum(arr == 0)

def process_phrase_txt(filename):
    with open(filename, "r", encoding="utf-8") as f:
        text = f.read().strip()

    # 找出所有數字
    numbers = list(map(int, re.findall(r"\d+", text)))

    # 做累積和
    cumsum = np.cumsum(numbers)

    # 乘以2
    result = (cumsum * 2).tolist()

    return result[:-1]

def show(_data, autoreg=None, name="rollimage"):
    data = np.zeros((_data.shape[0], _data.shape[1], _data.shape[2]), dtype=np.float32)
    data[0:2, 0:_data.shape[1], 0:128] = _data
    
    titles = ['music roll']
    #print(data.shape)
    if autoreg is not None:
        fig, axs = plt.subplots(1, 2, figsize=(20, 40))
    else:
        fig, axs = plt.subplots(1, 1, figsize=(10, 40))
        axs = [axs]

    for i in range(1):
        img = data[2 * i: 2 * i + 2]
        
        img = np.pad(img, pad_width=((0, 1), (0, 0), (0, 0)), mode='constant')
        img[2][img[0] < 0] = 1
        img[img < 0] = 0
        img = img.transpose((2, 1, 0))
        #img = img.astype(np.float32)
        #print("img shape:", img.shape)
        #print("img min:", img.min(), "max:", img.max(), "dtype:", img.dtype)
        axs[0].imshow(img, origin='lower', aspect='auto')
        axs[0].title.set_text(titles[i])

        if autoreg is not None:
            autoreg_img = autoreg[2 * i: 2 * i + 2]
            autoreg_img = np.pad(autoreg_img, pad_width=((0, 1), (0, 0), (0, 0)), mode='constant')
            autoreg_img[2][autoreg_img[0] < 0] = 1
            autoreg_img[autoreg_img < 0] = 0
            autoreg_img = autoreg_img.transpose((2, 1, 0))
            #autoreg_img = autoreg_img.astype(np.float32)
            axs[1].imshow(autoreg_img, origin='lower', aspect='auto')
    
    plt.savefig(f"{name}.png")
    plt.close()


def load_split_file(split_fn):
    split_data = np.load(split_fn)
    train_inds = split_data['train_inds']
    valid_inds = split_data['valid_inds']
    test_inds = split_data['test_inds']
    return train_inds, valid_inds, test_inds

def analyze_embedding(sampler:GenOpBase, num_of_songs = 1, ts=0, seg_length = 8, hop_size = 8, channels=2, height=128, device="cuda"):
    picked_list, matrolls, bpms, phrases = pick_songs(num_of_songs, False)
    num_of_songs = len(picked_list)
    embed_scores = []
    loss_scores = []
    for i in range(num_of_songs):
        start = 0
        matroll = matrolls[i]
        print(matroll.shape)
        total_length = matroll.shape[1]
        matroll = np.expand_dims(matroll, axis=0)
        song_name = picked_list[i]
        print("song name:",song_name)
        print("total length:",total_length)
        embeds = []
        losses = []
        steps = (total_length - seg_length) // hop_size + 1
        for j in tqdm(range(steps), desc=f"Extract {song_name} song's embeddings"):
            start = j*hop_size
            show(matroll[0,:,start:start+seg_length,:])
            loss, time_emb, skip_features, embed = sampler.show_embed(matroll[:,:,start:start+seg_length,:],ts)
            embed = embed.reshape(-1)
            embeds.append(embed)
            losses.append(loss)
            #print("start:", start)
            #print(embed)
        print("===============Embedding extracted, start analyzing============")
        #embedscore = analyze_pipeline(embeds, bpms[i], ref_peaks=phrases[i], song_name=song_name)
        embedscore = analyze_pipeline(embeds, bpms[i], ref_peaks=phrases[i])
        if embedscore is not None:
            embed_scores.append(embedscore)
        print(f"===============Analyze MSE loss for t={ts}=============================")
        #lossscore = analyze_loss(losses, bpms[i], ref_peaks=phrases[i], song_name=song_name)
        lossscore = analyze_loss(losses, bpms[i], ts, ref_peaks=phrases[i])
        if lossscore is not None:
            loss_scores.append(lossscore)
    plot_metrics_with_values(embed_scores, loss_scores)

def generate_full_song(sampler:GenOpBase, total_length=128, block_len=64, batch_size=1, channels=2, height=128, canvas=None, device="cuda", start_step = 0):
    """
    Generate a full song using inpainting block-by-block.
    sampler: 已經訓練好的 sampler，支援 inpainting
    """
    '''
    'unsup':{
        'autoreg_seg_lgth': 4, 'max_n_autoreg': 3, 'n_autoreg_prob': np.array([0.1, 0.1, 0.1, 0.7]),
        'seg_pad_unit': 4, 'autoreg_max_l': 204
    }
    '''
    autoreg_params = AUTOREG_PARAMS['unsup']
    autoreg_max_l = autoreg_params['autoreg_max_l']
    autoreg_seg_lgth_step = autoreg_params['autoreg_seg_lgth']*4*4
    seg_pad_unit = autoreg_params['seg_pad_unit']
    max_n_autoreg = autoreg_params['max_n_autoreg']
    #n_autoreg_prob = autoreg_params['n_autoreg_prob']
    n_autoreg_prob = np.array([0.1, 0.2, 0.7])

    tgtl = seg_pad_unit + autoreg_seg_lgth_step
    canvas = np.zeros((batch_size, channels, total_length, height), dtype=np.float32) if canvas is None else canvas
    
    
    x = np.zeros((batch_size, channels, 2*block_len, height), dtype=np.float32)
    mask = np.zeros_like(x)  # 1 = keep, 0 = generate
    mask[:, :, :block_len, :] = 1  # front half keeps the same
    print("mask: ", mask)
    target_len = 2*block_len
    # def predict(self, background_cond, autoreg_cond, external_cond, orig_x, mask, uncond_scale=None, n_sample=None):
    start = start_step
    starts = []
    while start <= (total_length - block_len):
        # max_l = 204
        starts.append(start)
        autoreg_cond_roll = -np.ones((batch_size, channels, autoreg_max_l, height), dtype=np.float32)
        max_autoregre_start_index = max(-1, start - autoreg_seg_lgth_step) // 16
        if max_autoregre_start_index >= 0:
            for batch_i in range(batch_size):    
                print(max_n_autoreg)
                num_autoregre_indices = np.random.choice(range(1,max_n_autoreg+1), p=n_autoreg_prob)
                print("number of indices:", num_autoregre_indices)
                possible_indices = list(range(0, max_autoregre_start_index + 1))
                if len(possible_indices) <= num_autoregre_indices:
                    # 範圍不足，直接全選
                    start_indices = np.array(possible_indices) * 16
                else:
                    # 隨機抽取 num_autoregre_indices 個
                    start_indices = random.sample(possible_indices, k=num_autoregre_indices)
                    #start_indices = [random.randint(0, max_autoregre_start_index + 1) for _ in range(num_autoregre_indices)]
                    start_indices = np.array(start_indices) * 16
                print("start_indices:", start_indices)
                for index, si in enumerate(start_indices):
                    autoreg_cond_roll[batch_i,:,index*tgtl:index*tgtl+autoreg_seg_lgth_step,:] = canvas[batch_i,:,si:si+autoreg_seg_lgth_step,:]
        else:
            autoreg_cond_roll = None
        mid = min(start + block_len, total_length)
        end = min(start + 2*block_len, total_length)
        length = end - start
        print(f"start: {start}, end: {end}")
        print(f"Use autoregressive condition: {(autoreg_cond_roll is not None)}")
        with torch.no_grad():
            # if start == 0
            if start == 0:
                orig_x = None
            else:
                orig_x = canvas[:, :, start:end, :]
                if length < target_len:
                    pad_len = target_len - length
                    # 在 dim=2 (時間軸) 右邊補 0
                    orig_x = np.pad(orig_x, ((0,0), (0,0), (0,pad_len), (0,0)), mode='constant')

            x = sampler.predict(
                None,
                autoreg_cond_roll,
                None,
                orig_x,
                mask,
                batch_size
            )
            #analyze_generation_result(x, start, block_len)
            analyze_generation_result(x, start)
            '''
            print("Number of elements > 0.5 of first song:")
            print(count_greater_than_half(x[0, :, 0:block_len, :]))
            print(count_greater_than_half(x[0, :, block_len:2*block_len, :]))
            '''
            canvas[:, :, start:end, :] = x[:, :, 0:length, :]

        start += block_len

    return starts, canvas

def pick_songs(num_of_songs, random_pick = True, total_songs = 2656, picked_list = None):
    
    if picked_list is None:
        picked_list = list(range(0, num_of_songs))
        if random_pick:
            picked_list = random.sample(list(range(0, total_songs)), k=num_of_songs)
        else:
            _, _, picked_list = load_split_file(os.path.join(DATASET_PATH, 'split2.npz'))
            picked_list = random.sample(list(picked_list), k=num_of_songs)
    '''
    picked_list = [3, 4, 10, 14, 18, 30, 41, 45, 53, 76, 81, 84, 86, 116, 120, 135, 139, 154,
    158, 195, 197, 204, 233, 236, 243, 244, 246, 258, 275, 279, 284, 293, 294, 303, 325, 336,
    365, 371, 372, 374, 383, 431, 440, 442, 455, 459, 465, 476, 482, 490, 506, 510, 528, 546,
    559, 562, 597]
    '''
    picked_list = [3]
    matrolls = []
    bpms = []
    phrases = []
    for index, song_id in enumerate(picked_list):
        label_number = 1
        song_name = str(song_id).zfill(4)
        label_name = str(song_id+1).zfill(3)
        print(f"batch_id:{index}, song_id:{song_id}")
        data_fn = os.path.join(DATASET_PATH, song_name)
        label_fn = os.path.join(LABEL_PATH, label_name)
        matfn = os.path.join(data_fn, f"{song_name}_mat.npz")
        infofn = os.path.join(data_fn, f"{song_name}_info.json")
        phrasefn = os.path.join(label_fn, f"human_label{label_number}.txt")
        with open(infofn, "r", encoding="utf-8") as f:
            info = json.load(f)
        mat = np.load(matfn, allow_pickle=True)
        mat = process_multi_channel_matrix(mat)
        matroll = note_end_matrix_to_piano_roll(mat)
        matrolls.append(matroll)
        bpm = info['bpm']
        bpm = int(bpm)
        bpms.append(bpm)
        if song_id < 909:
            phrases.append(process_phrase_txt(phrasefn))

    #matrolls = np.array(matrolls)
    return picked_list, matrolls, bpms, phrases

def fill_original_song(canvas, inpainting_length, random_pick = True, total_songs = 2656, picked_list = None):
    num_of_songs = canvas.shape[0]
    if picked_list is None:
        picked_list = list(range(0, num_of_songs))
        if random_pick:
            picked_list = random.sample(list(range(0, total_songs)), k=num_of_songs)
            
    for index, song_id in enumerate(picked_list):
        song_name = str(song_id).zfill(4)
        print(f"batch_id:{index}, song_id:{song_id}")
        data_fn = os.path.join(DATASET_PATH, song_name)
        matfn = os.path.join(data_fn, f"{song_name}_mat.npz")
        mat = np.load(matfn, allow_pickle=True)
        mat = process_multi_channel_matrix(mat)
        matroll = note_end_matrix_to_piano_roll(mat)
        canvas[index, :, 0:inpainting_length, :] = matroll[:, 0:inpainting_length, :]
        print(inpainting_length)
    return picked_list, canvas

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 10
    total_length = 512
    inpainting_length = total_length//2
    block_len = 64

    embeddingAnalyze = True
    inpainting = False
    generation = True
    
    canvas = np.zeros((batch_size, 2, total_length, 128), dtype=np.float32)

    params = PARAMS_DICTS['unsup']
    # Specify the model path
    model_path = os.path.join(MODEL_OUTPUT_PATH, 'unsup-a-u', '08-23_105359', 'chkpts', 'weights_best.pt')
    #model_path = os.path.join(MODEL_OUTPUT_PATH, 'unsup-a-', '08-23_105052', 'chkpts', 'weights_best.pt')
    #model_path = os.path.join(MODEL_OUTPUT_PATH, 'unsup-a-', '08-21_084534', 'chkpts', 'weights_best.pt')
    #model_path = os.path.join(MODEL_OUTPUT_PATH, 'unsup-a-', '08-17_130337', 'chkpts', 'weights_best.pt')
    
    generation_sampler = GenOpBase(params, model_path, device, True, False)

    dummy = torch.randn(32, 3, 224, 224).cuda()
    for _ in range(10):
        _ = dummy * dummy   # warm up
    torch.cuda.synchronize()

    if embeddingAnalyze:
        print("===================Start Analyze Embedding==============")
        analyze_embedding(generation_sampler)

    elif inpainting:
        print("================Start Music Inpainting===============")
        picked_list, canvas = fill_original_song(canvas, inpainting_length, True, 900)
        start = inpainting_length - block_len
        starts, generated_song = generate_full_song(
            generation_sampler,
            total_length=total_length,
            block_len=block_len,
            batch_size=batch_size,
            channels=2,
            height=128,
            device=device,
            start_step=start,
            canvas=canvas
        )
        print("Starts:", starts)
        print("picked_song_list:",picked_list)
        np.save("generated_song.npy", generated_song)
        print("Generated song shape:", generated_song.shape)
    elif generation:
        print("================Start Direct Music Generation===============")
        _, generated_song = generate_full_song(
            generation_sampler,
            total_length=total_length,
            block_len=block_len,
            batch_size=batch_size,
            channels=2,
            height=128,
            device=device
        )
        np.save("generated_song.npy", generated_song)
        print("Generated song shape:", generated_song.shape)
    else:
        g = checkerboard_kernel(2)
        print(g)
    
    
