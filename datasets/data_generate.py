import os
from tqdm import tqdm
import numpy as np
import soundfile as sf
from mido import MidiFile
from PIL import Image
import json

PROJECT_PATH = os.path.join(os.path.dirname(__file__), '..')
DATASET_PATH = os.path.join(PROJECT_PATH, 'datasets')

POP909_PATH = os.path.join(DATASET_PATH, 'pop909data', 'matched_pop909_acc')
POP1K7_PATH = os.path.join(DATASET_PATH, 'Pop1K7', 'midi_synchronized')

OUTPUT_PATH = os.path.join(DATASET_PATH, 'data')

channel_colors = {
    0: [1.0, 0.0, 0.0],  # 紅
    1: [0.0, 1.0, 0.0],  # 綠
    2: [0.0, 0.0, 1.0],  # 藍
    3: [0.0, 1.0, 1.0],  # 黃
    4: [1.0, 0.0, 1.0],  # 紫
    5: [0.0, 1.0, 1.0],  # 青
    6: [0.5, 0.5, 0.0],
    7: [0.0, 0.5, 0.5],
    8: [0.5, 0.0, 0.5],
    9: [0.3, 0.3, 0.3],
    10: [0.6, 0.3, 0.1],
    11: [0.2, 0.6, 0.1],
    12: [0.1, 0.3, 0.6],
    13: [0.8, 0.2, 0.4],
    14: [0.4, 0.8, 0.2],
    15: [0.2, 0.4, 0.8],
}

def midi_to_instrument_pianorolls(id, midi_path, outputfile_name="piano_roll", output_folder = None):
    midi = MidiFile(midi_path)
    # we have one beat accounts for how many ticks
    ticks_per_beat = midi.ticks_per_beat
    #print("ticks per beat: ", ticks_per_beat, "ticks/beat")
    instruments = {}
    abs_time = 0
    numerator = 4
    dominator = 4
    tempo = 0
    bpm = 0
    channel_program = {}
    for i, track in enumerate(midi.tracks):
        #print("This is the ", i, "th track:")
        abs_time = 0
        for msg in track:
            #print(msg)
            abs_time += msg.time
            if msg.is_meta:
                #print(msg)
                if msg.type == "time_signature":
                    numerator = msg.numerator
                    dominator = msg.denominator
                    #print("Time signature = ", numerator, "/", dominator)
                elif msg.type == "set_tempo":
                    tempo = msg.tempo
                    #print("Tempo = ", tempo, "mu-second/beat time=", msg.time)
                    bpm = 60000000.0 / tempo
                    #print("BPM = ", bpm, "beats/min")
            elif msg.type == 'program_change':
                #print("Program change at tick: ", abs_time, ", channel# = ", msg.channel, "; program# = ", msg.program)
                channel_program[msg.channel] = msg.program

            elif msg.type == 'note_on' and msg.velocity > 0:
                program = channel_program.get(msg.channel, 0)
                key = (msg.channel, program)
                instruments.setdefault(key, []).append({
                    'pitch': msg.note,
                    'start_tick': abs_time,
                    'velocity': msg.velocity,
                    'end_tick': None
                })
            elif (msg.type == 'note_off') or (msg.type == 'note_on' and msg.velocity == 0):
                key = (msg.channel, channel_program[msg.channel])
                for note in reversed(instruments[key]):
                    if note['pitch'] == msg.note:
                        if note['end_tick'] is None:
                            note['end_tick'] = abs_time
                            break
                        #else:
                            #print(f"{id}/msg:{msg}/No matching end tick, key:{key}")

    #print(instruments[(0,0)])
    max_tick = 0
    min_tick = 100000
    scaling = int(ticks_per_beat/4)
    #scaling = 1
    num_channels = 0
    for (channel, program), notes in instruments.items():
        num_channels = num_channels + 1
        min_tick_temp = min(note['start_tick'] for note in notes if note['start_tick'] is not None)
        max_tick_temp = max(note['end_tick'] for note in notes if note['end_tick'] is not None)
        min_tick_temp = min_tick_temp // scaling
        max_tick_temp = max_tick_temp // scaling
        #print("min_tick = ", min_tick_temp, " for channel ", channel)
        #print("max_tick = ",max_tick_temp, " for channel ", channel)
        if max_tick_temp > max_tick:
            max_tick = max_tick_temp
        if min_tick_temp < min_tick:
            min_tick = min_tick_temp
    #print("MAX tick = ", max_tick)
    #print("min tick = ", min_tick)
    #print("Number of channels =", num_channels)
    roll = np.zeros((128, max_tick + 1 - min_tick), dtype=np.uint8)
    canvas = np.ones((128, max_tick + 1 - min_tick, 3), dtype=np.float32)  # R,G,B 都為 1（白色）
    channel_data = {ch: [] for ch in range(num_channels)}
    shift = min_tick
    #print("Scaling = ", scaling, "shift = ", shift)

    for (channel, program), notes in instruments.items():
        base_color = channel_colors.get(channel, [0.0, 0.0, 0.0])
        #print("channel =", channel, " base_color =", base_color)
        ch = channel
        for note in notes:
            if note['end_tick'] is None:
                print("id:{id}/[Error]: No off time of note:{note}")
                continue
            pitch = note['pitch']
            if channel == 0:
                pitch = pitch # melody
            st = note['start_tick'] // scaling - shift
            et = note['end_tick'] // scaling - shift
            channel_data[ch].append([st, pitch, et])
            velocity = note['velocity'] / 127 # normalize to 0 ~ 1
            color = 1.0 - velocity * (1.0 - np.array(base_color))
            canvas[pitch, st:et, :] = color.reshape(1, 1, 3)

            #print("Note: pitch=", note['pitch'], " start=", st, " end=", et, " velocity=", note['velocity'])
            roll[note['pitch'], note['start_tick']:note['end_tick']] = note['velocity']

    canvas_uint8 = np.clip(canvas, 0, 1)  # 保險一下
    canvas_uint8 = (canvas_uint8 * 255).astype(np.uint8)

    # 上下翻轉，讓 pitch=127 在最上方
    canvas_uint8_flipped = np.flipud(canvas_uint8)

    # 建立 PIL Image 物件
    img = Image.fromarray(canvas_uint8_flipped, mode='RGB')

    # 儲存圖片（路徑自訂)
    imgfilename = f"{outputfile_name}.png"
    matfilename = f"{outputfile_name}_mat.npz"
    jsonfilename = f"{outputfile_name}_info.json"
    imgpath = os.path.join(output_folder, imgfilename)
    matpath = os.path.join(output_folder, matfilename)
    jsonpath = os.path.join(output_folder, jsonfilename)

    img.save(imgpath)
    for ch in channel_data:
        channel_data[ch] = np.array(channel_data[ch], dtype=int)
    np.savez_compressed(
        matpath,
        **{f"{ch}": arr for ch, arr in channel_data.items()}
    )

    midi_info = {
        "ticks_per_beat": ticks_per_beat,
        "tempo": tempo,
        "bpm": bpm,
        "time_signature": f"{numerator}/{dominator}",
        "num_channels": num_channels,
        "max_tick": max_tick,
        "min_tick": min_tick,
        "scaling": scaling
    }
    with open(jsonpath, "w", encoding="utf-8") as f:
        json.dump(midi_info, f, indent=4, ensure_ascii=False)
    
    return channel_data

def generate_909_data(id):
    for i in tqdm(range(1,910), desc ="Generating 909 roll information"):
        folder_name = f"{i:03d}" # 001 002 003...
        midi_path = os.path.join(POP909_PATH, folder_name, "aligned_demo.mid")
        if os.path.exists(midi_path):
            output_folder = os.path.join(OUTPUT_PATH, f"{id:04d}")
            os.makedirs(output_folder, exist_ok=True)
            _ = midi_to_instrument_pianorolls(id, midi_path, f"{id:04d}", output_folder)
            id = id + 1
        else:
            print(f"Warning, missing pop 909 file {midi_path}")
            id = id + 1
    return id

def generate_1k7_data(id):
    for src_idx in range(1,5): #src_001 to src_004
        src_folder = os.path.join(POP1K7_PATH, f'src_{src_idx:03d}')
        if not os.path.exists(src_folder):
            print("Warning, Mssing 1k7 folder {src_folder}")
            continue
        print(f"Current reading the {src_idx} folder of pop1k7")
        for file_name in tqdm(sorted(os.listdir(src_folder)), desc=f"Reading {src_folder}"):
            if file_name.lower().endswith(".mid"):
                midi_path = os.path.join(src_folder, file_name)
                output_folder = os.path.join(OUTPUT_PATH, f"{id:04d}")
                os.makedirs(output_folder, exist_ok=True)
                _ = midi_to_instrument_pianorolls(id, midi_path, f"{id:04d}", output_folder)
                id = id + 1
    return id

if __name__ == "__main__":
    id = 0
    id = generate_909_data(id)
    print(id)
    id = generate_1k7_data(id)
    print(id)