from bark.generation import preload_models
preload_models(
    text_use_gpu=True,
    text_use_small=False,
    coarse_use_gpu=True,
    coarse_use_small=False,
    fine_use_gpu=True,
    fine_use_small=False,
    codec_use_gpu=True,
    force_reload=False,
    path="models"
)

from bark.generation import load_codec_model, generate_text_semantic
from encodec.utils import convert_audio

import torchaudio
import torch

model = load_codec_model(use_gpu=True)


# Load and pre-process the audio waveform
audio_filepath = 'audio_CXM.wav'
text = "面对的是一个几十亿几百亿的市场，所以他们不会在乎这么一点的投入，他们不会想到他们根本做不成，你懂吗？"
audio_filepath = 'audio_CXM_short.wav' # the audio you want to clone (will get truncated so 5-10 seconds is probably fine, existing samples that I checked are around 7 seconds)
text = "所以他们不会在乎这么一点的投入"
#device = 'cuda' # or 'cpu'
device = 'cpu'
wav, sr = torchaudio.load(audio_filepath)
wav = convert_audio(wav, sr, model.sample_rate, model.channels)
wav = wav.unsqueeze(0).to(device)

# Extract discrete codes from EnCodec
with torch.no_grad():
    encoded_frames = model.encode(wav)
codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1).squeeze()  # [n_q, T]

# get seconds of audio
seconds = wav.shape[-1] / model.sample_rate
print(">>> generate semantic tokens")
semantic_tokens = generate_text_semantic(text, max_gen_duration_s=seconds, top_k=50, top_p=.95, temp=0.7) # not 100% sure on this part

# move codes to cpu
codes = codes.cpu().numpy()

import numpy as np
voice_name = 'CXM' # whatever you want the name of the voice to be
output_path = 'bark/assets/prompts/' + voice_name + '.npz'
np.savez(output_path, fine_prompt=codes, coarse_prompt=codes[:2, :], semantic_prompt=semantic_tokens)


##########################################

def generate_scope(text_prompt, voice_name, output_fp="./output/audio.wav"):
    print(">>> START Generation parts")
    from bark.api import generate_audio
    from transformers import BertTokenizer
    from bark.generation import SAMPLE_RATE, preload_models, codec_decode, generate_coarse, generate_fine, generate_text_semantic

    # # Enter your prompt and speaker here
    # text_prompt = "牛啊，这个就是私人管家了"
    # text_prompt = "可以白嫖亚马逊的"
    # voice_name = "CXM" # use your custom voice name here if you have one

    # load the tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

    # download and load all models
    preload_models(
        text_use_gpu=True,
        text_use_small=False,
        coarse_use_gpu=True,
        coarse_use_small=False,
        fine_use_gpu=True,
        fine_use_small=False,
        codec_use_gpu=True,
        force_reload=False,
        path="models"
    )

    # simple generation
    audio_array = generate_audio(text_prompt, history_prompt=voice_name, text_temp=0.7, waveform_temp=0.7)
    # generation with more control
    x_semantic = generate_text_semantic(
        text_prompt,
        history_prompt=voice_name,
        temp=0.7,
        top_k=50,
        top_p=0.95,
    )

    x_coarse_gen = generate_coarse(
        x_semantic,
        history_prompt=voice_name,
        temp=0.7,
        top_k=50,
        top_p=0.95,
    )
    x_fine_gen = generate_fine(
        x_coarse_gen,
        history_prompt=voice_name,
        temp=0.5,
    )
    audio_array = codec_decode(x_fine_gen)
    from scipy.io.wavfile import write as write_wav
    # save audio
    filepath = output_fp # change this to your desired output path
    write_wav(filepath, SAMPLE_RATE, audio_array)

generate_scope(text, "CXM", "./output/audio_1.wav")
generate_scope("可以白嫖亚马逊的", "CXM", "./output/audio_1.wav")
generate_scope("牛啊，人工智能管家", "CXM", "./output/audio_1.wav")
#generate_scope("", "CXM", "./output/audio_1.wav")