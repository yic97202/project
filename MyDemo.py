from resemblyzer import preprocess_wav, VoiceEncoder
from demo_utils import *
from pathlib import Path
from tqdm import tqdm
import numpy as np


## Load and preprocess the audio
data_dir = Path("audio_data", "donald_trump")
wav_fpaths = list(data_dir.glob("*.mp3"))
wavs = [preprocess_wav(wav_fpath) for wav_fpath in \
        tqdm(wav_fpaths, "Preprocessing wavs", len(wav_fpaths), unit=" utterances")]


## Compute the embeddings
encoder = VoiceEncoder()
embeds = np.array([encoder.embed_utterance(wav) for wav in wavs])
speakers = np.array([fpath.parent.name for fpath in wav_fpaths])
names = np.array([fpath.stem for fpath in wav_fpaths])


# Take 3 real embeddings at random, and leave the 3 others for testing
#gt_indices = np.random.choice(*np.where(speakers == "real"), 3, replace=False) 
mask = np.zeros(len(embeds), dtype=np.bool)
mask[0] = True
gt_embeds = embeds[mask]
gt_names = names[mask]
gt_speakers = speakers[mask]
embeds, speakers, names = embeds[~mask], speakers[~mask], names[~mask]



## Compare all embeddings against the ground truth embeddings, and compute the average similarities.
scores = (gt_embeds @ embeds.T).mean(axis=0)

"""
# Order the scores by decreasing order
sort = np.argsort(scores)[::-1]
scores, names, speakers = scores[sort], names[sort], speakers[sort]
"""

## Plot the scores
fig, _ = plt.subplots(figsize=(6, 6))
indices = np.arange(len(scores))
#plt.axhline(0.84, ls="dashed", label="Prediction threshold", c="black")
plt.bar(indices[:4], scores[:4], color="green", label="Cheng")
plt.bar(indices[4:], scores[4:], color="blue", label="Yeh")
plt.legend()
plt.xticks(indices, names, rotation="vertical", fontsize=8)
plt.xlabel("Ground truth: Cheng's voice")
plt.ylim(0, 1)
plt.ylabel("Similarity to ground truth")
fig.subplots_adjust(bottom=0.25)
plt.show()
