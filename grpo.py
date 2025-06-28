import soundfile as sf
import os
import json
import random
import tempfile
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')
os.environ["MPLBACKEND"] = "agg"
from transformers import pipeline
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import librosa
import numpy as np
from tqdm import tqdm
from huggingface_hub import hf_hub_download
import jiwer
from scipy.spatial.distance import cosine
import soundfile as sf
import gc

# Import Chatterbox components
from chatterbox.tts import ChatterboxTTS, punc_norm
from chatterbox.models.s3gen import S3Gen, S3GEN_SR
from chatterbox.models.s3tokenizer import S3_SR
from chatterbox.models.voice_encoder import VoiceEncoder
from chatterbox.models.tokenizers.tokenizer import EnTokenizer, SpanishTokenizer
from chatterbox.models.t3.modules.cond_enc import T3Cond
from chatterbox.models.t3.modules.t3_config import T3SpanishConfig, resize_t3_embeddings

# Add matplotlib imports for metrics tracking
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
from datetime import datetime
import threading
import time
from collections import deque

# Hardcoded configuration
AUDIO_DATA_DIR = "./audio_data"
BATCH_SIZE = 10
EPOCHS = 2
LEARNING_RATE = 1e-5
WARMUP_STEPS = 500
MAX_AUDIO_LENGTH = 400.0
MIN_AUDIO_LENGTH = 1.0
LORA_RANK = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
GRADIENT_ACCUMULATION_STEPS = 4
SAVE_EVERY_N_STEPS = 200
CHECKPOINT_DIR = "checkpoints_grpo"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WHISPER_MODEL = "openai/whisper-large-v3-turbo"
MAX_TEXT_LENGTH = 1000
VALIDATION_SPLIT = 0.1

# GRPO specific parameters
NUM_SAMPLES_PER_INPUT = 2
KL_COEFF = 0.01
REWARD_BASELINE_MOMENTUM = 0.9
TEMPERATURE = 1.0
TOP_K = 50
TOP_P = 0.95

# Reward weights
WER_WEIGHT = -1.0
SPEAKER_SIM_WEIGHT = 1.0
LENGTH_PENALTY_WEIGHT = -0.5

TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

#TARGET_MODULES = [
    #"q_proj", "v_proj", "k_proj", "o_proj", 
    #"gate_proj", "up_proj", "down_proj",
    #"embed_tokens",  # Text embeddings
    #"lm_head"        # Output projection
#]




def safe_tensor_index(tensor: torch.Tensor, start: int, end: int, dim: int = 1) -> torch.Tensor:
    """Safely index into a tensor with bounds checking"""
    if tensor.numel() == 0:
        # Return empty tensor with correct shape
        shape = list(tensor.shape)
        shape[dim] = 0
        return torch.empty(shape, dtype=tensor.dtype, device=tensor.device)
    
    tensor_size = tensor.size(dim)
    
    # Ensure we have valid bounds
    if tensor_size == 0:
        shape = list(tensor.shape)
        shape[dim] = 0
        return torch.empty(shape, dtype=tensor.dtype, device=tensor.device)
    
    # Clamp indices to valid range
    start = max(0, min(start, tensor_size))
    end = max(start, min(end, tensor_size))
    
    if start >= end or start >= tensor_size:
        # Return empty tensor with correct shape
        shape = list(tensor.shape)
        shape[dim] = 0
        return torch.empty(shape, dtype=tensor.dtype, device=tensor.device)
    
    # Create slice indices
    indices = [slice(None)] * tensor.dim()
    indices[dim] = slice(start, end)
    
    try:
        return tensor[tuple(indices)]
    except (IndexError, RuntimeError) as e:
        print(f"Tensor indexing error: {e}, tensor shape: {tensor.shape}, start: {start}, end: {end}, dim: {dim}")
        shape = list(tensor.shape)
        shape[dim] = 0
        return torch.empty(shape, dtype=tensor.dtype, device=tensor.device)


def safe_gather(input_tensor: torch.Tensor, dim: int, index: torch.Tensor) -> torch.Tensor:
    """Safely gather from tensor with bounds checking"""
    if input_tensor.numel() == 0 or index.numel() == 0:
        return torch.zeros_like(index, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Get the size of the dimension we're gathering from
    max_index = input_tensor.size(dim) - 1
    if max_index < 0:
        return torch.zeros_like(index, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Clamp all indices to valid range
    safe_index = torch.clamp(index, 0, max_index)
    
    try:
        return torch.gather(input_tensor, dim, safe_index)
    except (IndexError, RuntimeError) as e:
        print(f"Gather error: {e}, input shape: {input_tensor.shape}, index shape: {index.shape}, dim: {dim}")
        return torch.zeros_like(index, dtype=input_tensor.dtype, device=input_tensor.device)


def validate_tensor_operation(tensor: torch.Tensor, operation: str) -> bool:
    """Validate tensor before operations to prevent CUDA errors"""
    if tensor is None:
        print(f"Warning: {operation} - tensor is None")
        return False
    
    if not torch.is_tensor(tensor):
        print(f"Warning: {operation} - not a tensor")
        return False
    
    if tensor.numel() == 0:
        print(f"Warning: {operation} - empty tensor")
        return False
    
    if torch.isnan(tensor).any():
        print(f"Warning: {operation} - tensor contains NaN")
        return False
    
    if torch.isinf(tensor).any():
        print(f"Warning: {operation} - tensor contains Inf")
        return False
    
    return True


class GRPOMetricsTracker:
    def __init__(self, save_path="grpo_training_metrics.png", update_interval=2.0):
        self.save_path = save_path
        self.update_interval = update_interval
        self.metrics = {
            'train_loss': deque(maxlen=1000),
            'val_loss': deque(maxlen=100),
            'learning_rate': deque(maxlen=1000),
            'steps': deque(maxlen=1000),
            'epochs': deque(maxlen=1000),
            'batch_loss': deque(maxlen=100),
            'gradient_norm': deque(maxlen=1000),
            'avg_reward': deque(maxlen=1000),
            'wer_score': deque(maxlen=1000),
            'speaker_sim': deque(maxlen=1000),
            'length_penalty': deque(maxlen=1000),
            'kl_divergence': deque(maxlen=1000),
            'baseline_reward': deque(maxlen=1000),
        }
        self.start_time = time.time()
        self.last_update = 0
        self.running = True
        self.lock = threading.Lock()
        
        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(24, 14))
        self.fig.suptitle('Chatterbox TTS GRPO Training Metrics', fontsize=16, fontweight='bold')
        
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()
        
        self._create_initial_plot()
    
    def _create_initial_plot(self):
        """Create the initial plot layout"""
        self.fig.clf()
        
        gs = self.fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        self.ax_loss = self.fig.add_subplot(gs[0, :2])
        self.ax_reward = self.fig.add_subplot(gs[0, 2:])
        self.ax_wer = self.fig.add_subplot(gs[1, 0])
        self.ax_speaker = self.fig.add_subplot(gs[1, 1])
        self.ax_length = self.fig.add_subplot(gs[1, 2])
        self.ax_kl = self.fig.add_subplot(gs[1, 3])
        self.ax_lr = self.fig.add_subplot(gs[2, 0])
        self.ax_grad = self.fig.add_subplot(gs[2, 1])
        self.ax_baseline = self.fig.add_subplot(gs[2, 2:])
        self.ax_info = self.fig.add_subplot(gs[3, :2])
        self.ax_epoch = self.fig.add_subplot(gs[3, 2:])
        
        self.ax_info.axis('off')
        
        self.ax_loss.set_title('Training Loss', fontweight='bold')
        self.ax_reward.set_title('Average Reward', fontweight='bold')
        self.ax_wer.set_title('WER Score', fontweight='bold')
        self.ax_speaker.set_title('Speaker Similarity', fontweight='bold')
        self.ax_length.set_title('Length Penalty', fontweight='bold')
        self.ax_kl.set_title('KL Divergence', fontweight='bold')
        self.ax_lr.set_title('Learning Rate', fontweight='bold')
        self.ax_grad.set_title('Gradient Norm', fontweight='bold')
        self.ax_baseline.set_title('Baseline Reward', fontweight='bold')
        self.ax_epoch.set_title('Rewards by Epoch', fontweight='bold')
        
        for ax in [self.ax_loss, self.ax_reward, self.ax_wer, self.ax_speaker,
                   self.ax_length, self.ax_kl, self.ax_lr, self.ax_grad,
                   self.ax_baseline, self.ax_epoch]:
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self.fig.savefig(self.save_path, dpi=100, bbox_inches='tight', facecolor='black')
    
    def add_metrics(self, **kwargs):
        """Add metrics to the tracker"""
        with self.lock:
            for key, value in kwargs.items():
                if key in self.metrics and value is not None:
                    self.metrics[key].append(value)
            self.last_update = time.time()
    
    def _update_loop(self):
        """Background thread to update plots"""
        while self.running:
            time.sleep(self.update_interval)
            if time.time() - self.last_update < self.update_interval * 2:
                self._update_plot()
    
    def _update_plot(self):
        """Update the plot with current metrics"""
        with self.lock:
            try:
                for ax in [self.ax_loss, self.ax_reward, self.ax_wer, self.ax_speaker,
                          self.ax_length, self.ax_kl, self.ax_lr, self.ax_grad,
                          self.ax_baseline, self.ax_epoch]:
                    ax.clear()
                
                if len(self.metrics['train_loss']) > 0:
                    steps = list(self.metrics['steps'])[-len(self.metrics['train_loss']):]
                    self.ax_loss.plot(steps, list(self.metrics['train_loss']), 
                                     'b-', label='Train Loss', linewidth=2)
                    self.ax_loss.set_ylim(bottom=0)
                    self.ax_loss.legend()
                    self.ax_loss.set_title('Training Loss', fontweight='bold')
                    self.ax_loss.set_xlabel('Steps')
                    self.ax_loss.set_ylabel('Loss')
                    self.ax_loss.grid(True, alpha=0.3)
                
                if len(self.metrics['avg_reward']) > 0:
                    steps = list(self.metrics['steps'])[-len(self.metrics['avg_reward']):]
                    self.ax_reward.plot(steps, list(self.metrics['avg_reward']), 
                                       'g-', linewidth=2)
                    self.ax_reward.set_title('Average Reward', fontweight='bold')
                    self.ax_reward.set_xlabel('Steps')
                    self.ax_reward.set_ylabel('Reward')
                    self.ax_reward.grid(True, alpha=0.3)
                
                if len(self.metrics['wer_score']) > 0:
                    steps = list(self.metrics['steps'])[-len(self.metrics['wer_score']):]
                    self.ax_wer.plot(steps, list(self.metrics['wer_score']), 
                                    'r-', linewidth=2)
                    self.ax_wer.set_title('WER Score', fontweight='bold')
                    self.ax_wer.set_xlabel('Steps')
                    self.ax_wer.set_ylabel('WER')
                    self.ax_wer.grid(True, alpha=0.3)
                
                if len(self.metrics['speaker_sim']) > 0:
                    steps = list(self.metrics['steps'])[-len(self.metrics['speaker_sim']):]
                    self.ax_speaker.plot(steps, list(self.metrics['speaker_sim']), 
                                        'c-', linewidth=2)
                    self.ax_speaker.set_title('Speaker Similarity', fontweight='bold')
                    self.ax_speaker.set_xlabel('Steps')
                    self.ax_speaker.set_ylabel('Similarity')
                    self.ax_speaker.grid(True, alpha=0.3)
                
                if len(self.metrics['length_penalty']) > 0:
                    steps = list(self.metrics['steps'])[-len(self.metrics['length_penalty']):]
                    self.ax_length.plot(steps, list(self.metrics['length_penalty']), 
                                       'm-', linewidth=2)
                    self.ax_length.set_title('Length Penalty', fontweight='bold')
                    self.ax_length.set_xlabel('Steps')
                    self.ax_length.set_ylabel('Penalty')
                    self.ax_length.grid(True, alpha=0.3)
                
                if len(self.metrics['kl_divergence']) > 0:
                    steps = list(self.metrics['steps'])[-len(self.metrics['kl_divergence']):]
                    self.ax_kl.plot(steps, list(self.metrics['kl_divergence']), 
                                   'orange', linewidth=2)
                    self.ax_kl.set_title('KL Divergence', fontweight='bold')
                    self.ax_kl.set_xlabel('Steps')
                    self.ax_kl.set_ylabel('KL')
                    self.ax_kl.grid(True, alpha=0.3)
                
                if len(self.metrics['learning_rate']) > 0:
                    steps = list(self.metrics['steps'])[-len(self.metrics['learning_rate']):]
                    self.ax_lr.plot(steps, list(self.metrics['learning_rate']), 
                                   'g-', linewidth=2)
                    self.ax_lr.set_title('Learning Rate', fontweight='bold')
                    self.ax_lr.set_xlabel('Steps')
                    self.ax_lr.set_ylabel('LR')
                    self.ax_lr.grid(True, alpha=0.3)
                    self.ax_lr.ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
                
                if len(self.metrics['gradient_norm']) > 0:
                    steps = list(self.metrics['steps'])[-len(self.metrics['gradient_norm']):]
                    self.ax_grad.plot(steps, list(self.metrics['gradient_norm']), 
                                     'lime', linewidth=2)
                    self.ax_grad.set_title('Gradient Norm', fontweight='bold')
                    self.ax_grad.set_xlabel('Steps')
                    self.ax_grad.set_ylabel('Norm')
                    self.ax_grad.grid(True, alpha=0.3)
                
                if len(self.metrics['baseline_reward']) > 0:
                    steps = list(self.metrics['steps'])[-len(self.metrics['baseline_reward']):]
                    self.ax_baseline.plot(steps, list(self.metrics['baseline_reward']), 
                                         'yellow', linewidth=2)
                    self.ax_baseline.set_title('Baseline Reward', fontweight='bold')
                    self.ax_baseline.set_xlabel('Steps')
                    self.ax_baseline.set_ylabel('Baseline')
                    self.ax_baseline.grid(True, alpha=0.3)
                
                self.ax_info.clear()
                self.ax_info.axis('off')
                
                info_text = [
                    f"GRPO Training Information",
                    f"{'='*30}",
                    f"Device: {DEVICE}",
                    f"Batch Size: {BATCH_SIZE}",
                    f"Samples per Input: {NUM_SAMPLES_PER_INPUT}",
                    f"KL Coefficient: {KL_COEFF}",
                    f"Temperature: {TEMPERATURE}",
                    f"",
                    f"Reward Weights:",
                    f"  WER: {WER_WEIGHT}",
                    f"  Speaker Sim: {SPEAKER_SIM_WEIGHT}",
                    f"  Length: {LENGTH_PENALTY_WEIGHT}",
                    f"",
                    f"Current Stats",
                    f"{'='*30}",
                ]
                
                if len(self.metrics['steps']) > 0:
                    current_step = self.metrics['steps'][-1]
                    info_text.append(f"Step: {current_step}")
                
                if len(self.metrics['epochs']) > 0:
                    current_epoch = self.metrics['epochs'][-1]
                    info_text.append(f"Epoch: {current_epoch}/{EPOCHS}")
                
                if len(self.metrics['avg_reward']) > 0:
                    current_reward = self.metrics['avg_reward'][-1]
                    info_text.append(f"Current Reward: {current_reward:.4f}")
                
                elapsed_time = time.time() - self.start_time
                info_text.append(f"")
                info_text.append(f"Time Elapsed: {elapsed_time/3600:.2f}h")
                
                self.ax_info.text(0.05, 0.95, '\n'.join(info_text), 
                                 transform=self.ax_info.transAxes,
                                 fontsize=10, verticalalignment='top',
                                 fontfamily='monospace',
                                 bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))
                
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.fig.text(0.99, 0.01, f"Last updated: {timestamp}", 
                             ha='right', va='bottom', fontsize=8, color='gray')
                
                self.fig.savefig(self.save_path, dpi=100, bbox_inches='tight', facecolor='black')
                
            except Exception as e:
                print(f"Error updating plot: {e}")
    
    def stop(self):
        """Stop the metrics tracker"""
        self.running = False
        if hasattr(self, 'update_thread') and self.update_thread.is_alive():
            self.update_thread.join()
        plt.close(self.fig)


class LoRALayer(nn.Module):
    """LoRA adapter layer"""
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        alpha: float = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) / np.sqrt(rank))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.lora_dropout = nn.Dropout(dropout)
        
        nn.init.normal_(self.lora_A, mean=0.0, std=1.0/np.sqrt(rank))
        nn.init.zeros_(self.lora_B)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not validate_tensor_operation(x, "LoRA forward"):
            return torch.zeros_like(x)
        
        result = self.lora_dropout(x)
        result = result @ self.lora_A.T @ self.lora_B.T
        return result * self.scaling


def inject_lora_layers(model: nn.Module, target_modules: List[str], rank: int, alpha: float, dropout: float):
    lora_layers = {}
    device = next(model.parameters()).device
    
    for name, module in model.named_modules():
        if any(target in name for target in target_modules):
            if isinstance(module, nn.Linear):
                if min(module.in_features, module.out_features) < rank:
                    continue
                    
                lora_layer = LoRALayer(
                    module.in_features,
                    module.out_features,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout
                )
                lora_layer = lora_layer.to(device)
                lora_layers[name] = lora_layer
                
                original_forward = module.forward
                def make_new_forward(orig_forward, lora):
                    def new_forward(x):
                        base_output = orig_forward(x)
                        lora_output = lora(x)
                        return base_output + lora_output
                    return new_forward
                
                module.forward = make_new_forward(original_forward, lora_layer)
    
    return lora_layers


@dataclass
class AudioSample:
    """Container for audio sample data"""
    audio_path: Path
    transcript: str
    duration: float
    sample_rate: int


class TTSDataset(Dataset):
    """Dataset handling with improved error checking"""
    def __init__(
        self,
        samples: List[AudioSample],
        tokenizer: EnTokenizer,
        s3_sr: int = S3_SR,
        s3gen_sr: int = S3GEN_SR,
        max_audio_length: float = MAX_AUDIO_LENGTH,
        max_text_length: int = MAX_TEXT_LENGTH,
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.s3_sr = s3_sr
        self.s3gen_sr = s3gen_sr
        self.max_audio_length = max_audio_length
        self.max_text_length = max_text_length
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        try:
            audio, sr = librosa.load(sample.audio_path, sr=self.s3gen_sr)
            audio = librosa.util.normalize(audio)
            
            max_samples = int(self.max_audio_length * self.s3gen_sr)
            if len(audio) > max_samples:
                audio = audio[:max_samples]
            else:
                pad_amount = max_samples - len(audio)
                audio = np.pad(audio, (0, pad_amount), mode='constant', constant_values=0)
            
            audio_16k = librosa.resample(audio, orig_sr=self.s3gen_sr, target_sr=self.s3_sr)
            
            text = punc_norm(sample.transcript)
            if len(text) > self.max_text_length:
                text = text[:self.max_text_length]
            
            return {
                'audio': torch.FloatTensor(audio),
                'audio_16k': torch.FloatTensor(audio_16k),
                'text': text,
                'transcript': sample.transcript,
                'audio_path': str(sample.audio_path),
                'duration': sample.duration,
            }
        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            # Return dummy data instead of failing
            dummy_audio = np.zeros(int(self.max_audio_length * self.s3gen_sr))
            dummy_audio_16k = np.zeros(int(self.max_audio_length * self.s3_sr))
            
            return {
                'audio': torch.FloatTensor(dummy_audio),
                'audio_16k': torch.FloatTensor(dummy_audio_16k),
                'text': "dummy text",
                'transcript': "dummy transcript",
                'audio_path': str(sample.audio_path),
                'duration': 1.0,
            }


def prepare_batch_conditionals(
    batch: Dict[str, torch.Tensor],
    model: ChatterboxTTS,
    ve: VoiceEncoder,
    s3gen: S3Gen,
) -> Tuple[T3Cond, List[dict]]:
    B = batch['audio'].size(0)
    device = model.device

    ve_embeds = []
    for i in range(B):
        try:
            wav_16k = batch['audio_16k'][i].cpu().numpy()
            
            if len(wav_16k) < S3_SR:
                wav_16k = np.pad(wav_16k, (0, S3_SR - len(wav_16k)), mode='reflect')
            
            utt_embeds = ve.embeds_from_wavs([wav_16k],
                                             sample_rate=S3_SR,
                                             as_spk=False,
                                             batch_size=8,
                                             rate=1.3,
                                             overlap=0.5)

            parts = torch.from_numpy(utt_embeds)
            ref = parts[0].unsqueeze(0)
            sims = F.cosine_similarity(parts, ref, dim=-1)
            voiced = parts[sims > 0.6]
            ve_embed = voiced.mean(0, keepdim=True) if len(voiced) > 0 else parts.mean(0, keepdim=True)
            ve_embeds.append(ve_embed)
        except Exception as e:
            print(f"Error in voice embedding {i}: {e}")
            if ve_embeds:
                ve_embed = ve_embeds[-1].clone()
            else:
                ve_embed = torch.zeros(1, 256)
            ve_embeds.append(ve_embed)

    ve_embeds = torch.cat(ve_embeds, dim=0).to(device)

    s3gen_refs = []
    for i in range(B):
        try:
            audio = batch['audio'][i].cpu().numpy()
            ref_length = min(len(audio), model.DEC_COND_LEN)
            ref_audio = audio[:ref_length]
            if len(ref_audio) < model.DEC_COND_LEN:
                ref_audio = np.pad(ref_audio, (0, model.DEC_COND_LEN - len(ref_audio)), mode='constant')
            s3gen_refs.append(s3gen.embed_ref(ref_audio, S3GEN_SR, device=device))
        except Exception as e:
            print(f"Error in S3Gen ref {i}: {e}")
            ref_audio = np.zeros(model.DEC_COND_LEN)
            s3gen_refs.append(s3gen.embed_ref(ref_audio, S3GEN_SR, device=device))

    t3_tokzr = s3gen.tokenizer
    plen = model.t3.hp.speech_cond_prompt_len
    tok_list = []
    if plen > 0:
        for i in range(B):
            try:
                wav_16k = batch['audio_16k'][i].cpu().numpy()
                ref_length = min(len(wav_16k), model.ENC_COND_LEN)
                ref_16k = wav_16k[:ref_length]
                
                if len(ref_16k) < S3_SR // 2:
                    ref_16k = np.pad(ref_16k, (0, S3_SR // 2 - len(ref_16k)), mode='reflect')
                
                tokens, _ = t3_tokzr.forward([ref_16k], max_len=plen)
                
                # Ensure tokens is a 2D tensor
                if isinstance(tokens, np.ndarray):
                    tokens = torch.from_numpy(tokens)
                if tokens.dim() == 1:
                    tokens = tokens.unsqueeze(0)
                
                # Validate token values are within bounds
                vocab_size = getattr(t3_tokzr, 'vocab_size', 1024)
                tokens = torch.clamp(tokens, 0, vocab_size - 1)
                
                tok_list.append(tokens)
            except Exception as e:
                print(f"Error tokenizing speech {i}: {e}")
                dummy_tokens = torch.zeros(1, plen, dtype=torch.long)
                tok_list.append(dummy_tokens)
        
        if tok_list:
            t3_cond_tokens = torch.cat(tok_list, dim=0).to(device)
        else:
            t3_cond_tokens = torch.empty(B, 0, dtype=torch.long, device=device)
    else:
        t3_cond_tokens = torch.empty(B, 0, dtype=torch.long, device=device)

    t3_cond = T3Cond(
        speaker_emb=ve_embeds,
        cond_prompt_speech_tokens=t3_cond_tokens,
        emotion_adv=0.5 * torch.ones(B, 1, 1, device=device),
    )

    return t3_cond, s3gen_refs


def generate_samples(
    model: ChatterboxTTS,
    batch: Dict[str, torch.Tensor],
    t3_cond: T3Cond,
    s3gen_refs: List[dict],
    num_samples: int = NUM_SAMPLES_PER_INPUT,
    temperature: float = TEMPERATURE,
    top_k: int = TOP_K,
    top_p: float = TOP_P,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Generate multiple samples"""
    batch_size = batch['audio'].size(0)
    device = model.device
    samples = []
    
    # Get pad token ID safely
    pad_token_id = getattr(model.tokenizer, 'pad_token_id', 0)
    
    with torch.cuda.amp.autocast(enabled=(DEVICE == 'cuda'), dtype=torch.float16):
        with torch.no_grad():
            for sample_idx in range(num_samples):
                if DEVICE == 'cuda':
                    torch.cuda.empty_cache()
                
                try:
                    text_tokens_list = []
                    for i in range(batch_size):
                        text = batch['text'][i]
                        tokens = model.tokenizer.text_to_tokens(text).to(device)
                        text_tokens_list.append(tokens)
                    
                    if not text_tokens_list:
                        continue
                    
                    max_text_len = max(t.size(-1) for t in text_tokens_list if t.numel() > 0)
                    if max_text_len == 0:
                        continue
                        
                    text_tokens_padded = []
                    for t in text_tokens_list:
                        if t.numel() == 0:
                            padded = torch.zeros(1, max_text_len, dtype=torch.long, device=device)
                        else:
                            pad_amount = max_text_len - t.size(-1)
                            if pad_amount > 0:
                                padded = F.pad(t, (0, pad_amount), value=pad_token_id)
                            else:
                                padded = t
                        text_tokens_padded.append(padded)
                    
                    text_tokens = torch.cat(text_tokens_padded, dim=0)
                    
                    sot = model.t3.hp.start_text_token
                    eot = model.t3.hp.stop_text_token
                    
                    # Ensure proper start/end tokens
                    if text_tokens.size(1) == 0 or text_tokens[0, 0] != sot:
                        text_tokens = F.pad(text_tokens, (1, 0), value=sot)
                    if text_tokens.size(1) == 0 or text_tokens[0, -1] != eot:
                        text_tokens = F.pad(text_tokens, (0, 1), value=eot)
                    
                    max_speech_len = min(256, int(MAX_AUDIO_LENGTH * S3_SR / 320))
                    
                    # Double everything for generating two samples
                    text_tokens_doubled = torch.cat([text_tokens, text_tokens], dim=0)
                    t3_cond_doubled = T3Cond(
                        speaker_emb=torch.cat([t3_cond.speaker_emb, t3_cond.speaker_emb], dim=0),
                        cond_prompt_speech_tokens=torch.cat(
                            [t3_cond.cond_prompt_speech_tokens, t3_cond.cond_prompt_speech_tokens], dim=0
                        ) if t3_cond.cond_prompt_speech_tokens.numel() > 0 else torch.empty(batch_size * 2, 0, dtype=torch.long, device=device),
                        emotion_adv=torch.cat(
                            [t3_cond.emotion_adv, t3_cond.emotion_adv], dim=0
                        ) if t3_cond.emotion_adv is not None else None,
                    )
                    
                    # Initialize with empty speech tokens
                    empty_speech = torch.empty(batch_size * 2, 0, dtype=torch.long, device=device)
                    
                    embeds, len_cond = model.t3.prepare_input_embeds(
                        t3_cond=t3_cond_doubled,
                        text_tokens=text_tokens_doubled,
                        speech_tokens=empty_speech,
                    )
                    
                    if not validate_tensor_operation(embeds, "initial embeds"):
                        continue
                    
                    generated_tokens = []
                    max_context_len = 512  # Reduced for safety
                    vocab_size = getattr(model.t3, 'speech_vocab_size', 1024)
                    
                    for step in range(max_speech_len):
                        # Truncate if too long
                        if embeds.size(1) > max_context_len:
                            embeds = embeds[:, -max_context_len:]
                        
                        if not validate_tensor_operation(embeds, f"embeds step {step}"):
                            break
                        
                        hidden_states = model.t3.tfmr(inputs_embeds=embeds)[0]
                        
                        if not validate_tensor_operation(hidden_states, f"hidden states step {step}"):
                            break
                            
                        if hidden_states.size(1) == 0:
                            break
                            
                        speech_logits = model.t3.speech_head(hidden_states[:, -1:])
                        
                        if not validate_tensor_operation(speech_logits, f"speech logits step {step}"):
                            break
                        
                        speech_logits = speech_logits / temperature
                        
                        # Apply top-k top-p filtering
                        filtered_logits = top_k_top_p_filtering(speech_logits[0, 0], top_k=top_k, top_p=top_p)
                        
                        # Clamp logits to prevent overflow
                        filtered_logits = torch.clamp(filtered_logits, -10.0, 10.0)
                        
                        probs = F.softmax(filtered_logits, dim=-1)
                        
                        if not validate_tensor_operation(probs, f"probs step {step}"):
                            break
                            
                        next_token = torch.multinomial(probs, num_samples=1)
                        
                        # Validate token is within vocab bounds
                        if next_token.item() >= vocab_size:
                            next_token = torch.tensor([vocab_size - 1], device=device, dtype=torch.long)
                        
                        if next_token.item() == model.t3.hp.stop_speech_token:
                            break
                        
                        generated_tokens.append(next_token)
                        
                        # Get speech embedding
                        if hasattr(model.t3, 'speech_embed'):
                            next_embed = model.t3.speech_embed(next_token.unsqueeze(0).expand(batch_size * 2, -1))
                        elif hasattr(model.t3, 'speech_emb'):
                            next_embed = model.t3.speech_emb(next_token.unsqueeze(0).expand(batch_size * 2, -1))
                        else:
                            print("Warning: Could not find speech embedding layer")
                            break
                            
                        if not validate_tensor_operation(next_embed, f"next embed step {step}"):
                            break
                            
                        embeds = torch.cat([embeds, next_embed], dim=1)
                    
                    if generated_tokens:
                        speech_tokens = torch.cat(generated_tokens, dim=0).unsqueeze(0)
                    else:
                        speech_tokens = torch.empty(1, 0, dtype=torch.long, device=device)
                    
                    samples.append((speech_tokens, text_tokens[0]))
                    
                    del embeds, hidden_states, speech_logits
                    if DEVICE == 'cuda':
                        torch.cuda.empty_cache()
                        
                except Exception as e:
                    print(f"Error generating sample {sample_idx}: {e}")
                    import traceback
                    traceback.print_exc()
                    dummy_speech = torch.empty(1, 0, dtype=torch.long, device=device)
                    dummy_text = torch.tensor([model.t3.hp.start_text_token, model.t3.hp.stop_text_token], device=device)
                    samples.append((dummy_speech, dummy_text))
    
    return samples

def compute_rewards(
    model: "ChatterboxTTS",
    samples: List[Tuple[torch.Tensor, torch.Tensor]],
    batch: Dict[str, torch.Tensor],
    t3_cond: T3Cond,
    s3gen_refs: List[dict],
    whisper_model,
    *,
    wer_weight: float = 1.0,
    speaker_sim_weight: float = 1.0,
    length_penalty_weight: float = 1.0,
    min_tok_for_synth: int = 3,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute rewards"""
    
    device, sr_gen = model.device, model.sr
    rew, wer_vals, sim_vals, lp_vals = [], [], [], []

    def safe_flatten_to_numpy(tensor_or_array):
        try:
            if isinstance(tensor_or_array, torch.Tensor):
                arr = tensor_or_array.detach().cpu().numpy()
            elif isinstance(tensor_or_array, np.ndarray):
                arr = tensor_or_array
            else:
                arr = np.array(tensor_or_array)
            
            return arr.flatten()
        except Exception as e:
            print(f"Error flattening tensor: {e}")
            return np.zeros(256)

    try:
        ref_speaker_emb = t3_cond.speaker_emb
        if isinstance(ref_speaker_emb, (list, tuple)):
            ref_speaker_emb = ref_speaker_emb[0]
        elif ref_speaker_emb.dim() > 1 and ref_speaker_emb.size(0) > 1:
            ref_speaker_emb = ref_speaker_emb[0]
        
        ref_speaker_emb = safe_flatten_to_numpy(ref_speaker_emb)
        
    except Exception as e:
        print(f"Error extracting reference speaker embedding: {e}")
        ref_speaker_emb = np.zeros(256)

    for i, (speech_tok, _) in enumerate(samples):
        try:
            if isinstance(speech_tok, np.ndarray):
                speech_tok = torch.as_tensor(speech_tok, dtype=torch.long)
            if not torch.is_tensor(speech_tok):
                raise TypeError("speech_tokens must be tensor or ndarray")

            if speech_tok.numel() < min_tok_for_synth:
                raise ValueError("too few speech tokens for reliable synthesis")

            speech_tok = speech_tok.to(device)

            with torch.cuda.amp.autocast(enabled=(DEVICE == 'cuda'), dtype=torch.float16):
                with torch.no_grad():
                    try:
                        if speech_tok.dim() == 1:
                            speech_tok_batch = speech_tok.unsqueeze(0)
                        else:
                            speech_tok_batch = speech_tok

                        if speech_tok_batch.size(1) == 0:
                            raise ValueError("Empty speech tokens")

                        # Validate speech tokens are within bounds
                        vocab_size = getattr(model.s3gen.tokenizer, 'vocab_size', 1024)
                        speech_tok_batch = torch.clamp(speech_tok_batch, 0, vocab_size - 1)

                        mel = model.s3gen.flow_inference(
                            speech_tokens=speech_tok_batch,
                            ref_dict=s3gen_refs[0] if s3gen_refs else {},
                            finalize=True,
                        )
                        
                        if not validate_tensor_operation(mel, "mel generation"):
                            raise ValueError("Invalid mel generated")
                        
                        if mel.size(-1) < 3:
                            mel = F.pad(mel, (0, 3 - mel.size(-1)), mode="replicate")

                        wav, _ = model.s3gen.hift_inference(
                            mel, torch.zeros(1, 1, 0, device=device)
                        )

                        if not validate_tensor_operation(wav, "wav generation"):
                            raise ValueError("Invalid wav generated")

                        audio = wav.squeeze().cpu().numpy()
                        
                        if audio.size == 0:
                            raise ValueError("Empty audio generated")
                        
                    except Exception as e:
                        print(f"Audio synthesis error for sample {i}: {e}")
                        raise ValueError("synthesis failed")

            # Compute WER
            try:
                if audio.size == 0:
                    wer = 1.0
                else:
                    fd, tmp = tempfile.mkstemp(suffix=f"_cmp_{i}.wav")
                    os.close(fd)
                    
                    audio_1d = audio.flatten()
                    audio_1d = np.clip(audio_1d, -1.0, 1.0)
                    
                    # Ensure audio is not too short
                    if len(audio_1d) < sr_gen // 10:  # At least 0.1 seconds
                        audio_1d = np.pad(audio_1d, (0, sr_gen // 10 - len(audio_1d)), mode='constant')
                    
                    sf.write(tmp, audio_1d, sr_gen)

                    try:
                        result = whisper_model(tmp, return_timestamps=True)
                        hyp = result["text"].strip() if result and "text" in result else ""
                        
                        ref_transcript = batch["transcript"][0] if batch["transcript"] else ""
                        
                        if hyp and ref_transcript:
                            wer = min(1.0, jiwer.wer(ref_transcript, hyp))
                        else:
                            wer = 1.0
                            
                    except Exception as e:
                        print(f"Whisper transcription error for sample {i}: {e}")
                        wer = 1.0
                    finally:
                        try:
                            os.remove(tmp)
                        except:
                            pass
                        
            except Exception as e:
                print(f"WER computation error for sample {i}: {e}")
                wer = 1.0

            wer_vals.append(float(wer))

            # Compute speaker similarity
            try:
                if audio.size <= S3_SR:
                    raise ValueError("clip <1 s, embedding invalid")

                audio_16k = librosa.resample(
                    y=audio.astype(np.float32),
                    orig_sr=sr_gen,
                    target_sr=S3_SR,
                )

                try:
                    gen_emb_raw = model.ve.embeds_from_wavs([audio_16k], sample_rate=S3_SR)
                    gen_emb_np = safe_flatten_to_numpy(gen_emb_raw)
                    
                except Exception as e:
                    print(f"Generated embedding error for sample {i}: {e}")
                    raise ValueError("generated embedding failed")
                
                min_len = min(len(gen_emb_np), len(ref_speaker_emb))
                if min_len == 0:
                    sim = 0.0
                else:
                    gen_emb_trimmed = gen_emb_np[:min_len]
                    ref_emb_trimmed = ref_speaker_emb[:min_len]
                    
                    if np.allclose(gen_emb_trimmed, 0) or np.allclose(ref_emb_trimmed, 0):
                        sim = 0.0
                    else:
                        try:
                            sim = 1.0 - cosine(gen_emb_trimmed, ref_emb_trimmed)
                            sim = float(max(0.0, min(1.0, sim)))
                        except Exception as e:
                            print(f"Cosine similarity error for sample {i}: {e}")
                            sim = 0.0
                
            except Exception as e:
                print(f"Speaker similarity error for sample {i}: {e}")
                sim = 0.0
                
            sim_vals.append(float(sim))

            # Compute length penalty
            try:
                # Extract scalar value from duration tensor
                if isinstance(batch["duration"], torch.Tensor):
                    # Get the first element and convert to Python float
                    duration_tensor = batch["duration"]
                    if duration_tensor.numel() > 0:
                        # Ensure we're working with a single value
                        if duration_tensor.dim() > 0:
                            tgt_sec = float(duration_tensor[0].item())
                        else:
                            tgt_sec = float(duration_tensor.item())
                    else:
                        tgt_sec = 1.0
                elif isinstance(batch["duration"], (list, tuple)):
                    tgt_sec = float(batch["duration"][0]) if len(batch["duration"]) > 0 else 1.0
                else:
                    tgt_sec = float(batch["duration"]) if batch["duration"] else 1.0
                
                gen_sec = float(audio.size / sr_gen)
                
                # Ensure tgt_sec is a positive scalar
                tgt_sec = max(0.1, float(tgt_sec))
                    
                r = gen_sec / tgt_sec
                
                # Use explicit float comparisons
                if r < 0.8:
                    lp = float((0.8 - r) ** 2)
                elif r > 1.2:
                    lp = float((r - 1.2) ** 2)
                else:
                    lp = 0.0
                    
            except Exception as e:
                print(f"Length penalty error for sample {i}: {e}")
                import traceback
                traceback.print_exc()
                lp = 1.0
                
            lp_vals.append(float(lp))

            reward = (
                -wer_weight * wer +
                speaker_sim_weight * sim -
                length_penalty_weight * lp
            )
            
            reward = max(-10.0, min(10.0, reward))
            rew.append(float(reward))

            del audio
            if DEVICE == 'cuda':
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"[compute_rewards] sample {i} -> {e}")
            rew.append(-5.0)
            wer_vals.append(1.0)
            sim_vals.append(0.0)
            lp_vals.append(1.0)

    if not rew:
        rew = [-5.0]
        wer_vals = [1.0]
        sim_vals = [0.0]
        lp_vals = [1.0]

    try:
        rewards_tensor = torch.tensor(rew, device=device, dtype=torch.float32)
    except Exception as e:
        print(f"Error creating rewards tensor: {e}")
        rewards_tensor = torch.tensor([-5.0] * len(rew), device=device, dtype=torch.float32)
    
    metrics = {
        "wer": float(np.mean(wer_vals)) if wer_vals else 1.0,
        "speaker_sim": float(np.mean(sim_vals)) if sim_vals else 0.0,
        "length_penalty": float(np.mean(lp_vals)) if lp_vals else 1.0,
    }
    
    return rewards_tensor, metrics

def compute_grpo_loss(
    model: ChatterboxTTS,
    samples: List[Tuple[torch.Tensor, torch.Tensor]],
    rewards: torch.Tensor,
    baseline_reward: float,
    t3_cond: T3Cond,
    kl_coeff: float = KL_COEFF,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute GRPO loss with improved bounds checking"""
    device = model.device
    batch_size = t3_cond.speaker_emb.size(0)  # Get actual batch size from t3_cond
    
    if not validate_tensor_operation(rewards, "rewards"):
        return torch.tensor(0.0, device=device, requires_grad=True), torch.tensor(0.0, device=device)
    
    advantages = rewards - baseline_reward
    ranked_indices = torch.argsort(advantages, descending=True)
    
    total_loss = 0.0
    total_kl = 0.0
    valid_samples = 0
    
    with torch.cuda.amp.autocast(enabled=(DEVICE == 'cuda'), dtype=torch.float16):
        for rank, idx in enumerate(ranked_indices):
            try:
                speech_tokens, text_tokens = samples[idx]
                
                if not validate_tensor_operation(speech_tokens, f"speech tokens {idx}"):
                    continue
                    
                if not validate_tensor_operation(text_tokens, f"text tokens {idx}"):
                    continue
                
                if speech_tokens.numel() == 0:
                    continue
                
                text_tokens = text_tokens.unsqueeze(0).to(device) if text_tokens.dim() == 1 else text_tokens.to(device)
                speech_tokens = speech_tokens.to(device)
                
                # Validate token values are within bounds
                vocab_size = getattr(model.t3, 'speech_vocab_size', 1024)
                speech_tokens = torch.clamp(speech_tokens, 0, vocab_size - 1)
                
                # Ensure text_tokens is repeated for the batch size
                if text_tokens.size(0) == 1 and batch_size > 1:
                    text_tokens = text_tokens.repeat(batch_size, 1)
                
                # Double for the two samples per input
                text_tokens_doubled = torch.cat([text_tokens, text_tokens], dim=0)
                speech_tokens_doubled = torch.cat([speech_tokens, speech_tokens], dim=0) if speech_tokens.dim() == 2 else torch.cat([speech_tokens.unsqueeze(0), speech_tokens.unsqueeze(0)], dim=0)
                
                # Create doubled conditionals
                t3_cond_doubled = T3Cond(
                    speaker_emb=torch.cat([t3_cond.speaker_emb, t3_cond.speaker_emb], dim=0),
                    cond_prompt_speech_tokens=torch.cat(
                        [t3_cond.cond_prompt_speech_tokens, t3_cond.cond_prompt_speech_tokens], dim=0
                    ) if t3_cond.cond_prompt_speech_tokens.numel() > 0 else torch.empty(batch_size * 2, 0, dtype=torch.long, device=device),
                    emotion_adv=torch.cat(
                        [t3_cond.emotion_adv, t3_cond.emotion_adv], dim=0
                    ) if t3_cond.emotion_adv is not None else None,
                )
                
                # Prepare input tokens (all but last for input)
                input_speech_tokens = speech_tokens_doubled[:, :-1] if speech_tokens_doubled.size(1) > 1 else torch.empty(batch_size * 2, 0, dtype=torch.long, device=device)
                
                embeds, len_cond = model.t3.prepare_input_embeds(
                    t3_cond=t3_cond_doubled,
                    text_tokens=text_tokens_doubled,
                    speech_tokens=input_speech_tokens,
                )
                
                if not validate_tensor_operation(embeds, f"embeds {idx}"):
                    continue
                
                if hasattr(model.t3.tfmr, 'gradient_checkpointing_enable'):
                    model.t3.tfmr.gradient_checkpointing_enable()
                
                hidden_states = model.t3.tfmr(inputs_embeds=embeds)[0]
                
                if not validate_tensor_operation(hidden_states, f"hidden states {idx}"):
                    continue
                
                # Calculate speech portion bounds
                speech_start = len_cond + text_tokens_doubled.size(1)
                speech_end = min(speech_start + speech_tokens_doubled.size(1) - 1, hidden_states.size(1))
                
                if speech_start < speech_end and speech_start >= 0 and speech_end <= hidden_states.size(1):
                    speech_hidden = safe_tensor_index(hidden_states, speech_start, speech_end, dim=1)
                    
                    if speech_hidden.numel() > 0 and validate_tensor_operation(speech_hidden, f"speech hidden {idx}"):
                        speech_logits = model.t3.speech_head(speech_hidden)
                        
                        if not validate_tensor_operation(speech_logits, f"speech logits {idx}"):
                            continue
                        
                        # Prepare target tokens (shifted by 1 for prediction)
                        target_end = min(speech_end - speech_start + 1, speech_tokens_doubled.size(1) - 1)
                        target_tokens = safe_tensor_index(speech_tokens_doubled, 1, 1 + target_end, dim=1)
                        
                        if target_tokens.numel() > 0 and speech_logits.size(1) >= target_tokens.size(1):
                            speech_logits = speech_logits[:, :target_tokens.size(1)]
                            
                            # Clamp logits to prevent overflow
                            speech_logits = torch.clamp(speech_logits, -10.0, 10.0)
                            
                            log_probs = F.log_softmax(speech_logits, dim=-1)
                            
                            if not validate_tensor_operation(log_probs, f"log probs {idx}"):
                                continue
                            
                            # Take only the first sample from the doubled batch
                            gathered_log_probs = safe_gather(
                                log_probs[0],
                                -1,
                                target_tokens[0].unsqueeze(-1)
                            ).squeeze(-1)
                            
                            if gathered_log_probs.numel() > 0 and validate_tensor_operation(gathered_log_probs, f"gathered log probs {idx}"):
                                rank_weight = 1.0 / (rank + 1)
                                sample_loss = -gathered_log_probs.mean() * rank_weight * advantages[idx]
                                
                                if validate_tensor_operation(sample_loss, f"sample loss {idx}"):
                                    total_loss += sample_loss
                                    valid_samples += 1
                                
                                # Compute KL divergence safely
                                probs = log_probs[0].exp()
                                if validate_tensor_operation(probs, f"probs for KL {idx}"):
                                    kl_div = (probs * log_probs[0]).sum(-1).mean()
                                    if validate_tensor_operation(kl_div, f"kl div {idx}"):
                                        total_kl += kl_div
                
                del embeds, hidden_states, speech_logits
                if DEVICE == 'cuda':
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"Error in GRPO loss computation for sample {idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    if valid_samples > 0:
        total_loss = total_loss / valid_samples + kl_coeff * total_kl / valid_samples
        total_kl = total_kl / valid_samples
    else:
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        total_kl = torch.tensor(0.0, device=device)
    
    return total_loss, total_kl


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering"""
    if logits.numel() == 0:
        return logits
    
    # Clamp logits to prevent overflow
    logits = torch.clamp(logits, -10.0, 10.0)
        
    top_k = min(top_k, logits.size(-1)) if top_k > 0 else 0
    if top_k > 0:
        # Get the top-k values
        values, _ = torch.topk(logits, top_k)
        if values.numel() > 0:
            min_value = values[..., -1, None]
            indices_to_remove = logits < min_value
            logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    
    return logits


def main():
    """Main GRPO training function"""
    print(f"Starting Chatterbox TTS GRPO fine-tuning")
    print(f"Device: {DEVICE}")
    
    # Enable CUDA debugging for better error messages
    if DEVICE == 'cuda':
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    metrics_tracker = GRPOMetricsTracker(save_path="grpo_training_metrics.png", update_interval=2.0)
    
    try:
        print("Loading Whisper model...")
        whisper_device = "cuda" if torch.cuda.is_available() else "cpu"
        whisper_model = pipeline("automatic-speech-recognition", model=WHISPER_MODEL, device=whisper_device)
        
        samples = load_audio_samples(AUDIO_DATA_DIR, whisper_model)
        if len(samples) == 0:
            raise ValueError(f"No valid audio samples found in {AUDIO_DATA_DIR}")
        
        random.shuffle(samples)
        val_size = int(len(samples) * VALIDATION_SPLIT)
        val_samples = samples[:val_size]
        train_samples = samples[val_size:]
        
        print(f"Train samples: {len(train_samples)}, Validation samples: {len(val_samples)}")
        
        print("Loading Chatterbox TTS model...")
        model = ChatterboxTTS.from_pretrained(DEVICE)
        
        # Replace tokenizer with Spanish version for Spanish training
        print("🔄 Switching to Spanish tokenizer...")
        spanish_tokenizer = SpanishTokenizer()
        spanish_tokenizer.check_vocabset_sot_eot()  # Verify special tokens
        
        # CRITICAL: Resize model embeddings to match Spanish vocabulary
        print(f"🔄 Resizing model embeddings for Spanish vocabulary...")
        model = resize_t3_embeddings(model, spanish_tokenizer.vocab_size, DEVICE)
        
        # Replace the tokenizer
        model.tokenizer = spanish_tokenizer
        print(f"✅ Spanish tokenizer installed with {spanish_tokenizer.vocab_size:,} tokens")
        
        if hasattr(model.t3.tfmr, 'gradient_checkpointing_enable'):
            model.t3.tfmr.gradient_checkpointing_enable()
            print("Enabled gradient checkpointing for transformer")
        
        print("Injecting LoRA layers...")
        target_modules = TARGET_MODULES
        
        lora_layers = inject_lora_layers(
            model.t3.tfmr,
            target_modules,
            rank=LORA_RANK,
            alpha=LORA_ALPHA,
            dropout=LORA_DROPOUT
        )
        print(f"Injected {len(lora_layers)} LoRA layers")
        
        train_dataset = TTSDataset(train_samples, model.tokenizer)
        val_dataset = TTSDataset(val_samples, model.tokenizer)
        
        num_workers = 0 if os.name == 'nt' else 2  # Reduced for stability
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True if DEVICE == 'cuda' else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True if DEVICE == 'cuda' else False
        )
        
        # ---------------------------------------------------------------
        # Optimizer trains LoRA params *and* new Spanish embedding rows
        # ---------------------------------------------------------------
        lora_params: List[torch.nn.Parameter] = []
        for layer in lora_layers.values():
            lora_params.extend([layer.lora_A, layer.lora_B])
        
        base_params: List[torch.nn.Parameter] = list(model.t3.text_emb.parameters()) + \
                                              list(model.t3.text_head.parameters())

        optimizer = AdamW(base_params + lora_params, lr=LEARNING_RATE)
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=len(train_loader) * EPOCHS,
            eta_min=LEARNING_RATE * 0.1
        )
        
        baseline_reward = 0.0
        
        print("Starting GRPO training...")
        global_step = 0
        scaler = torch.cuda.amp.GradScaler(enabled=True) if DEVICE == 'cuda' else None
        
        for epoch in range(EPOCHS):
            model.t3.train()
            model.ve.eval()
            model.s3gen.eval()
            train_loss = 0.0
            train_steps = 0
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
            for batch_idx, batch in enumerate(progress_bar):
                try:
                    if DEVICE == 'cuda':
                        torch.cuda.empty_cache()
                    
                    t3_cond, s3gen_refs = prepare_batch_conditionals(batch, model, model.ve, model.s3gen)
                    
                    samples = generate_samples(
                        model, batch, t3_cond, s3gen_refs,
                        num_samples=NUM_SAMPLES_PER_INPUT,
                        temperature=TEMPERATURE,
                        top_k=TOP_K,
                        top_p=TOP_P
                    )
                    
                    if not samples:
                        continue
                    
                    rewards, reward_metrics = compute_rewards(
                        model, samples, batch, t3_cond, s3gen_refs, whisper_model,
                        wer_weight=WER_WEIGHT,
                        speaker_sim_weight=SPEAKER_SIM_WEIGHT,
                        length_penalty_weight=LENGTH_PENALTY_WEIGHT,
                    )
                    
                    if not validate_tensor_operation(rewards, "rewards"):
                        continue
                    
                    avg_reward = rewards.mean().item()
                    baseline_reward = (
                        REWARD_BASELINE_MOMENTUM * baseline_reward +
                        (1 - REWARD_BASELINE_MOMENTUM) * avg_reward
                    )
                    
                    loss, kl_div = compute_grpo_loss(
                        model, samples, rewards, baseline_reward, t3_cond, kl_coeff=KL_COEFF
                    )
                    
                    if not validate_tensor_operation(loss, "loss"):
                        continue
                    
                    loss = loss / GRADIENT_ACCUMULATION_STEPS
                    
                    if scaler and DEVICE == 'cuda':
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    
                    if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                        grad_norm = 0.0
                        for p in lora_params:
                            if p.grad is not None:
                                grad_norm += p.grad.data.norm(2).item() ** 2
                        grad_norm = grad_norm ** 0.5
                        
                        # Clip gradients
                        if scaler and DEVICE == 'cuda':
                            scaler.unscale_(optimizer)
                        
                        torch.nn.utils.clip_grad_norm_(lora_params, max_norm=1.0)
                        
                        if scaler and DEVICE == 'cuda':
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            optimizer.step()
                        optimizer.zero_grad()
                        scheduler.step()
                        
                        global_step += 1
                        train_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
                        train_steps += 1
                        
                        current_lr = scheduler.get_last_lr()[0]
                        
                        metrics_tracker.add_metrics(
                            train_loss=train_loss / train_steps,
                            learning_rate=current_lr,
                            steps=global_step,
                            epochs=epoch,
                            batch_loss=loss.item() * GRADIENT_ACCUMULATION_STEPS,
                            gradient_norm=grad_norm,
                            avg_reward=avg_reward,
                            wer_score=reward_metrics['wer'],
                            speaker_sim=reward_metrics['speaker_sim'],
                            length_penalty=reward_metrics['length_penalty'],
                            kl_divergence=kl_div.item() if validate_tensor_operation(kl_div, "kl_div") else 0.0,
                            baseline_reward=baseline_reward,
                        )
                        
                        progress_bar.set_postfix({
                            'loss': f'{train_loss/train_steps:.4f}',
                            'reward': f'{avg_reward:.4f}',
                            'wer': f'{reward_metrics["wer"]:.3f}'
                        })
                        
                        if global_step % SAVE_EVERY_N_STEPS == 0:
                            save_checkpoint(model, lora_layers, optimizer, epoch, global_step, 
                                           train_loss/train_steps, CHECKPOINT_DIR)
                    
                    del samples, rewards, loss
                    if DEVICE == 'cuda':
                        torch.cuda.empty_cache()
                        
                except Exception as e:
                    print(f"Error in training batch {batch_idx}: {e}")
                    import traceback
                    traceback.print_exc()
                    if DEVICE == 'cuda':
                        torch.cuda.empty_cache()
                    optimizer.zero_grad()  # Clear any partial gradients
                    continue
            
            # Validation
            model.t3.eval()
            model.ve.eval()
            model.s3gen.eval()
            val_rewards = []
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validation"):
                    try:
                        if DEVICE == 'cuda':
                            torch.cuda.empty_cache()
                        
                        t3_cond, s3gen_refs = prepare_batch_conditionals(batch, model, model.ve, model.s3gen)
                        
                        samples = generate_samples(
                            model, batch, t3_cond, s3gen_refs,
                            num_samples=1,
                            temperature=1.0,
                            top_k=0,
                            top_p=0.0
                        )
                        
                        if samples:
                            rewards, _ = compute_rewards(
                                model, samples, batch, t3_cond, s3gen_refs, whisper_model,
                                wer_weight=WER_WEIGHT,
                                speaker_sim_weight=SPEAKER_SIM_WEIGHT,
                                length_penalty_weight=LENGTH_PENALTY_WEIGHT,
                            )
                            
                            if validate_tensor_operation(rewards, "validation rewards"):
                                val_rewards.append(rewards.mean().item())
                        
                        del samples
                        if DEVICE == 'cuda':
                            torch.cuda.empty_cache()
                            
                    except Exception as e:
                        print(f"Error in validation batch: {e}")
                        continue
            
            avg_val_reward = np.mean(val_rewards) if val_rewards else 0.0
            print(f"Epoch {epoch+1} - Train Loss: {train_loss/max(train_steps, 1):.4f}, Val Reward: {avg_val_reward:.4f}")
            
            save_checkpoint(model, lora_layers, optimizer, epoch, global_step, avg_val_reward, CHECKPOINT_DIR)
        
        print("Training completed!")
        
        metrics_tracker.stop()
        
        final_adapter_path = Path(CHECKPOINT_DIR) / "final_grpo_lora_adapter.pt"
        save_lora_adapter(lora_layers, str(final_adapter_path))
        
        print("Creating merged model...")
        # ---------------------------------------------------------------
        # Build merged model that already has Spanish vocab & weights
        # ---------------------------------------------------------------
        merged_model = ChatterboxTTS.from_pretrained(DEVICE)
        merged_model = resize_t3_embeddings(merged_model, spanish_tokenizer.vocab_size, DEVICE)
        merged_model.tokenizer = spanish_tokenizer

        # copy embedding/head weights
        merged_model.t3.text_emb.weight.data = model.t3.text_emb.weight.data.clone()
        merged_model.t3.text_head.weight.data = model.t3.text_head.weight.data.clone()
        if merged_model.t3.text_head.bias is not None:
            merged_model.t3.text_head.bias.data = model.t3.text_head.bias.data.clone()
        
        merged_lora_layers = inject_lora_layers(
            merged_model.t3.tfmr,
            target_modules,
            rank=LORA_RANK,
            alpha=LORA_ALPHA,
            dropout=LORA_DROPOUT
        )
        for name, layer in lora_layers.items():
            if name in merged_lora_layers:
                merged_lora_layers[name].lora_A.data = layer.lora_A.data.clone()
                merged_lora_layers[name].lora_B.data = layer.lora_B.data.clone()
        
        merged_model = merge_lora_weights(merged_model, merged_lora_layers)
        
        merged_dir = Path(CHECKPOINT_DIR) / "merged_grpo_model"
        merged_dir.mkdir(parents=True, exist_ok=True)
        
        torch.save(merged_model.ve.state_dict(), merged_dir / "ve.pt")
        torch.save(merged_model.t3.state_dict(), merged_dir / "t3_cfg.pt")
        torch.save(merged_model.s3gen.state_dict(), merged_dir / "s3gen.pt")
        
        # save spanish tokenizer
        spanish_tokenizer.save_pretrained(merged_dir, safe_serialization=False)
        
        print(f"Saved GRPO merged model to {merged_dir}")
        print("\nTraining complete!")
        
    except Exception as e:
        print(f"Fatal error in main: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        if 'metrics_tracker' in locals():
            metrics_tracker.stop()

def load_audio_samples(audio_dir: str, whisper_model) -> List[AudioSample]:
    """Load audio files and generate transcripts using Whisper"""
    samples = []
    audio_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
    
    cache_file = Path(audio_dir) / "transcripts_cache.json"
    transcript_cache = {}
    
    if cache_file.exists():
        print(f"Loading transcript cache from {cache_file}")
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                transcript_cache = json.load(f)
        except Exception as e:
            print(f"Error loading cache: {e}")
            transcript_cache = {}
    
    print(f"Loading audio files from {audio_dir}...")
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(Path(audio_dir).glob(f"*{ext}"))
    
    print(f"Found {len(audio_files)} audio files")
    
    cache_updated = False
    
    for audio_path in tqdm(audio_files, desc="Processing audio"):
        try:
            audio, sr = librosa.load(audio_path, sr=None)
            duration = len(audio) / sr
            
            if duration < MIN_AUDIO_LENGTH or duration > MAX_AUDIO_LENGTH:
                continue
            
            audio_path_str = str(audio_path.relative_to(Path(audio_dir)))
            
            if audio_path_str in transcript_cache:
                transcript = transcript_cache[audio_path_str]['transcript']
                print(f"Using cached transcript for {audio_path.name}")
            else:
                print(f"\nTranscribing {audio_path.name}...")
                try:
                    result = whisper_model(str(audio_path), return_timestamps=True)
                    transcript = result['text'].strip()
                    
                    transcript_cache[audio_path_str] = {
                        'transcript': transcript,
                        'duration': duration,
                        'sample_rate': sr
                    }
                    cache_updated = True
                except Exception as e:
                    print(f"Error transcribing {audio_path}: {e}")
                    continue
            
            if transcript:
                samples.append(AudioSample(
                    audio_path=audio_path,
                    transcript=transcript,
                    duration=duration,
                    sample_rate=sr
                ))
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            continue
    
    if cache_updated:
        print(f"Saving transcript cache to {cache_file}")
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(transcript_cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving cache: {e}")
    
    print(f"Successfully loaded {len(samples)} samples")
    return samples


def save_checkpoint(
    model: ChatterboxTTS,
    lora_layers: Dict[str, LoRALayer],
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    metric: float,
    checkpoint_dir: str,
    is_best: bool = False,
):
    """Save training checkpoint"""
    try:
        checkpoint_path = Path(checkpoint_dir) / f"checkpoint_epoch{epoch}_step{step}.pt"
        if is_best:
            checkpoint_path = Path(checkpoint_dir) / "best_model.pt"
        
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        lora_state_dict = {}
        for name, layer in lora_layers.items():
            try:
                lora_state_dict[f"{name}.lora_A"] = layer.lora_A.cpu()
                lora_state_dict[f"{name}.lora_B"] = layer.lora_B.cpu()
            except Exception as e:
                print(f"Error saving LoRA layer {name}: {e}")
        
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'metric': metric,
            'lora_state_dict': lora_state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            'model_state_dict': model.state_dict(),  # keep trained embeddings
        }
        
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")
        
        # Move tensors back to GPU
        for name, layer in lora_layers.items():
            layer.lora_A = layer.lora_A.to(model.device)
            layer.lora_B = layer.lora_B.to(model.device)
            
    except Exception as e:
        print(f"Error saving checkpoint: {e}")


def merge_lora_weights(model: ChatterboxTTS, lora_layers: Dict[str, LoRALayer]):
    """Merge LoRA weights into the base model"""
    with torch.no_grad():
        for name, lora_layer in lora_layers.items():
            try:
                parts = name.split('.')
                module = model.t3.tfmr
                for part in parts[:-1]:
                    module = getattr(module, part)
                linear_layer = getattr(module, parts[-1])
                
                lora_update = (lora_layer.lora_B @ lora_layer.lora_A) * lora_layer.scaling
                linear_layer.weight.data += lora_update
            except Exception as e:
                print(f"Error merging LoRA layer {name}: {e}")
    
    return model


def save_lora_adapter(lora_layers: Dict[str, LoRALayer], filepath: str):
    """Save LoRA adapter weights and configuration"""
    try:
        adapter_dict = {
            'lora_config': {
                'rank': LORA_RANK,
                'alpha': LORA_ALPHA,
                'dropout': LORA_DROPOUT,
                'target_modules': list(set(name.split('.')[-1] for name in lora_layers.keys())),
            },
            'lora_weights': {},
        }
        
        for name, layer in lora_layers.items():
            adapter_dict['lora_weights'][name] = {
                'lora_A': layer.lora_A.cpu(),
                'lora_B': layer.lora_B.cpu(),
            }
        
        torch.save(adapter_dict, filepath)
        print(f"Saved LoRA adapter to {filepath}")
    except Exception as e:
        print(f"Error saving LoRA adapter: {e}")


def collate_fn(samples):
    """Custom collate function for DataLoader"""
    try:
        # Filter out None samples
        samples = [s for s in samples if s is not None]
        if not samples:
            return None
            
        return {
            'audio': torch.stack([s['audio'] for s in samples]),
            'audio_16k': torch.stack([s['audio_16k'] for s in samples]),
            'text': [s['text'] for s in samples],
            'transcript': [s['transcript'] for s in samples],
            'audio_path': [s['audio_path'] for s in samples],
            'duration': torch.tensor([s['duration'] for s in samples]),
        }
    except Exception as e:
        print(f"Error in collate_fn: {e}")
        return None


if __name__ == "__main__":
    main()
