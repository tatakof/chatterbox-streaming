## Data preparation

To get the .wav files for training, run the following command:

```bash
python wav_from_hf_dataset.py
```


This will download the dataset from Hugging Face, extract the audio data, and save the .wav files to the `audio_data` directory.



## Training results
### LoRA Training (`lora.py`)

**Checkpoint Directory:** `checkpoints_lora/`

**Saved Files:**
- **Checkpoints:** `checkpoint_epoch{X}_step{Y}.pt`
- **Final LoRA Adapter:** `final_lora_adapter.pt`
- **Merged Model Directory:** `merged_model/`
  - `ve.pt` (voice encoder)
  - `t3_cfg.pt` (T3 model)
  - `s3gen.pt` (S3Gen model)
  - `tokenizer.json`
  - `conds.pt` (if exists)

### GRPO Training (`grpo.py`)

**Checkpoint Directory:** `checkpoints_grpo/`

**Saved Files:**
- **Checkpoints:** `checkpoint_epoch{X}_step{Y}.pt`
- **Final LoRA Adapter:** `final_grpo_lora_adapter.pt`
- **Merged Model Directory:** `merged_grpo_model/`
  - `ve.pt` (voice encoder)
  - `t3_cfg.pt` (T3 model)  
  - `s3gen.pt` (S3Gen model)
  - `tokenizer.json`

### Summary of Differences

| Component | LoRA Training | GRPO Training |
|-----------|---------------|---------------|
| **Base Directory** | `checkpoints_lora/` | `checkpoints_grpo/` |
| **Final Adapter** | `final_lora_adapter.pt` | `final_grpo_lora_adapter.pt` |
| **Merged Model Dir** | `merged_model/` | `merged_grpo_model/` |
| **Metrics Plot** | `training_metrics.png` | `grpo_training_metrics.png` |

### Loading the Models Later

After training, you can load either model:

```python
# Load LoRA-trained model
lora_model = ChatterboxTTS.from_local("./checkpoints_lora/merged_model", device)

# Load GRPO-trained model  
grpo_model = ChatterboxTTS.from_local("./checkpoints_grpo/merged_grpo_model", device)
```

### Comparing Results
1. **Final adapters** for analysis of learned weights
2. **Merged models** for inference quality comparison
3. **Training metrics** from the different PNG files generated
4. **Checkpoint progression** from each training approach
