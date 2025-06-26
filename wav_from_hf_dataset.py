import os
import soundfile as sf
import numpy as np
from datasets import load_dataset
from pathlib import Path
from tqdm import tqdm

def extract_audio_from_dataset(dataset, output_dir="./extracted_audio"):
    """
    Extract audio files from a Hugging Face dataset and save them as WAV files.
    
    Args:
        dataset: The Hugging Face dataset
        output_dir: Directory to save the audio files
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Extracting audio files to: {output_path}")
    print(f"Dataset has {len(dataset)} samples")
    
    successful = 0
    failed = 0
    
    for idx, sample in tqdm(enumerate(dataset), total=len(dataset), desc="Extracting audio"):
        try:
            # Get audio data from the sample
            audio_data = sample['audio']
            
            # Extract the audio array and sampling rate
            if isinstance(audio_data, dict):
                # If audio is stored as a dictionary with 'array' and 'sampling_rate'
                audio_array = audio_data['array']
                sampling_rate = audio_data['sampling_rate']
            else:
                # If audio is stored directly as an array
                audio_array = audio_data
                sampling_rate = 48000  # Default sampling rate, adjust if needed
            
            # Convert to numpy array if it isn't already
            if not isinstance(audio_array, np.ndarray):
                audio_array = np.array(audio_array)
            
            # Ensure audio is 1D
            if audio_array.ndim > 1:
                audio_array = audio_array.flatten()
            
            # Normalize audio if needed (ensure values are between -1 and 1)
            if audio_array.max() > 1.0 or audio_array.min() < -1.0:
                max_val = max(abs(audio_array.max()), abs(audio_array.min()))
                if max_val > 0:
                    audio_array = audio_array / max_val
            
            # Generate filename
            # Use text if available, otherwise use index
            if 'text' in sample:
                text = str(sample['text'])
                # Create a safe filename from text
                safe_text = "".join(c for c in text if c.isalnum() or c in (' ', '-', '_')).rstrip()
                safe_text = safe_text[:50]  # Limit length
                filename = f"{idx:06d}_{safe_text}.wav"
            else:
                filename = f"audio_{idx:06d}.wav"
            
            # Save as WAV file
            output_file = output_path / filename
            sf.write(str(output_file), audio_array, sampling_rate)
            
            successful += 1
            
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            failed += 1
            continue
    
    print(f"\nExtraction complete!")
    print(f"Successfully extracted: {successful} files")
    if failed > 0:
        print(f"Failed to extract: {failed} files")
    print(f"Output directory: {output_path}")

# Load your dataset
ds = load_dataset("ylacombe/google-argentinian-spanish", "female")

# Extract audio from the train split
print("Dataset structure:")
print(ds)

# Extract audio files
extract_audio_from_dataset(ds['train'], output_dir="./audio_data")