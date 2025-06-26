from ..llama_configs import LLAMA_CONFIGS


class T3Config:
    start_text_token = 255
    stop_text_token = 0
    text_tokens_dict_size = 704
    max_text_tokens = 2048

    start_speech_token = 6561
    stop_speech_token = 6562
    speech_tokens_dict_size = 8194
    max_speech_tokens = 4096

    llama_config_name = "Llama_520M"
    input_pos_emb = "learned"
    speech_cond_prompt_len = 150

    # For T3CondEnc
    encoder_type = "voice_encoder"
    speaker_embed_size = 256
    use_perceiver_resampler = True
    emotion_adv = True

    @property
    def n_channels(self):
        return LLAMA_CONFIGS[self.llama_config_name]["hidden_size"]

    def estimate_memory_impact(self):
        """Estimate additional memory usage from vocabulary size"""
        hidden_size = self.n_channels
        text_embed_params = self.text_tokens_dict_size * hidden_size
        text_head_params = self.text_tokens_dict_size * hidden_size
        total_params = text_embed_params + text_head_params
        
        # 4 bytes per parameter (float32)
        memory_mb = total_params * 4 / 1e6
        
        print(f"T3 Text Vocabulary Memory Impact:")
        print(f"  Vocabulary size: {self.text_tokens_dict_size:,}")
        print(f"  Text embedding params: {text_embed_params:,}")
        print(f"  Text head params: {text_head_params:,}")
        print(f"  Total additional memory: ~{memory_mb:.1f} MB")
        
        return memory_mb


class T3SpanishConfig(T3Config):
    """Spanish-optimized T3 configuration"""
    
    def __init__(self, spanish_vocab_size=50265):  # Default RoBERTa size
        """
        Args:
            spanish_vocab_size: Size of Spanish tokenizer vocabulary
        """
        # Copy all settings from base config
        self.start_text_token = 255
        self.stop_text_token = 0
        self.text_tokens_dict_size = spanish_vocab_size  # Updated for Spanish
        self.max_text_tokens = 2048

        self.start_speech_token = 6561
        self.stop_speech_token = 6562
        self.speech_tokens_dict_size = 8194
        self.max_speech_tokens = 4096

        self.llama_config_name = "Llama_520M"
        self.input_pos_emb = "learned"
        self.speech_cond_prompt_len = 150

        # For T3CondEnc
        self.encoder_type = "voice_encoder"
        self.speaker_embed_size = 256
        self.use_perceiver_resampler = True
        self.emotion_adv = True
        
        print(f"✅ T3SpanishConfig created:")
        print(f"   Text vocabulary size: {self.text_tokens_dict_size:,}")
        print(f"   Speech vocabulary size: {self.speech_tokens_dict_size:,}")

    @property
    def n_channels(self):
        return LLAMA_CONFIGS[self.llama_config_name]["hidden_size"]

    def estimate_memory_impact(self):
        """Estimate additional memory usage from vocabulary size"""
        hidden_size = self.n_channels
        
        # Memory for text embeddings (input + output layers)
        text_embed_memory = 2 * self.text_tokens_dict_size * hidden_size * 4 / 1e6  # 4 bytes per float32
        
        # Memory for speech embeddings
        speech_embed_memory = 2 * self.speech_tokens_dict_size * hidden_size * 4 / 1e6
        
        # Compare with original English config
        original_text_memory = 2 * 704 * hidden_size * 4 / 1e6
        additional_memory = text_embed_memory - original_text_memory
        
        total_memory = text_embed_memory + speech_embed_memory
        
        print(f"Memory impact analysis:")
        print(f"   Original text embeddings: {original_text_memory:.1f} MB")
        print(f"   Spanish text embeddings: {text_embed_memory:.1f} MB")
        print(f"   Additional memory needed: {additional_memory:.1f} MB")
        print(f"   Total embeddings: {total_memory:.1f} MB")
        
        return total_memory


def resize_t3_embeddings(model, new_vocab_size, device='cuda'):
    """
    Resize T3 model embeddings to accommodate Spanish vocabulary
    
    Args:
        model: ChatterboxTTS model
        new_vocab_size: New vocabulary size (Spanish tokenizer size)
        device: Device to place tensors on
    """
    print(f"🔄 Resizing T3 model embeddings...")
    
    # Access the T3 model's embedding layers directly
    t3_model = model.t3  # T3 model
    old_text_embeddings = t3_model.text_emb
    old_text_head = t3_model.text_head
    
    old_vocab_size = old_text_embeddings.num_embeddings
    embedding_dim = old_text_embeddings.embedding_dim
    
    print(f"   Old text vocabulary size: {old_vocab_size:,}")
    print(f"   New text vocabulary size: {new_vocab_size:,}")
    print(f"   Embedding dimension: {embedding_dim}")
    
    # Create new text embedding layer with larger vocabulary
    import torch
    new_text_embeddings = torch.nn.Embedding(
        new_vocab_size, 
        embedding_dim,
        padding_idx=old_text_embeddings.padding_idx
    ).to(device)
    
    # Create new text head layer
    new_text_head = torch.nn.Linear(
        old_text_head.in_features,
        new_vocab_size,
        bias=old_text_head.bias is not None
    ).to(device)
    
    # Copy existing embeddings for tokens that overlap
    min_vocab_size = min(old_vocab_size, new_vocab_size)
    with torch.no_grad():
        # Copy old text embeddings
        new_text_embeddings.weight[:min_vocab_size] = old_text_embeddings.weight[:min_vocab_size]
        
        # Copy old text head weights
        new_text_head.weight[:min_vocab_size] = old_text_head.weight[:min_vocab_size]
        if new_text_head.bias is not None:
            new_text_head.bias[:min_vocab_size] = old_text_head.bias[:min_vocab_size]
        
        # Initialize new token embeddings and head weights with small random values
        if new_vocab_size > old_vocab_size:
            new_text_embeddings.weight[old_vocab_size:].normal_(mean=0.0, std=0.02)
            new_text_head.weight[old_vocab_size:].normal_(mean=0.0, std=0.02)
            if new_text_head.bias is not None:
                new_text_head.bias[old_vocab_size:].zero_()
            print(f"   Initialized {new_vocab_size - old_vocab_size:,} new token embeddings and head weights")
    
    # Replace the embedding and head layers
    t3_model.text_emb = new_text_embeddings
    t3_model.text_head = new_text_head
    
    # Update the T3 config to reflect the new vocabulary size
    t3_model.hp.text_tokens_dict_size = new_vocab_size
    
    print("✅ T3 text embeddings and head resized successfully")
    print(f"   Text embedding: {old_vocab_size:,} → {new_vocab_size:,} tokens")
    print(f"   Text head: {old_vocab_size:,} → {new_vocab_size:,} outputs")
    
    return model


def punc_norm_spanish(text: str) -> str:
    """Spanish-specific text normalization"""
    if len(text) == 0:
        return "Necesitas agregar texto para que pueda hablar."
    
    # Spanish-specific replacements
    punc_to_replace = [
        ("...", ", "),
        ("…", ", "),
        (":", ","),
        (" - ", ", "),
        (";", ", "),
        ("—", "-"),
        ("–", "-"),
        (" ,", ","),
        ("¿", ""),  # Spanish question marks
        ("¡", ""),  # Spanish exclamation marks
    ]
    
    for old, new in punc_to_replace:
        text = text.replace(old, new)
    
    # Add period if no ending punctuation
    sentence_enders = {".", "!", "?", "-", ","}
    if not any(text.endswith(p) for p in sentence_enders):
        text += "."
    
    return text
