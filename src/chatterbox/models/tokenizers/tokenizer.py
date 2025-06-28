import logging

import torch
from tokenizers import Tokenizer
from transformers import AutoTokenizer
from pathlib import Path


# Special tokens
SOT = "[START]"
EOT = "[STOP]"
UNK = "[UNK]"
SPACE = "[SPACE]"
SPECIAL_TOKENS = [SOT, EOT, UNK, SPACE, "[PAD]", "[SEP]", "[CLS]", "[MASK]"]

logger = logging.getLogger(__name__)

class EnTokenizer:
    def __init__(self, vocab_file_path):
        self.tokenizer: Tokenizer = Tokenizer.from_file(vocab_file_path)
        self.check_vocabset_sot_eot()

        # Helpful debug print so we know what we really loaded
        try:
            _vs = len(self.tokenizer.get_vocab())
            print(f"✅ Loaded tokenizer from '{vocab_file_path}' containing {_vs:,} tokens")
        except Exception:
            pass

    def check_vocabset_sot_eot(self):
        voc = self.tokenizer.get_vocab()
        assert SOT in voc
        assert EOT in voc

    def text_to_tokens(self, text: str):
        text_tokens = self.encode(text)
        text_tokens = torch.IntTensor(text_tokens).unsqueeze(0)
        return text_tokens

    def encode( self, txt: str, verbose=False):
        """
        clean_text > (append `lang_id`) > replace SPACE > encode text using Tokenizer
        """
        txt = txt.replace(' ', SPACE)
        code = self.tokenizer.encode(txt)
        ids = code.ids
        return ids

    def decode(self, seq):
        if isinstance(seq, torch.Tensor):
            seq = seq.cpu().numpy()

        txt: str = self.tokenizer.decode(seq,
        skip_special_tokens=False)
        txt = txt.replace(' ', '')
        txt = txt.replace(SPACE, ' ')
        txt = txt.replace(EOT, '')
        txt = txt.replace(UNK, '')
        return txt

    # Allow loader utilities to query size uniformly
    @property
    def vocab_size(self):
        return len(self.tokenizer.get_vocab())

class SpanishTokenizer:
    """Clean Spanish tokenizer for Chatterbox TTS - matches EnTokenizer interface"""
    
    def __init__(self, model_name="PlanTL-GOB-ES/roberta-base-bne"):
        """Create a Spanish-compatible tokenizer.

        `model_name` can be either:
          • A HF repo id such as "PlanTL-GOB-ES/roberta-base-bne" (default).
          • A path to a *directory* already containing a tokenizer config.
          • A direct path to a `tokenizer.json` file (as used by Chatterbox
            checkpoints).  In that case we resolve the parent directory so
            `AutoTokenizer.from_pretrained` can load the config properly.
        """

        # If a file path ending with .json is supplied, use its parent folder
        possible_path = Path(str(model_name))
        if possible_path.is_file() and possible_path.suffix == ".json":
            resolved_path = str(possible_path.parent)
        else:
            resolved_path = model_name

        print(f"🔄 Loading Spanish tokenizer: {resolved_path}")

        # Load tokenizer (will work for local dir or HF repo id)
        self.base_tokenizer = AutoTokenizer.from_pretrained(resolved_path)
        
        # Add our special tokens to the vocabulary
        special_tokens = {
            'additional_special_tokens': [SOT, EOT, SPACE]
        }
        num_added = self.base_tokenizer.add_special_tokens(special_tokens)
        
        # Get final vocabulary size after adding tokens
        self.vocab_size = len(self.base_tokenizer)
        
        print(f"✅ Spanish tokenizer initialized:")
        print(f"   Base model: {resolved_path}")
        print(f"   Vocabulary size: {self.vocab_size:,} tokens")
        print(f"   Added special tokens: {num_added}")
        
        # Get special token IDs
        self.sot_id = self.base_tokenizer.convert_tokens_to_ids(SOT)
        self.eot_id = self.base_tokenizer.convert_tokens_to_ids(EOT)
        self.space_id = self.base_tokenizer.convert_tokens_to_ids(SPACE)
        self.unk_id = self.base_tokenizer.unk_token_id
        
        print(f"   Special token IDs: SOT={self.sot_id}, EOT={self.eot_id}, SPACE={self.space_id}")
        
    def check_vocabset_sot_eot(self):
        """Verify special tokens exist (compatibility with EnTokenizer)"""
        assert self.sot_id != self.unk_id, f"SOT token '{SOT}' not found in vocabulary"
        assert self.eot_id != self.unk_id, f"EOT token '{EOT}' not found in vocabulary"
        print("✅ Special tokens verified successfully")
        
    def text_to_tokens(self, text: str):
        """Convert text to tensor (main interface method)"""
        text_tokens = self.encode(text)
        text_tokens = torch.IntTensor(text_tokens).unsqueeze(0)
        return text_tokens
        
    def encode(self, txt: str, verbose=False):
        """Encode Spanish text to token IDs"""
        if verbose:
            print(f"Encoding: '{txt}'")
            
        # Replace spaces with our special SPACE token
        txt = txt.replace(' ', SPACE)
        
        # Encode using Spanish tokenizer (no special tokens added)
        token_ids = self.base_tokenizer.encode(txt, add_special_tokens=False)
        
        if verbose:
            decoded_check = self.base_tokenizer.decode(token_ids)
            print(f"Tokens: {token_ids}")
            print(f"Decoded check: '{decoded_check}'")
            
        return token_ids
        
    def decode(self, seq):
        """Decode token IDs back to Spanish text"""
        if isinstance(seq, torch.Tensor):
            seq = seq.cpu().numpy()
            
        # Decode using Spanish tokenizer
        txt = self.base_tokenizer.decode(seq, skip_special_tokens=False)
        
        # Post-process to match EnTokenizer behavior
        txt = txt.replace(' ', '')          # Remove tokenizer-added spaces
        txt = txt.replace(SPACE, ' ')       # Restore our SPACE tokens as spaces
        txt = txt.replace(EOT, '')          # Remove end-of-text tokens
        txt = txt.replace(UNK, '')          # Remove unknown tokens
        
        return txt.strip()

    def analyze_training_coverage(self, training_texts):
        """Analyze how well the tokenizer covers your training data"""
        total_tokens = 0
        unk_tokens = 0
        
        for text in training_texts:
            tokens = self.encode(text)
            total_tokens += len(tokens)
            unk_tokens += tokens.count(self.unk_id)
        
        coverage = (total_tokens - unk_tokens) / total_tokens * 100
        
        print(f"Training Data Coverage Analysis:")
        print(f"  Total tokens: {total_tokens:,}")
        print(f"  Unknown tokens: {unk_tokens:,}")
        print(f"  Coverage: {coverage:.2f}%")
        
        return coverage

    def test_inference_words(self, test_words):
        """Test how the tokenizer handles common Spanish words for inference"""
        print("Testing inference vocabulary coverage:")
        
        for word in test_words:
            tokens = self.encode(word)
            decoded = self.decode(tokens)
            
            has_unk = self.unk_id in tokens
            status = "❌ UNK" if has_unk else "✅ OK"
            
            print(f"  '{word}' → {tokens} → '{decoded}' {status}")

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def save_pretrained(self, save_directory: str, **kwargs):
        """Persist the underlying tokenizer to *save_directory*.

        This mirrors the HuggingFace `PreTrainedTokenizer` API so that
        training scripts can simply call `spanish_tokenizer.save_pretrained(...)`.
        Any extra keyword arguments are forwarded verbatim.
        """
        Path(save_directory).mkdir(parents=True, exist_ok=True)
        return self.base_tokenizer.save_pretrained(save_directory, **kwargs)
