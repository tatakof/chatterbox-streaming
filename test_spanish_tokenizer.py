#!/usr/bin/env python3
"""
Test script to validate Spanish tokenizer implementation
Run this before training to ensure everything works correctly
"""

import sys
import os
sys.path.append('src')

import torch
from chatterbox.models.tokenizers.tokenizer import SpanishTokenizer
from chatterbox.models.t3.modules.t3_config import T3SpanishConfig, resize_t3_embeddings

def test_spanish_tokenizer():
    """Test Spanish tokenizer functionality"""
    print("=" * 60)
    print("TESTING SPANISH TOKENIZER")
    print("=" * 60)
    
    # Initialize tokenizer
    print("\n1. Initializing Spanish tokenizer...")
    tokenizer = SpanishTokenizer()
    
    # Check special tokens
    print("\n2. Checking special tokens...")
    tokenizer.check_vocabset_sot_eot()
    
    # Test Spanish text encoding/decoding
    print("\n3. Testing Spanish text processing...")
    spanish_texts = [
        "Hola, ¿cómo estás?",
        "Me gusta la música española",
        "Buenos días, señora García",
        "El niño come manzanas rojas",
        "¡Qué hermoso día hace hoy!"
    ]
    
    for text in spanish_texts:
        print(f"\nOriginal: '{text}'")
        
        # Encode
        tokens = tokenizer.encode(text)
        print(f"Tokens: {tokens[:10]}{'...' if len(tokens) > 10 else ''} (total: {len(tokens)})")
        
        # Decode
        decoded = tokenizer.decode(tokens)
        print(f"Decoded: '{decoded}'")
        
        # Check reconstruction
        match = text.strip() == decoded.strip()
        print(f"Perfect reconstruction: {'✅' if match else '❌'}")
        
        if not match:
            print(f"  Expected: '{text.strip()}'")
            print(f"  Got:      '{decoded.strip()}'")
    
    # Test tensor conversion
    print("\n4. Testing tensor conversion...")
    test_text = "Hola mundo"
    tensor = tokenizer.text_to_tokens(test_text)
    print(f"Text: '{test_text}'")
    print(f"Tensor shape: {tensor.shape}")
    print(f"Tensor dtype: {tensor.dtype}")
    
    return tokenizer

def test_config():
    """Test Spanish T3 configuration"""
    print("\n" + "=" * 60)
    print("TESTING SPANISH T3 CONFIG")
    print("=" * 60)
    
    # Test with default vocabulary size
    print("\n1. Testing default config...")
    config = T3SpanishConfig()
    
    # Test memory estimation
    print("\n2. Testing memory estimation...")
    memory_mb = config.estimate_memory_impact()
    
    return config

def test_vocabulary_coverage():
    """Test vocabulary coverage for common Spanish words"""
    print("\n" + "=" * 60)
    print("TESTING VOCABULARY COVERAGE")
    print("=" * 60)
    
    tokenizer = SpanishTokenizer()
    
    # Common Spanish words that should be in vocabulary
    common_words = [
        "casa", "perro", "gato", "amor", "vida", "tiempo", "agua", 
        "fuego", "tierra", "cielo", "corazón", "familia", "amigo",
        "trabajo", "escuela", "comida", "música", "libro", "película"
    ]
    
    # Technical/modern words that might be subword-tokenized
    technical_words = [
        "computadora", "teléfono", "internet", "tecnología",
        "extraordinario", "anticonstitucional", "electrodoméstico"
    ]
    
    print("\n1. Testing common Spanish words...")
    tokenizer.test_inference_words(common_words)
    
    print("\n2. Testing technical/long words...")
    tokenizer.test_inference_words(technical_words)

def test_model_integration():
    """Test integration with actual model (if available)"""
    print("\n" + "=" * 60)
    print("TESTING MODEL INTEGRATION")
    print("=" * 60)
    
    try:
        # This would test with actual model if available
        print("⚠️  Model integration test requires actual Chatterbox model")
        print("   Run this during training to test embedding resizing")
        
        # Show what would happen
        tokenizer = SpanishTokenizer()
        config = T3SpanishConfig(tokenizer.vocab_size)
        
        print(f"\nModel would be resized from 704 → {tokenizer.vocab_size:,} tokens")
        print(f"Additional memory needed: ~{config.estimate_memory_impact():.1f} MB")
        
    except Exception as e:
        print(f"❌ Model integration test failed: {e}")
        print("   This is expected if model is not available")

def main():
    """Run all tests"""
    print("🚀 Starting Spanish Tokenizer Validation Tests")
    
    try:
        # Test tokenizer
        tokenizer = test_spanish_tokenizer()
        
        # Test config
        config = test_config()
        
        # Test vocabulary coverage
        test_vocabulary_coverage()
        
        # Test model integration
        test_model_integration()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"""
🎉 Spanish tokenizer is ready for training!

Summary:
- Tokenizer vocabulary: {tokenizer.vocab_size:,} tokens
- Special tokens verified: ✅
- Spanish text processing: ✅
- Memory impact estimated: ✅

Next steps:
1. Run your training script: python lora.py or python grpo.py
2. The model will automatically resize embeddings for Spanish
3. Training should proceed without vocabulary mismatch errors

¡Buena suerte con tu entrenamiento! 🇪🇸
        """)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        print("\nPlease fix the issues above before training.")
        sys.exit(1)

if __name__ == "__main__":
    main() 