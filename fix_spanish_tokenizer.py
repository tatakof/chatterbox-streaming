#!/usr/bin/env python3
"""
Fix Spanish Tokenizer Script
Replaces the English tokenizer with Spanish tokenizer in merged model directory
"""

import sys
sys.path.append('src')

from transformers import AutoTokenizer
from pathlib import Path
import shutil
import json

def fix_spanish_tokenizer():
    """Replace English tokenizer with Spanish tokenizer in merged model"""
    
    # Paths
    merged_model_dir = Path("./checkpoints_lora/merged_model")
    
    if not merged_model_dir.exists():
        print("❌ Merged model directory not found!")
        print("   Expected location: ./checkpoints_lora/merged_model")
        return False
    
    print("🔄 Replacing English tokenizer with Spanish tokenizer...")
    print(f"   Target directory: {merged_model_dir}")
    
    try:
        # Load Spanish tokenizer
        print("📥 Loading Spanish tokenizer (PlanTL-GOB-ES/roberta-base-bne)...")
        spanish_tokenizer = AutoTokenizer.from_pretrained("PlanTL-GOB-ES/roberta-base-bne")
        print(f"✅ Spanish tokenizer loaded with {spanish_tokenizer.vocab_size:,} tokens")
        
        # Backup original tokenizer
        original_tokenizer = merged_model_dir / "tokenizer.json"
        if original_tokenizer.exists():
            backup_path = merged_model_dir / "tokenizer_english_backup.json"
            shutil.copy(original_tokenizer, backup_path)
            print(f"📁 Backed up English tokenizer to {backup_path}")
            
            # Check original size
            with open(original_tokenizer, 'r') as f:
                original_data = json.load(f)
            original_size = len(original_data['model']['vocab'])
            print(f"   Original tokenizer had {original_size:,} tokens")
        
        # Save Spanish tokenizer (this creates tokenizer.json and other files)
        print("💾 Saving Spanish tokenizer...")
        spanish_tokenizer.save_pretrained(merged_model_dir, safe_serialization=False)
        
        # Verify the file was created and has correct size
        new_tokenizer_path = merged_model_dir / "tokenizer.json"
        if new_tokenizer_path.exists():
            with open(new_tokenizer_path, 'r') as f:
                tokenizer_data = json.load(f)
            vocab_size = len(tokenizer_data['model']['vocab'])
            
            print(f"✅ Spanish tokenizer saved successfully!")
            print(f"   New vocabulary size: {vocab_size:,} tokens")
            print(f"   Location: {new_tokenizer_path}")
            
            if vocab_size > 40000:
                print("🎉 SUCCESS! Spanish tokenizer installed correctly!")
                print("\n📋 Next steps:")
                print("   1. Test your Spanish TTS with: python tts_spanish.py")
                print("   2. The model should now handle Spanish text properly")
                return True
            else:
                print("⚠️ WARNING: Tokenizer seems too small for Spanish")
                return False
        else:
            print("❌ ERROR: tokenizer.json was not created!")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: Failed to fix Spanish tokenizer: {e}")
        return False

def verify_fix():
    """Verify that the fix worked by checking tokenizer size"""
    tokenizer_path = Path("./checkpoints_lora/merged_model/tokenizer.json")
    
    if not tokenizer_path.exists():
        print("❌ No tokenizer.json found!")
        return False
    
    try:
        with open(tokenizer_path, 'r') as f:
            data = json.load(f)
        
        vocab_size = len(data['model']['vocab'])
        print(f"\n🔍 Verification:")
        print(f"   Tokenizer vocabulary size: {vocab_size:,} tokens")
        
        if vocab_size > 40000:
            print("✅ Spanish tokenizer is correctly installed!")
            return True
        else:
            print("❌ Still using English tokenizer (too small)")
            return False
            
    except Exception as e:
        print(f"❌ Error verifying tokenizer: {e}")
        return False

def main():
    """Main function"""
    print("=" * 60)
    print("SPANISH TOKENIZER FIX SCRIPT")
    print("=" * 60)
    
    # Fix the tokenizer
    success = fix_spanish_tokenizer()
    
    if success:
        # Verify the fix
        verify_fix()
        
        print("\n" + "=" * 60)
        print("🎯 SPANISH TOKENIZER FIX COMPLETED!")
        print("=" * 60)
        print("\nYour merged model now uses Spanish tokenizer with 50k+ tokens")
        print("This should resolve the CUDA indexing errors and improve Spanish TTS quality.")
        
    else:
        print("\n" + "=" * 60)
        print("❌ SPANISH TOKENIZER FIX FAILED!")
        print("=" * 60)
        print("Please check the error messages above and try again.")

if __name__ == "__main__":
    main() 