#!/bin/bash

echo "=================================================================="
echo "Targeted Model Cleanup - Free ~70GB"
echo "=================================================================="

# Models to delete (duplicates/old versions)
echo -e "\nModels that will be DELETED:\n"

echo "1. Meta-Llama-3.1-8B-Instruct (9.5G) - INCOMPLETE, only 2/4 shards"
du -sh ~/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct 2>/dev/null

echo "2. Meta-Llama-3-8B (15G) - Base model (not instruct)"
du -sh ~/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B 2>/dev/null

echo "3. Llama-2-7b-hf (13G) - Old version"
du -sh ~/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf 2>/dev/null

echo "4. Phi-3-small-128k-instruct (14G) - If not needed"
du -sh ~/.cache/huggingface/hub/models--microsoft--Phi-3-small-128k-instruct 2>/dev/null

echo "5. Old pip cache (8.2G)"
echo "8.2G	~/.cache/pip"

echo "6. Old torch cache (2.4G)"
echo "2.4G	~/.cache/torch"

echo "7. Old HuggingFace downloads (27G)"
echo "27G	~/.cache/huggingface (old files)"

echo -e "\n=================================================================="
echo "Models that will be KEPT:"
echo "=================================================================="
echo "✓ Llama-3.1-8B-Instruct (15G) - COMPLETE - THIS IS WHAT WE NEED"
du -sh ~/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct 2>/dev/null

echo "✓ Meta-Llama-3-8B-Instruct (15G) - If you need it"
du -sh ~/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct 2>/dev/null

echo -e "\n=================================================================="
echo "Estimated space to free: ~70GB"
echo "=================================================================="

echo -e "\nTo DELETE these files, run:\n"
echo "# Delete incomplete/duplicate Llama models:"
echo "rm -rf ~/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct"
echo "rm -rf ~/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B"
echo "rm -rf ~/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf"
echo ""
echo "# Delete Phi-3 if not needed:"
echo "rm -rf ~/.cache/huggingface/hub/models--microsoft--Phi-3-small-128k-instruct"
echo ""
echo "# Delete old cache files:"
echo "find ~/.cache/pip -type f -atime +90 -delete 2>/dev/null"
echo "find ~/.cache/torch -type f -atime +90 -delete 2>/dev/null"
echo "find ~/.cache/huggingface/hub/downloads -type f -mtime +90 -delete 2>/dev/null"
echo ""
echo "=================================================================="
echo "OR run this script with 'delete' argument to delete them:"
echo "./cleanup_models_targeted.sh delete"
echo "=================================================================="

if [ "$1" == "delete" ]; then
    echo -e "\n⚠️  DELETING FILES..."
    
    echo "Deleting Meta-Llama-3.1-8B-Instruct (incomplete)..."
    rm -rf ~/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct
    
    echo "Deleting Meta-Llama-3-8B (base model)..."
    rm -rf ~/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B
    
    echo "Deleting Llama-2-7b-hf (old)..."
    rm -rf ~/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf
    
    echo "Deleting Phi-3-small..."
    rm -rf ~/.cache/huggingface/hub/models--microsoft--Phi-3-small-128k-instruct
    
    echo "Deleting old pip cache..."
    find ~/.cache/pip -type f -atime +90 -delete 2>/dev/null
    
    echo "Deleting old torch cache..."
    find ~/.cache/torch -type f -atime +90 -delete 2>/dev/null
    
    echo -e "\n✅ DONE! Check space with: df -h /"
fi

