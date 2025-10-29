#!/bin/bash

# Clean up old model caches and files not accessed in last 3 months (90 days)
# This script finds and optionally deletes old models from various cache directories

echo "=================================================================="
echo "Finding models/caches not accessed in last 90 days"
echo "=================================================================="

# Calculate date 90 days ago
DAYS_OLD=90

# Common cache locations for models
CACHE_DIRS=(
    "$HOME/.cache/huggingface"
    "$HOME/.cache/torch"
    "$HOME/.cache/pip"
    "$HOME/.torch"
)

total_size=0

echo -e "\nScanning cache directories for old files (not accessed in ${DAYS_OLD}+ days)...\n"

for cache_dir in "${CACHE_DIRS[@]}"; do
    if [ -d "$cache_dir" ]; then
        echo "Checking: $cache_dir"
        
        # Find directories older than 90 days and calculate size
        while IFS= read -r -d '' dir; do
            size=$(du -sh "$dir" 2>/dev/null | cut -f1)
            last_access=$(stat -c %x "$dir" 2>/dev/null | cut -d' ' -f1)
            echo "  [OLD] $size - Last accessed: $last_access"
            echo "        $dir"
        done < <(find "$cache_dir" -maxdepth 2 -type d -atime +$DAYS_OLD -print0 2>/dev/null)
        
        echo ""
    fi
done

echo "=================================================================="
echo "SPECIFIC MODEL DIRECTORIES (HuggingFace)"
echo "=================================================================="

if [ -d "$HOME/.cache/huggingface/hub" ]; then
    echo -e "\nHuggingFace models and their last access time:\n"
    
    for model_dir in "$HOME/.cache/huggingface/hub"/models--*; do
        if [ -d "$model_dir" ]; then
            model_name=$(basename "$model_dir" | sed 's/models--//; s/--/\//g')
            size=$(du -sh "$model_dir" 2>/dev/null | cut -f1)
            last_access=$(stat -c %x "$model_dir" 2>/dev/null | cut -d' ' -f1)
            days_old=$(( ($(date +%s) - $(stat -c %X "$model_dir" 2>/dev/null)) / 86400 ))
            
            if [ $days_old -gt $DAYS_OLD ]; then
                echo "  [DELETE] $size - $model_name (${days_old} days old)"
                echo "           Last accessed: $last_access"
                echo "           Path: $model_dir"
                echo ""
            else
                echo "  [KEEP]   $size - $model_name (${days_old} days old, still recent)"
            fi
        fi
    done
fi

echo "=================================================================="
echo "SUMMARY"
echo "=================================================================="

# Calculate total size that can be freed
total_old_size=0
echo -e "\nCalculating total space that can be freed...\n"

for cache_dir in "${CACHE_DIRS[@]}"; do
    if [ -d "$cache_dir" ]; then
        old_size=$(find "$cache_dir" -type f -atime +$DAYS_OLD -exec du -ch {} + 2>/dev/null | grep total$ | cut -f1)
        if [ ! -z "$old_size" ]; then
            echo "  $cache_dir: $old_size"
        fi
    fi
done

echo ""
echo "=================================================================="
echo "TO DELETE OLD FILES, RUN:"
echo "=================================================================="
echo ""
echo "# Delete old HuggingFace models (90+ days):"
echo "find ~/.cache/huggingface -type d -atime +90 -maxdepth 2 -exec rm -rf {} + 2>/dev/null"
echo ""
echo "# Delete old pip cache:"
echo "find ~/.cache/pip -type f -atime +90 -delete 2>/dev/null"
echo ""
echo "# Delete old torch cache:"
echo "find ~/.cache/torch -type f -atime +90 -delete 2>/dev/null"
echo ""
echo "# OR delete specific models by name:"
echo "# rm -rf ~/.cache/huggingface/hub/models--<model-name>"
echo ""
echo "=================================================================="

