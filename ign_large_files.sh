#!/bin/bash

# Define the file size threshold in MB
MAX_SIZE_MB=100
# Convert the threshold to kilobytes (for `find`)
MAX_SIZE_KB=$((MAX_SIZE_MB * 1024))

# Find files larger than the threshold size
echo "Searching for files larger than $MAX_SIZE_MB MB..."

# Search for files, and exclude those already in .gitignore
LARGE_FILES=$(find . -type f -size +"${MAX_SIZE_KB}"k ! -path "./.git/*")

# Check if we found any large files
if [[ -z "$LARGE_FILES" ]]; then
    echo "No files found larger than $MAX_SIZE_MB MB."
    exit 0
fi

# Add each large file to .gitignore
for file in $LARGE_FILES; do
    # Check if the file is already in .gitignore
    if ! grep -qx "$file" .gitignore; then
        echo "Adding $file to .gitignore"
        echo "$file" >> .gitignore
    else
        echo "$file is already in .gitignore"
    fi
done

echo "Finished adding large files to .gitignore."

