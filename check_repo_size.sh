#!/bin/bash
echo "Repository Size Analysis"
echo "========================"

# Check total size
echo "Total directory size:"
du -sh .

echo -e "\nTop 20 largest files/directories:"
du -sh * .[!.]* 2>/dev/null | sort -rh | head -20

echo -e "\nFiles that would be committed (excluding .gitignore):"
git ls-files | while read file; do
    size=$(du -h "$file" 2>/dev/null | cut -f1)
    echo "  $file ($size)"
done | head -30

echo -e "\nTotal size to be pushed:"
git ls-files | xargs du -ch 2>/dev/null | tail -1
