import os
import re

# Build paths relative to the current directory
base_dir = os.path.join(os.getcwd(), "dataset")
input_file = os.path.join(base_dir, "dataKB_Blogs2.txt")
output_file = os.path.join(base_dir, "cleaneddataKB_Blogs.txt")

with open(input_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

cleaned_lines = []

for line in lines:
    original_line = line

    # Remove exact phrase
    line = re.sub(r"Was this helpful\?", "", line)

    # Remove "Last updated ... ago"
    line = re.sub(r"Last updated.*?ago", "", line, flags=re.IGNORECASE)

    if line != original_line:  # Something was removed
        print("Removed from line:", original_line.strip())

    cleaned_lines.append(line)

with open(output_file, "w", encoding="utf-8") as f:
    f.writelines(cleaned_lines)

print(f"\n✅ Cleaned file written to: {output_file}")
