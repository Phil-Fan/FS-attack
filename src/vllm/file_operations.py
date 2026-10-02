import os
import csv
import shutil
import argparse
import filecmp
import time

def process_files(csv_path, base_dir, target_base_dir):
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for dst, src in reader:
            dst_path = os.path.join(target_base_dir, dst)
            src_path = os.path.join(base_dir, src.lstrip('./'))

            
            if os.path.exists(dst_path):
                # Compare the source and destination files
                if filecmp.cmp(src_path, dst_path, shallow=False):
                    print(f"No changes detected between {src_path} and {dst_path}. Skipping move.")
                    continue

                # Create backup of dst file with .cpy extension
                timestamp = time.strftime("%Y%m%d%H%M%S")
                backup_path = f"{dst_path}.{timestamp}.cpy"
                print(f"[FILE] Renaming dst file name!")
                os.rename(dst_path, backup_path)
            
            # Move source file to destination
            print(f"[FILE] Moving src file to dst file!")
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copy2(src_path, dst_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Copy vLLM files based on a csv list")
    parser.add_argument('--base-dir', type=str, required=True, help='Source base dir')
    parser.add_argument('--target-base-dir', type=str, default=None, help='Target base dir')
    parser.add_argument('--csv-path', type=str, default='src/vllm/file_list.csv', help='CSV file path')
    
    args = parser.parse_args()
    target_base_dir = args.target_base_dir if args.target_base_dir else args.base_dir
    
    csv_path = args.csv_path
    if not os.path.isabs(csv_path):
        csv_path = os.path.join(args.base_dir, csv_path)
    print(args.base_dir,target_base_dir,csv_path,sep='\n') # debug
    process_files(csv_path, args.base_dir, target_base_dir)
