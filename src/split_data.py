# # Save as src/split_data.py
# import json
# import argparse
# import os
# import random
# import math

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--input", required=True, help="Path to original data json")
#     parser.add_argument("--output_dir", required=True, help="Where to save batch files")
#     parser.add_argument("--batch_size", type=int, required=True, help="Number of prompts per batch")
#     parser.add_argument("--shuffle", action="store_true", help="Shuffle data before splitting")
#     args = parser.parse_args()

#     # Load Data
#     with open(args.input, 'r') as f:
#         data = json.load(f)
    
#     if args.shuffle:
#         random.shuffle(data)
#         print("🔀 Data shuffled.")
    
#     os.makedirs(args.output_dir, exist_ok=True)
    
#     num_batches = math.ceil(len(data) / args.batch_size)
#     print(f"📦 Splitting {len(data)} prompts into {num_batches} batches of size {args.batch_size}...")

#     # Create Batch Files
#     for i in range(num_batches):
#         batch = data[i * args.batch_size : (i + 1) * args.batch_size]
#         fname = os.path.join(args.output_dir, f"batch_{i:04d}.json")
#         with open(fname, 'w') as f:
#             json.dump(batch, f, indent=2)
        
#     print(f"✅ Created {num_batches} batch files in {args.output_dir}")

# if __name__ == "__main__":
#     main()

# Save as src/split_data.py
import json
import argparse
import os
import random
import math

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to original data json")
    parser.add_argument("--output_dir", required=True, help="Where to save batch files")
    parser.add_argument("--batch_size", type=int, required=True, help="Number of prompts per batch")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle data before splitting")
    # NEW ARGUMENT
    parser.add_argument("--limit", type=int, default=None, help="Max number of examples to use")
    
    args = parser.parse_args()

    # Load Data
    with open(args.input, 'r') as f:
        data = json.load(f)
    
    if args.shuffle:
        random.shuffle(data)
        print("🔀 Data shuffled.")

    # Apply Limit
    if args.limit is not None and args.limit > 0:
        original_len = len(data)
        data = data[:args.limit]
        print(f"⚠️  Limit applied: Reduced dataset from {original_len} to {len(data)} examples.")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    num_batches = math.ceil(len(data) / args.batch_size)
    print(f"📦 Splitting {len(data)} prompts into {num_batches} batches of size {args.batch_size}...")

    # Create Batch Files
    for i in range(num_batches):
        batch = data[i * args.batch_size : (i + 1) * args.batch_size]
        # Using simple integer naming to match your bash sorting (batch_1.json, etc)
        # Or keep your 04d formatting if you updated the bash sort -V
        fname = os.path.join(args.output_dir, f"batch_{i+1}.json") 
        with open(fname, 'w') as f:
            json.dump(batch, f, indent=2)
        
    print(f"✅ Created {num_batches} batch files in {args.output_dir}")

if __name__ == "__main__":
    main()