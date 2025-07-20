import os

# Configuration
root_dir = "/home/oq55olys/Projects/neural_rendering/LiDAR4D/data/kitti360/KITTI-360/data_reco_cam"
train_seqs = [0, 2, 3, 4]
val_seqs = [5, 6, 7, 9, 10]

def generate_file_list(seq_ids, output_filename):
    lines = []

    for seq in seq_ids:
        seq_dir = f"{seq:04d}"
        rgb_path = os.path.join(root_dir, seq_dir, "rgb")
        albedo_path = os.path.join(root_dir, seq_dir, "albedo")
        raydrop_path = os.path.join(root_dir, seq_dir, "raydrop")
        print(rgb_path, albedo_path, raydrop_path)
        if not (os.path.isdir(rgb_path) and os.path.isdir(albedo_path) and os.path.isdir(raydrop_path)):
            print(f"Warning: Missing directories in {seq_dir}, skipping...")
            exit()

        # Loop through rgb files
        for fname in sorted(os.listdir(rgb_path)):
            rgb_file = os.path.join(seq_dir, "rgb", fname)
            albedo_file = os.path.join(seq_dir, "albedo", fname)
            raydrop_file = os.path.join(seq_dir, "raydrop", fname)

            # Optional: Check if corresponding files exist
            if not (os.path.exists(os.path.join(root_dir, albedo_file)) and os.path.exists(os.path.join(root_dir, raydrop_file))):
                print(f"Warning: Missing albedo or raydrop for {fname} in {seq_dir}, skipping...")
                continue

            line = f"{rgb_file} {albedo_file} {raydrop_file}"
            lines.append(line)

    # Write to output file
    with open(output_filename, "w") as f:
        f.write("\n".join(lines))
    print(f"Written {len(lines)} entries to {output_filename}")

# Generate the files
generate_file_list(train_seqs, "train.txt")
generate_file_list(val_seqs, "val.txt")