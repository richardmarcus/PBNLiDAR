import os
import re
import numpy as np
import pandas as pd
import shutil

# folder names are like kitti360_lidar4d_fxxxx_setting
# relevant metrics in kitti360_lidar4d_fxxxx_setting/log_default.txt
# Rdrop_error (RMSE, Acc, F1) = [0.11492169 0.98236084 0.99077173]
# == ↓ Final pred ↓ == RMSE      MedAE      LPIPS        SSIM        PSNR ===
# Inten_error = [ 0.10319256  0.0450173   0.27644364  0.48518251 19.73266183]
# Depth_error = [ 1.163459    0.03843233  0.06657864  0.95981822 36.77213713]
# Point_error (CD, F-score) = [0.05145555 0.9532318 ]

# use last metrics evaluation if there are multiple
# for each setting, add the metrics of the same fxxxx together, average them and write to .csv

id = "log"
id = "log_rfc"
#experiment = "default"
experiment = "default_distance"
experiment = "big_improved"

log_path = "/home/oq55olys/Cluster/LiDAR4D/"+id+"/"
log_path = "/home/oq55olys/Projects/neural_rendering/LiDAR4D/"+id+"/"
#make folder all_img
#log_path = "/home/oq55olys/Cluster/LiDAR4D/"+id+"/all_img/"

#static: 1538 1728 1908 3353
#dynamic: 2350 4950 8120 10200 10750 11400

def parse_log_file(file_path):
    """Parses a log file and extracts relevant metrics.

    Args:
        file_path (str): Path to the log file.

    Returns:
        dict: A dictionary containing the extracted metrics, or None if parsing fails.
    """
    metrics = {}
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()

            # Initialize lists to store multiple occurrences of metrics
            rdrop_errors = []
            inten_errors = []
            depth_errors = []
            point_errors = []

            for i, line in enumerate(lines):
                if "Rdrop_error (RMSE, Acc, F1) =" in line:
                    match = re.search(r"\[(.*?)\]", line)
                    if match:
                        rdrop_errors.append(list(map(float, re.findall(r"[-+]?\d*\.\d+|\d+", match.group(1)))))
                elif "== ↓ Final pred ↓ ==" in line:
                    inten_line_index = i + 1
                    depth_line_index = i + 2
                    point_line_index = i + 3

                    # Extract Inten_error
                    inten_line = lines[inten_line_index]
                    match = re.search(r"\[(.*?)\]", inten_line)
                    if match:
                        inten_errors.append(list(map(float, re.findall(r"[-+]?\d*\.\d+|\d+", match.group(1)))))

                    # Extract Depth_error
                    depth_line = lines[depth_line_index]
                    match = re.search(r"\[(.*?)\]", depth_line)
                    if match:
                        depth_errors.append(list(map(float, re.findall(r"[-+]?\d*\.\d+|\d+", match.group(1)))))

                    # Extract Point_error
                    point_line = lines[point_line_index]
                    match = re.search(r"\[(.*?)\]", point_line)
                    if match:
                        point_errors.append(list(map(float, re.findall(r"[-+]?\d*\.\d+|\d+", match.group(1)))))

            #if reflectance in file name take the first occurrence of each metric
            if False:#"reflectance" in file_path:
                if rdrop_errors:
                    metrics['Rdrop_error'] = rdrop_errors[0]
                if inten_errors:
                    metrics['Inten_error'] = inten_errors[0]
                if depth_errors:
                    metrics['Depth_error'] = depth_errors[0]
                if point_errors:
                    metrics['Point_error'] = point_errors[0]
            else:
                # Use the last occurrence of each metric if multiple exist
                if rdrop_errors:
                    metrics['Rdrop_error'] = rdrop_errors[-1]
                if inten_errors:
                    metrics['Inten_error'] = inten_errors[-1]
                if depth_errors:
                    metrics['Depth_error'] = depth_errors[-1]
                if point_errors:
                    metrics['Point_error'] = point_errors[-1]

        return metrics
    except Exception as e:
        print(f"Error parsing file {file_path}: {e}")
        return None


def combine_and_average_logs(log_path):
    """Combines metrics from log files, averages them, and saves to a CSV file.

    Args:
        log_path (str): Path to the directory containing the log files.
    """
    out_path = "/home/oq55olys/Projects/neural_rendering/LiDAR4D"

    combined_log_path = os.path.join(out_path, "combined_"+ id)
    if not os.path.exists(combined_log_path):
        os.makedirs(combined_log_path)
        print(f"Created directory: {combined_log_path}")

        # Copy folder structure and .txt files
        for folder_name in os.listdir(log_path):
            source_folder = os.path.join(log_path, folder_name)
            dest_folder = os.path.join(combined_log_path, folder_name)
            if os.path.isdir(source_folder):
                os.makedirs(dest_folder, exist_ok=True)
                for item in os.listdir(source_folder):
                    source_item = os.path.join(source_folder, item)
                    dest_item = os.path.join(dest_folder, item)
                    if item.endswith(".txt") and os.path.isfile(source_item):
                        shutil.copy2(source_item, dest_item)  # copy2 preserves metadata
        print(f"Copied folder structure and .txt files to: {combined_log_path}")

    # Use the copied log path
    copied_log_path = combined_log_path

    all_data = {}
    for folder_name in os.listdir(copied_log_path):
        if "kitti360_lidar4d_f" in folder_name:
            #if "11400" in folder_name:
            #    print(f"Skipping folder: {folder_name}")
            #    continue
            parts = folder_name.split('_')
            f_number = parts[2]  # Extract fxxxx
            #remove leading f
            f_number = f_number.lstrip('f')
            setting = parts[3] if len(parts) > 3 else 'default'  # Extract setting
            #check if setting is a number then setting is parts[4]
            if setting.isdigit():
                begin_id = 4
            else:
                begin_id = 3

            setting = "_".join(parts[begin_id:])  # Extract setting
            log_file_path = os.path.join(copied_log_path, folder_name, "debuglog_"+experiment+".txt")
            img_folder = os.path.join(copied_log_path, folder_name, "validation")
            #copy images from img_folder to combined_log_path/all_img
            if not os.path.exists(os.path.join(combined_log_path, "all_img")):
                os.makedirs(os.path.join(combined_log_path, "all_img"))
            if os.path.exists(img_folder):
                for img_file in os.listdir(img_folder):
                    source_img_file = os.path.join(img_folder, img_file)
                    #img_id is last four digits before the .png
                    imgid = img_file.split('_')[-1].split('.')[0]
                    destfile = f"{f_number}_{imgid}_{setting}.png"
                    dest_img_file = os.path.join(combined_log_path, "all_img", destfile)


                    if os.path.isfile(source_img_file):
                        shutil.copy2(source_img_file, dest_img_file)
            

            if os.path.exists(log_file_path):
                #print(f"Parsing log file: {log_file_path}")
                metrics = parse_log_file(log_file_path)
                if metrics:
                    key = (f_number, setting)  # Use a tuple as the key
                    if key not in all_data:
                        all_data[key] = []
                    all_data[key].append(metrics)
                    #print(f"Metrics added for f_number: {f_number}")
                else:
                    print(f"No metrics found in {log_file_path}")
            else:
                print(f"Log file not found: {log_file_path}")

    # Do not average the metrics, directly use the metrics
    averaged_data = {}
    for (f_number, setting), data_list in all_data.items():
        if data_list:
            # Directly use the metrics from the single data point
            averaged_data[(f_number, setting)] = data_list[0]
            print(f"Metrics used for f_number: {f_number}, setting: {setting}")

    # Convert the averaged data to a Pandas DataFrame
    df_data = []
    for (f_number, setting), metrics in averaged_data.items():
        row = {'f_number': f_number, 'setting': setting}
        for metric_name, value in metrics.items():
            if metric_name == 'Rdrop_error':
                row['Rdrop_RMSE'] = value[0]
                row['Rdrop_Acc'] = value[1]
                row['Rdrop_F1'] = value[2]
            elif metric_name == 'Inten_error':
                row['Inten_RMSE'] = value[0]
                row['Inten_MedAE'] = value[1]
                row['Inten_LPIPS'] = value[2]
                row['Inten_SSIM'] = value[3]
                row['Inten_PSNR'] = value[4]
            elif metric_name == 'Depth_error':
                row['Depth_RMSE'] = value[0]
                row['Depth_MedAE'] = value[1]
                row['Depth_LPIPS'] = value[2]
                row['Depth_SSIM'] = value[3]
                row['Depth_PSNR'] = value[4]
            elif metric_name == 'Point_error':
                row['Point_CD'] = value[0]
                row['Point_F-score'] = value[1]
            else:
                row[metric_name] = value
        df_data.append(row)

    df = pd.DataFrame(df_data)

    df = df.sort_values(by=['f_number', 'setting'])

    dynamic_settings = ['2350', '4950', '8120', '10200', '10750', '11400']
    static_settings = ['1538', '1728', '1908', '3353']
    # Add a new column to indicate whether the setting is dynamic or static
    df['is_dynamic'] = df['f_number'].apply(lambda x: 1 if x in dynamic_settings else 0)

    # Separate dynamic and static settings
    df_dynamic = df[df['is_dynamic'] == 1]
    df_static = df[df['is_dynamic'] == 0]

    modes = ["all", "dynamic", "static"]
    # Group by setting and calculate the mean
    df_dynamic_averaged = df_dynamic.groupby('setting').mean(numeric_only=True)
    df_static_averaged = df_static.groupby('setting').mean(numeric_only=True)
    df_all_averaged = df.groupby('setting').mean(numeric_only=True)

    for mode in modes:

        if mode == "all":
            df_averaged = df_all_averaged
        elif mode == "dynamic":
            df_averaged = df_dynamic_averaged
        elif mode == "static":
            df_averaged = df_static_averaged


        # Add a column to indicate whether the setting is dynamic or static
        #df_dynamic_averaged['is_dynamic'] = 1
        #df_static_averaged['is_dynamic'] = 0

         #sort by medae
        df_averaged = df_averaged.sort_values(by=['Inten_MedAE'], ascending=True)

        # Reset index
        df_averaged = df_averaged.reset_index()

              #round to 4 decimal places
        # Format to 4 decimal places with trailing zeros
        for col in df_averaged.columns:
            if df_averaged[col].dtype == 'float64':
             df_averaged[col] = df_averaged[col].map(lambda x: f'{x:.4f}')

       

        # Save the DataFrame to a CSV file
        csv_file_path = os.path.join(combined_log_path, f"averaged_metrics_{mode}.csv")
        df_averaged.to_csv(csv_file_path, index=False)
        print(f"Averaged metrics saved to {csv_file_path}")

        # Save the DataFrame to a formatted TXT file
        txt_file_path = os.path.join(combined_log_path, f"averaged_metrics_{mode}.txt")
        with open(txt_file_path, 'w') as f:
            f.write(df_averaged.to_string(index=False))
        print(f"Averaged metrics saved to {txt_file_path}")



# Example usage:
combine_and_average_logs(log_path)
