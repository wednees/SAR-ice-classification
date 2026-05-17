
#%%
import os
import sys
# Get the absolute path of the parent directory
parent = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(parent)


import glob
import xarray as xr
from convert_raw_icechart import convert_polygon_icechart
import numpy as np
from utils import SIC_GROUPS, SOD_GROUPS, FLOE_GROUPS
import matplotlib.pyplot as plt
import time
from icecream import ic

with open('/home/jnoat92/projects/rrg-dclausi/ai4arctic/dataset/data_split_setup/train_100.txt', 'r') as f:
    scene_files = f.read().splitlines()
# with open('/home/jnoat92/projects/rrg-dclausi/ai4arctic/dataset/data_split_setup/test_file.txt', 'r') as f:
#     scene_files = f.read().splitlines()

for i in range(len(scene_files)):
    scene_files[i] = '/home/jnoat92/projects/rrg-dclausi/ai4arctic/dataset/ai4arctic_raw_train_v3/' + scene_files[i]


percentage = 1  # Percentage of scenes to sample
n_samples = len(scene_files) * percentage // 100

print("Number of scenes: %d"%(len(scene_files)))

#%%
from parallel_stuff import Parallel

LABELS = {
            'SIC': SIC_GROUPS,
            'SOD': SOD_GROUPS,
            'FLOE': FLOE_GROUPS
            }


def Obtain_histograms(iterator):
    """
    Obtain histograms of the SIC, SOD and FLOE charts from a single scene file.
    """
    scene_file, idx = iterator
    print("Processing scene: %s"%(scene_file))
    # Use h5netcdf engine to read the file
    scene = xr.open_dataset(scene_file, engine='h5netcdf')

    #  ---------- Get the SIC, SOD, FLOE Charts
    scene = convert_polygon_icechart(scene)

    hist_pixel = {
                    'SIC': np.zeros(len(LABELS['SIC'])),
                    'SOD': np.zeros(len(LABELS['SOD'])),
                    'FLOE': np.zeros(len(LABELS['FLOE']))
                }
    scene_count = {
                    'SIC': np.zeros((len(LABELS['SIC']), len(scene_files))),
                    'SOD': np.zeros((len(LABELS['SOD']), len(scene_files))),
                    'FLOE': np.zeros((len(LABELS['FLOE']), len(scene_files)))
                }

    sea_ice_maps = ['SIC', 'FLOE', 'SOD']
    for var in sea_ice_maps:
        labels, n_pixels = np.unique(scene[var].values, return_counts=True)
        for label, n_pixel in zip(labels, n_pixels):
            if label == 255 or np.isnan(label):
                continue
            hist_pixel[var][int(label)] += n_pixel
            
            scene_count[var][int(label)][idx] = 1
    
    return hist_pixel, scene_count

if __name__ == '__main__':   
    
    
    start_time = time.time()
    iterator = zip(scene_files, range(len(scene_files)))
    if len(scene_files) > 1:
        hist = Parallel(Obtain_histograms, iterator)
    else:
        hist = Obtain_histograms(iterator)
    
    end_time = time.time()
    print("Processing time: %.2f seconds"%(end_time - start_time))

    hist_pixel = {
                    'SIC': np.zeros(len(LABELS['SIC'])),
                    'SOD': np.zeros(len(LABELS['SOD'])),
                    'FLOE': np.zeros(len(LABELS['FLOE']))
                }
    scene_count = {
                    'SIC': np.zeros((len(LABELS['SIC']), len(scene_files))),
                    'SOD': np.zeros((len(LABELS['SOD']), len(scene_files))),
                    'FLOE': np.zeros((len(LABELS['FLOE']), len(scene_files)))
                }
    hist_scene = {
                    'SIC': np.zeros(len(LABELS['SIC'])),
                    'SOD': np.zeros(len(LABELS['SOD'])),
                    'FLOE': np.zeros(len(LABELS['FLOE']))
                }

    for var in ['SIC', 'SOD', 'FLOE']:
        for h_p, s_c in hist:
            hist_pixel[var] += h_p[var]
            scene_count[var] += s_c[var]
        hist_scene[var] = np.sum(scene_count[var], axis=1) 

# %%
    # for var in ['SIC', 'SOD', 'FLOE']:
    #     # Example data
    #     pixel_data = hist_pixel[var] / hist_pixel[var].sum() * 100  # Convert to percentage
    #     scene_data = hist_scene[var].astype(np.int64)
    #     labels = LABELS[var].values()

    #     # Number of groups
    #     x = np.arange(len(pixel_data))
    #     width = 0.35

    #     # Create the bar chart
    #     fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    #     # --- Pixel Counts ---
    #     ax[0].set_title('Pixel Counts')
    #     bars1 = ax[0].bar(x, pixel_data, width)
    #     ax[0].set_ylabel('%')
    #     ax[0].set_xticks(x)
    #     ax[0].set_xticklabels(labels, rotation=45)

    #     # Add value labels
    #     for bar in bars1:
    #         height = bar.get_height()
    #         ax[0].text(bar.get_x() + bar.get_width()/2, height, f'{height:.1f}', ha='center', va='bottom', fontsize=8)

    #     # --- Scene Counts ---
    #     ax[1].set_title('Scene Counts')
    #     bars2 = ax[1].bar(x, scene_data, width)
    #     ax[1].set_ylabel('%')
    #     ax[1].set_xticks(x)
    #     ax[1].set_xticklabels(labels, rotation=45)

    #     for bar in bars2:
    #         height = bar.get_height()
    #         ax[1].text(bar.get_x() + bar.get_width()/2, height, f'{height:.1f}', ha='center', va='bottom', fontsize=8)

    #     # Layout
    #     plt.tight_layout()
    #     # plt.show()
    #     plt.savefig(f'./classes_stats_{var}.png', dpi=300)

#%%

    hist_scenes_chosen = {
                    'SIC': np.zeros(len(LABELS['SIC'])),
                    'SOD': np.zeros(len(LABELS['SOD'])),
                    'FLOE': np.zeros(len(LABELS['FLOE']))
                }
    args = []
    for var in ['SIC', 'SOD', 'FLOE']:
        for i in range(len(LABELS[var])): 
            args.append((var, i))
    
    labels = list(LABELS['SIC'].values()) + list(LABELS['SOD'].values()) + list(LABELS['FLOE'].values())
    chosen_scenes = []
    for i in range(n_samples):

        hist_scene_chosen_joint = np.concatenate((hist_scenes_chosen['SIC'], hist_scenes_chosen['SOD'], hist_scenes_chosen['FLOE']))

        plt.figure(figsize=(10, 4))
        bars = plt.bar(np.arange(len(hist_scene_chosen_joint)), hist_scene_chosen_joint, 0.35, color=len(hist_scene['SIC'])*['blue'] + len(hist_scene['SOD'])*['orange'] + len(hist_scene['FLOE'])*['green'])
        plt.xticks(np.arange(len(hist_scene_chosen_joint)), labels, rotation=45)
        # Add value labels
        for bar in bars:
            height = int(bar.get_height())
            plt.text(bar.get_x() + bar.get_width()/2, height, f'{height:.1f}', ha='center', va='bottom', fontsize=8)
        plt.tight_layout()
        plt.savefig('1.png', dpi=300)
        plt.close()
        
        hist_scene_joint = np.concatenate((hist_scene['SIC'], hist_scene['SOD'], hist_scene['FLOE']))

        plt.figure(figsize=(10, 4))
        bars = plt.bar(np.arange(len(hist_scene_joint)), hist_scene_joint, 0.35, color=len(hist_scene['SIC'])*['blue'] + len(hist_scene['SOD'])*['orange'] + len(hist_scene['FLOE'])*['green'])
        plt.xticks(np.arange(len(hist_scene_joint)), labels, rotation=45)
        # Add value labels
        for bar in bars:
            height = int(bar.get_height())
            plt.text(bar.get_x() + bar.get_width()/2, height, f'{height:.1f}', ha='center', va='bottom', fontsize=8)
        plt.tight_layout()
        plt.savefig('2.png', dpi=300)
        plt.close()

        if len(chosen_scenes) >= n_samples:
            break
        
        sort_lbl = []
        for s in np.argsort(hist_scene_joint[:len(LABELS['SIC'])]):
            if hist_scene_joint[s]:
                sort_lbl.append(s)
                break
        for s in np.argsort(hist_scene_joint[len(LABELS['SIC']):len(LABELS['SIC']) + len(LABELS['SOD'])]):
            if hist_scene_joint[s]:
                sort_lbl.append(s)
                break
        for s in np.argsort(hist_scene_joint[len(LABELS['SIC']) + len(LABELS['SOD']):]):
            if hist_scene_joint[s]:
                sort_lbl.append(s)
                break

        for s in sort_lbl:
            var, lbl = args[s]
            scene_id_sorted = np.argwhere(scene_count[var][lbl] > 0)
            if not len(scene_id_sorted):
                print("Error: chosen_scene_id is empty")
                break
            for id in scene_id_sorted:
                if id[0] not in chosen_scenes:
                    chosen_scene_id = id[0]
                    break

            for var in ['SIC', 'SOD', 'FLOE']:
                hist_scenes_chosen[var] += scene_count[var][:,chosen_scene_id]
                scene_count[var][:,chosen_scene_id] = 0
                hist_scene[var] = np.sum(scene_count[var], axis=1) 
            
            chosen_scenes.append(chosen_scene_id)
            
    with open(f'finetune_{percentage}.txt', 'w') as f:
        for scene_id in sorted(chosen_scenes, reverse=True):
            f.write(os.path.split(scene_files[scene_id])[1] + '\n')
            scene_files.pop(scene_id)

    with open(f'pretrain_{100-percentage}.txt', 'w') as f:
        for scene in scene_files:
            f.write(os.path.split(scene)[1] + '\n')

    print("finetune scenes: %d"%(len(chosen_scenes)))
    print("pretrain scenes: %d"%(len(scene_files)))
