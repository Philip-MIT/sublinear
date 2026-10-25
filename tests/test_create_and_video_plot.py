from lerobot.datasets import LeRobotDataset
from sublinear import Sublinear, video_plot
import os
import json

API_KEY= os.getenv("SUBLINEAR_API_KEY")
client = Sublinear(api_key=API_KEY)


test_lerobot_dataset_local = "/data/sls/scratch/pschro/sole/DreamMachines/eval_actuator_unboxing_pi05_sweep_v2_01_freeze_01_full/"



test_video_path_left_wrist1 = "../example_videos/dream_machines/eval_actuator_unboxing_pi05_sweep_v2_01_freeze_01_episode_000073_view_left_wrist.mp4"
test_video_path_top1 = "../example_videos/dream_machines/eval_actuator_unboxing_pi05_sweep_v2_01_freeze_01_episode_000073_view_top.mp4"
test_video_path_right_wrist1 = "../example_videos/dream_machines/eval_actuator_unboxing_pi05_sweep_v2_01_freeze_01_episode_000073_view_right_wrist.mp4"
test_video_path_stitched1 = "../example_videos/dream_machines/eval_actuator_unboxing_pi05_sweep_v2_01_freeze_01_episode_000073_view_stitched.mp4"
test_video_json1 = "../example_videos/dream_machines/eval_actuator_unboxing_pi05_sweep_v2_01_freeze_01_episode_000073.json"

test_video_path_left_wrist2 = "../example_videos/dream_machines/eval_actuator_unboxing_pi05_sweep_v2_01_freeze_01_episode_000000_view_left_wrist.mp4"
test_video_path_top2 = "../example_videos/dream_machines/eval_actuator_unboxing_pi05_sweep_v2_01_freeze_01_episode_000000_view_top.mp4"
test_video_path_right_wrist2 = "../example_videos/dream_machines/eval_actuator_unboxing_pi05_sweep_v2_01_freeze_01_episode_000000_view_right_wrist.mp4"
test_video_path_stitched2 = "../example_videos/dream_machines/eval_actuator_unboxing_pi05_sweep_v2_01_freeze_01_episode_000000_view_stitched.mp4"
test_video_json2 = "../example_videos/dream_machines/eval_actuator_unboxing_pi05_sweep_v2_01_freeze_01_episode_000000.json"

test_video_path_left_wrist3 = "../example_videos/dream_machines/eval_actuator_unboxing_pi05_sweep_v2_01_freeze_01_episode_000025_view_left_wrist.mp4"
test_video_path_top3 = "../example_videos/dream_machines/eval_actuator_unboxing_pi05_sweep_v2_01_freeze_01_episode_000025_view_top.mp4"
test_video_path_right_wrist3 = "../example_videos/dream_machines/eval_actuator_unboxing_pi05_sweep_v2_01_freeze_01_episode_000025_view_right_wrist.mp4"
test_video_path_stitched3 = "../example_videos/dream_machines/eval_actuator_unboxing_pi05_sweep_v2_01_freeze_01_episode_000025_view_stitched.mp4"
test_video_json3 = "../example_videos/dream_machines/eval_actuator_unboxing_pi05_sweep_v2_01_freeze_01_episode_000025.json"

# read jsons
with open(test_video_json1, "r") as f:
    data1 = json.load(f)

with open(test_video_json2, "r") as f:
    data2 = json.load(f)

with open(test_video_json3, "r") as f:
    data3 = json.load(f)

test_model = "dm-1026"
test_episode_idx1 = 73
test_episode_idx2 = 0
test_episode_idx3 = 25
task_description = "Take an actuator from the box and place it in the tray"


lerobot_view_map1 = {"left_wrist": "observation.images.left_wrist", "static": "observation.images.top", "right_wrist": "observation.images.right_wrist"}
lerobot_view_map2 = {"right_wrist": "observation.images.right_wrist", "static": "observation.images.top", "left_wrist": "observation.images.left_wrist"}
lerobot_view_map3 = { "static": "observation.images.top", "right_wrist": "observation.images.right_wrist", "left_wrist": "observation.images.left_wrist"}


max_frames_list = [20, 50, 9999]
test_downsample_to=30




dataset = LeRobotDataset(repo_id=f"local/{test_lerobot_dataset_local.split('/')[-2]}", root=test_lerobot_dataset_local,)
lerobot_response_list = []
lerobot_corr_list = []
view_type_map_list = [lerobot_view_map1, lerobot_view_map2, lerobot_view_map3]
view_type_map_idx=0
for lerobot_view_map in view_type_map_list:
    max_frame_idx = 0
    for max_frames in max_frames_list:
        lerobot_response = client.responses.create(
            model=test_model, 
            task_description=task_description, 
            # 
            lerobot_dataset=dataset, 
            episodes=[test_episode_idx1], 
            lerobot_view_map=lerobot_view_map, 
            # 
            verbose=True, 
            max_frames=max_frames, 
            downsample_to=test_downsample_to
        )
        lerobot_response_list.append(lerobot_response)
        # 
        corr = np.corrcoef(lerobot_response.rewards[0], data1['progress'])[0, 1]
        lerobot_corr_list.append(round(corr, 3))
        # 
        video_plot(
            outputs=[{"model": test_model, "rewards": lerobot_response.rewards[0]}],
            plot_save_path=f"../tests/tmp/test_plot_lerobot_{view_type_map_idx}_{max_frame_idx}.mp4",
            lerobot_dataset=dataset, 
            episode=test_episode_idx1,
            lerobot_view_map=lerobot_view_map,
            plot_width_relative_to_video=0.42,
            fps=15,
        )
        view_type_map_idx += 1
        max_frame_idx += 1



print('lerobot_corr_list', lerobot_corr_list)

# confirm all rewards are same length
if type(lerobot_response_list[0].rewards[0]) is list:
    lerobot_response_len_list = [len(response.rewards[0]) for response in lerobot_response_list]
else:
    lerobot_response_len_list = [len(response.rewards) for response in lerobot_response_list]

print(lerobot_response_len_list)
all_same_length = all(length == lerobot_response_len_list[0] for length in lerobot_response_len_list)
assert all_same_length, "Not all rewards have the same length"

# confirm all rewards are same across different view maps 
assert lerobot_response_list[0].rewards==lerobot_response_list[3].rewards==lerobot_response_list[6].rewards, "Rewards do not match for max_frames=20"
assert lerobot_response_list[1].rewards==lerobot_response_list[4].rewards==lerobot_response_list[7].rewards, "Rewards do not match for max_frames=50"
assert lerobot_response_list[2].rewards==lerobot_response_list[5].rewards==lerobot_response_list[8].rewards, "Rewards do not match for max_frames=9999"

# confirm rewards are different across different max lengths
assert lerobot_response_list[0].rewards!=lerobot_response_list[1].rewards, "Rewards should differ for different max lengths"
assert lerobot_response_list[0].rewards!=lerobot_response_list[2].rewards, "Rewards should differ for different max lengths"
assert lerobot_response_list[1].rewards!=lerobot_response_list[2].rewards, "Rewards should differ for different max lengths"


view_type_map1 = {
    'left_wrist': test_video_path_left_wrist1,
    'static': test_video_path_top1,
    'right_wrist': test_video_path_right_wrist1,
}
view_type_map2 = {
    'right_wrist': test_video_path_right_wrist1,
    'static': test_video_path_top1,
    'left_wrist': test_video_path_left_wrist1,
}

view_type_map3 = {
    'static': test_video_path_top1,
    'right_wrist': test_video_path_right_wrist1,
    'left_wrist': test_video_path_left_wrist1,
}

video_paths_response_list = []
view_type_map_list = [view_type_map1, view_type_map2, view_type_map3]
view_type_map_idx = 0
for view_type_map in view_type_map_list:
    max_frame_idx = 0
    for max_frames in max_frames_list:
        video_paths_response = client.responses.create(
            model=test_model, 
            task_description=task_description, 
            # 
            video_paths_view1 = [list(view_type_map.values())[0]],
            video_paths_view2 = [list(view_type_map.values())[1]],
            video_paths_view3 = [list(view_type_map.values())[2]],
            view_type=view_type_map.keys(),
            # 
            verbose=True, 
            max_frames=max_frames, 
            downsample_to=test_downsample_to 
        )
        video_paths_response_list.append(video_paths_response)
        # 
        video_plot(
            outputs=[{"model": test_model, "rewards": video_paths_response.rewards[0]}],
            plot_save_path=f"../tests/tmp/test_plot_lerobot_{view_type_map_idx}_{max_frame_idx}.mp4",
            video_path_view1 = [list(view_type_map.values())[0]],
            video_path_view2 = [list(view_type_map.values())[1]],
            video_path_view3 = [list(view_type_map.values())[2]],
            view_type = list(view_type_map.keys()),
            plot_width_relative_to_video=0.42,
            fps=15,
        )
        max_frame_idx += 1
    view_type_map_idx += 1



# confirm all rewards are same length
if type(video_paths_response_list[0].rewards[0]) is list:
    video_paths_response_len_list = [len(response.rewards[0]) for response in video_paths_response_list]
else:
    video_paths_response_len_list = [len(response.rewards) for response in video_paths_response_list]

print(video_paths_response_len_list)
all_same_length_video_paths = all(length == video_paths_response_len_list[0] for length in video_paths_response_len_list)
assert all_same_length_video_paths, "Not all video paths responses have the same length"

# confirm all rewards are same across different view maps 
assert video_paths_response_list[0].rewards==video_paths_response_list[3].rewards==video_paths_response_list[6].rewards, "Rewards do not match for max_frames=20"
assert video_paths_response_list[1].rewards==video_paths_response_list[4].rewards==video_paths_response_list[7].rewards, "Rewards do not match for max_frames=50"
assert video_paths_response_list[2].rewards==video_paths_response_list[5].rewards==video_paths_response_list[8].rewards, "Rewards do not match for max_frames=9999"

# confirm rewards are different across different max lengths
assert video_paths_response_list[0].rewards!=video_paths_response_list[1].rewards, "Rewards should differ for different max lengths"
assert video_paths_response_list[0].rewards!=video_paths_response_list[2].rewards, "Rewards should differ for different max lengths"
assert video_paths_response_list[1].rewards!=video_paths_response_list[2].rewards, "Rewards should differ for different max lengths"




from sublinear import load_video_frames
test_video_frames_left_wrist = load_video_frames(test_video_path_left_wrist1, max_frames=99999)
test_video_frames_top = load_video_frames(test_video_path_top1, max_frames=99999)
test_video_frames_right_wrist = load_video_frames(test_video_path_right_wrist1, max_frames=99999)

type(test_video_frames_left_wrist)
type(test_video_frames_left_wrist[0])
# >>> type(test_video_frames_left_wrist)
# <class 'list'>
# >>> type(test_video_frames_left_wrist[0])
# <class 'numpy.ndarray'>


view_type_map1 = {
    'left_wrist': test_video_frames_left_wrist,
    'static': test_video_frames_top,
    'right_wrist': test_video_frames_right_wrist,
}

view_type_map2 = {
    'right_wrist': test_video_frames_right_wrist,
    'static': test_video_frames_top,
    'left_wrist': test_video_frames_left_wrist,
}

view_type_map3 = {
    'static': test_video_frames_top,
    'right_wrist': test_video_frames_right_wrist,
    'left_wrist': test_video_frames_left_wrist,
}

video_frames_response_list = []
view_type_map_list = [view_type_map1, view_type_map2, view_type_map3]
for view_type_map in view_type_map_list:
    for max_frames in max_frames_list:
        video_frames_response = client.responses.create(
            model=test_model, 
            task_description=task_description, 
            # 
            video_frames_view1 = [list(view_type_map.values())[0]],
            video_frames_view2 = [list(view_type_map.values())[1]],
            video_frames_view3 = [list(view_type_map.values())[2]],
            view_type=view_type_map.keys(),
            # 
            verbose=True, 
            max_frames=max_frames, 
            downsample_to=test_downsample_to 
        )
        video_frames_response_list.append(video_frames_response)


# confirm all rewards are same length
if type(video_frames_response_list[0].rewards[0]) is list:
    reward_lengths = [len(response.rewards[0]) for response in video_frames_response_list]
else:
    reward_lengths = [len(response.rewards) for response in video_frames_response_list]

all_same_length = all(length == reward_lengths[0] for length in reward_lengths)
assert all_same_length, "Not all rewards have the same length"

# confirm reward_lengths matches video frames length
assert reward_lengths[0] == len(test_video_frames_left_wrist), "Reward length does not match left wrist video frames"
assert lerobot_response_len_list[0]==len(test_video_frames_left_wrist), "Lerobot response length does not match left wrist video frames"
assert video_paths_response_len_list[0]==len(test_video_frames_left_wrist), "Video paths response length does not match left wrist video frames"


# confirm all rewards are same across different view maps 
assert video_frames_response_list[0].rewards==video_frames_response_list[3].rewards==video_frames_response_list[6].rewards, "Rewards do not match for max_frames=20"
assert video_frames_response_list[1].rewards==video_frames_response_list[4].rewards==video_frames_response_list[7].rewards, "Rewards do not match for max_frames=50"
assert video_frames_response_list[2].rewards==video_frames_response_list[5].rewards==video_frames_response_list[8].rewards, "Rewards do not match for max_frames=9999"

# confirm rewards are different across different max lengths
assert video_frames_response_list[0].rewards!=video_frames_response_list[1].rewards, "Rewards should differ for different max lengths"
assert video_frames_response_list[0].rewards!=video_frames_response_list[2].rewards, "Rewards should differ for different max lengths"
assert video_frames_response_list[1].rewards!=video_frames_response_list[2].rewards, "Rewards should differ for different max lengths"





import numpy as np
test_video_frames_left_wrist_color_change = [np.moveaxis(frame, -1, 0) for frame in test_video_frames_left_wrist]
test_video_frames_top_color_change = [np.moveaxis(frame, -1, 0) for frame in test_video_frames_top]
test_video_frames_right_wrist_color_change = [np.moveaxis(frame, -1, 0) for frame in test_video_frames_right_wrist]

test_video_frames_left_wrist_color_change[0].shape
# (3, 224, 224)


test_video_frames_left_wrist_numpy = np.array(test_video_frames_left_wrist)
test_video_frames_top_numpy = np.array(test_video_frames_top)
test_video_frames_right_wrist_numpy = np.array(test_video_frames_right_wrist)

type(test_video_frames_left_wrist_numpy)
type(test_video_frames_left_wrist_numpy[0])
# >>> type(test_video_frames_left_wrist_numpy)
# <class 'numpy.ndarray'>
# >>> type(test_video_frames_left_wrist_numpy[0])
# <class 'numpy.ndarray'>

test_video_frames_left_wrist_numpy.shape # T, H, W, C
# (802, 224, 224, 3)

test_video_frames_left_wrist_numpy_color_change = np.moveaxis(test_video_frames_left_wrist_numpy, -1, 1) 
test_video_frames_top_numpy_color_change = np.moveaxis(test_video_frames_top_numpy, -1, 1) 
test_video_frames_right_wrist_numpy_color_change = np.moveaxis(test_video_frames_right_wrist_numpy, -1, 1) 

test_video_frames_left_wrist_numpy_color_change.shape # T, C, H, W
# (802, 3, 224, 224)

test_video_frames_left_wrist_numpy_batch = np.expand_dims(test_video_frames_left_wrist_numpy, axis=0)
test_video_frames_top_numpy_batch = np.expand_dims(test_video_frames_top_numpy, axis=0)
test_video_frames_right_wrist_numpy_batch = np.expand_dims(test_video_frames_right_wrist_numpy, axis=0)

test_video_frames_left_wrist_numpy_batch.shape # B, T, H, W, C
# (1, 802, 224, 224, 3)



test_video_frames_left_wrist_numpy_color_change_batch = np.expand_dims(test_video_frames_left_wrist_numpy_color_change, axis=0)
test_video_frames_top_numpy_color_change_batch = np.expand_dims(test_video_frames_top_numpy_color_change, axis=0)
test_video_frames_right_wrist_numpy_color_change_batch = np.expand_dims(test_video_frames_right_wrist_numpy_color_change, axis=0)

test_video_frames_left_wrist_numpy_color_change_batch.shape # B, T, C, H, W
# (1, 802, 3, 224, 224)





view_type_map1 = {
    'left_wrist': test_video_frames_left_wrist,
    'static': test_video_frames_top,
    'right_wrist': test_video_frames_right_wrist,
}

view_type_map2 = {
    'left_wrist': test_video_frames_left_wrist_color_change,
    'static': test_video_frames_top_color_change,
    'right_wrist': test_video_frames_right_wrist_color_change,
}

view_type_map3 = {
    'left_wrist': test_video_frames_left_wrist_numpy,
    'static': test_video_frames_top_numpy,
    'right_wrist': test_video_frames_right_wrist_numpy,
}

view_type_map4 = {
    'left_wrist': test_video_frames_left_wrist_numpy_color_change,
    'static': test_video_frames_top_numpy_color_change,
    'right_wrist': test_video_frames_right_wrist_numpy_color_change,
}

view_type_map5 = {
    'left_wrist': test_video_frames_left_wrist_numpy_batch,
    'static': test_video_frames_top_numpy_batch,
    'right_wrist': test_video_frames_right_wrist_numpy_batch,
}

view_type_map6 = {
    'left_wrist': test_video_frames_left_wrist_numpy_color_change_batch,
    'static': test_video_frames_top_numpy_color_change_batch,
    'right_wrist': test_video_frames_right_wrist_numpy_color_change_batch,
}


video_frames_response_list2 = []
view_type_map_list = [view_type_map1, view_type_map2, view_type_map3, view_type_map4, view_type_map5, view_type_map6]
for view_type_map in view_type_map_list:
    for max_frames in max_frames_list:
        video_frames_response = client.responses.create(
            model=test_model, 
            task_description=task_description, 
            # 
            video_frames_view1 = [list(view_type_map.values())[0]],
            video_frames_view2 = [list(view_type_map.values())[1]],
            video_frames_view3 = [list(view_type_map.values())[2]],
            view_type=view_type_map.keys(),
            # 
            verbose=True, 
            max_frames=max_frames, 
            downsample_to=test_downsample_to 
        )
        video_frames_response_list2.append(video_frames_response)

# confirm all rewards are same length
if type(video_frames_response_list2[0].rewards[0]) == list:
    reward_lengths = [len(response.rewards[0]) for response in video_frames_response_list2]
else:
    reward_lengths = [len(response.rewards) for response in video_frames_response_list2]

all_same_length = all(length == reward_lengths[0] for length in reward_lengths)
assert all_same_length, "Not all rewards have the same length"

# confirm all rewards are same across different view maps 
assert video_frames_response_list2[0*len(max_frames_list)].rewards==video_frames_response_list2[1*len(max_frames_list)].rewards==video_frames_response_list2[2*len(max_frames_list)].rewards==video_frames_response_list2[3*len(max_frames_list)].rewards==video_frames_response_list2[4*len(max_frames_list)].rewards==video_frames_response_list2[5*len(max_frames_list)].rewards, "Rewards do not match for max_frames=20"
assert video_frames_response_list2[0*len(max_frames_list)+1].rewards==video_frames_response_list2[1*len(max_frames_list)+1].rewards==video_frames_response_list2[2*len(max_frames_list)+1].rewards==video_frames_response_list2[3*len(max_frames_list)+1].rewards==video_frames_response_list2[4*len(max_frames_list)+1].rewards==video_frames_response_list2[5*len(max_frames_list)+1].rewards, "Rewards do not match for max_frames=50"
assert video_frames_response_list2[0*len(max_frames_list)+2].rewards==video_frames_response_list2[1*len(max_frames_list)+2].rewards==video_frames_response_list2[2*len(max_frames_list)+2].rewards==video_frames_response_list2[3*len(max_frames_list)+2].rewards==video_frames_response_list2[4*len(max_frames_list)+2].rewards==video_frames_response_list2[5*len(max_frames_list)+2].rewards, "Rewards do not match for max_frames=9999"


# confirm rewards are different across different max lengths
for i in range(len(view_type_map_list)):
    for j in range(len(max_frames_list)):
        for k in range(j + 1, len(max_frames_list)):
            assert (
                video_frames_response_list3[i * len(max_frames_list) + j].rewards !=
                video_frames_response_list3[i * len(max_frames_list) + k].rewards
            ), f"Rewards should differ for max_frames={max_frames_list[j]} and {max_frames_list[k]}"






# try torch tensors
import torch

test_video_frames_left_wrist_torch = torch.from_numpy(test_video_frames_left_wrist_numpy)
test_video_frames_top_torch = torch.from_numpy(test_video_frames_top_numpy)
test_video_frames_right_wrist_torch = torch.from_numpy(test_video_frames_right_wrist_numpy)

type(test_video_frames_left_wrist_torch)
type(test_video_frames_left_wrist_torch[0])
# >>> type(test_video_frames_left_wrist_torch)
# <class 'torch.Tensor'>
# >>> type(test_video_frames_left_wrist_torch[0])
# <class 'torch.Tensor'>

test_video_frames_left_wrist_torch.shape # T, H, W, C
# torch.Size([802, 224, 224, 3])


test_video_frames_left_wrist_torch_color_change = test_video_frames_left_wrist_torch.movedim(-1, 1)
test_video_frames_top_torch_color_change = test_video_frames_top_torch.movedim(-1, 1)
test_video_frames_right_wrist_torch_color_change = test_video_frames_right_wrist_torch.movedim(-1, 1)

test_video_frames_left_wrist_torch_color_change.shape # T, C, H, W
# torch.Size([802, 3, 224, 224])


test_video_frames_left_wrist_torch_batch = test_video_frames_left_wrist_torch.unsqueeze(0)
test_video_frames_top_torch_batch = test_video_frames_top_torch.unsqueeze(0)
test_video_frames_right_wrist_torch_batch = test_video_frames_right_wrist_torch.unsqueeze(0)

test_video_frames_left_wrist_torch_batch.shape # B, T, H, W, C
# torch.Size([1, 802, 224, 224, 3])


test_video_frames_left_wrist_torch_color_change_batch = test_video_frames_left_wrist_torch_color_change.unsqueeze(0)
test_video_frames_top_torch_color_change_batch = test_video_frames_top_torch_color_change.unsqueeze(0)
test_video_frames_right_wrist_torch_color_change_batch = test_video_frames_right_wrist_torch_color_change.unsqueeze(0)

test_video_frames_left_wrist_torch_color_change_batch.shape # B, T, C, H, W
# torch.Size([1, 802, 3, 224, 224])





view_type_map1 = {
    'left_wrist': test_video_frames_left_wrist,
    'static': test_video_frames_top,
    'right_wrist': test_video_frames_right_wrist,
}

view_type_map2 = {
    'left_wrist': test_video_frames_left_wrist_torch,
    'static': test_video_frames_top_torch,
    'right_wrist': test_video_frames_right_wrist_torch,
}

view_type_map3 = {
    'left_wrist': test_video_frames_left_wrist_torch_color_change,
    'static': test_video_frames_top_torch_color_change,
    'right_wrist': test_video_frames_right_wrist_torch_color_change,
}

view_type_map4 = {
    'left_wrist': test_video_frames_left_wrist_torch_batch,
    'static': test_video_frames_top_torch_batch,
    'right_wrist': test_video_frames_right_wrist_torch_batch,
}

view_type_map5 = {
    'left_wrist': test_video_frames_left_wrist_torch_color_change_batch,
    'static': test_video_frames_top_torch_color_change_batch,
    'right_wrist': test_video_frames_right_wrist_torch_color_change_batch,
}

view_type_map6 = {
    'left_wrist': test_video_frames_left_wrist_numpy,
    'static': test_video_frames_top_numpy,
    'right_wrist': test_video_frames_right_wrist_numpy,
}



video_frames_response_list3 = []
view_type_map_list = [view_type_map1, view_type_map2, view_type_map3, view_type_map4, view_type_map5, view_type_map6]

for view_type_map in view_type_map_list:
    for max_frames in max_frames_list:
        video_frames_response = client.responses.create(
            model=test_model,
            task_description=task_description,
            #
            video_frames_view1=[list(view_type_map.values())[0]],
            video_frames_view2=[list(view_type_map.values())[1]],
            video_frames_view3=[list(view_type_map.values())[2]],
            view_type=view_type_map.keys(),
            #
            verbose=True,
            max_frames=max_frames,
            downsample_to=test_downsample_to
        )
        video_frames_response_list3.append(video_frames_response)


# confirm all rewards are same length
if type(video_frames_response_list3[0].rewards[0]) == list:
    reward_lengths = [len(response.rewards[0]) for response in video_frames_response_list3]
else:
    reward_lengths = [len(response.rewards) for response in video_frames_response_list3]

all_same_length = all(length == reward_lengths[0] for length in reward_lengths)
assert all_same_length, "Not all rewards have the same length"


# confirm all rewards are same across different view maps
for i in range(len(max_frames_list)):
    assert all(
        video_frames_response_list3[j * len(max_frames_list) + i].rewards ==
        video_frames_response_list3[i].rewards
        for j in range(1, len(view_type_map_list))
    ), f"Rewards do not match for max_frames={max_frames_list[i]}"


# confirm rewards are different across different max lengths
for i in range(len(view_type_map_list)):
    for j in range(len(max_frames_list)):
        for k in range(j + 1, len(max_frames_list)):
            assert (
                video_frames_response_list3[i * len(max_frames_list) + j].rewards !=
                video_frames_response_list3[i * len(max_frames_list) + k].rewards
            ), f"Rewards should differ for max_frames={max_frames_list[j]} and {max_frames_list[k]}"




