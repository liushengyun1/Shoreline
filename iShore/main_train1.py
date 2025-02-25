import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from main_sampler import run_free_explore, run_restricted_explore
from utils.dataset import DatasetiCurb, DatasetDagger
import torch.nn.functional as F
from scipy.spatial import cKDTree
from stable_baselines3 import PPO  # 使用PPO算法


def train_SSRL(env, data, index):
    r'''
        Train iCurb with self-supervised reinforcement learning
    '''
    crop_size = env.crop_size
    train_len = env.network.train_len()

    # Load data
    cropped_feature_tensor, batch_v_now, batch_v_previous, crop_infos, batch_candidate_label_points, gt_stop_actions = data

    # Make prediction
    cropped_feature_tensor = cropped_feature_tensor.to(env.args.device).squeeze(1)
    pre_stop_actions = env.network.decoder_stop(cropped_feature_tensor,
                                                torch.FloatTensor(batch_v_now).to(env.args.device),
                                                torch.FloatTensor(batch_v_previous).to(env.args.device))
    pre_stop_actions = pre_stop_actions.squeeze(1)
    pre_coords = env.network.decoder_coord(cropped_feature_tensor, torch.FloatTensor(batch_v_now).to(env.args.device),
                                           torch.FloatTensor(batch_v_previous).to(env.args.device))

    # Generate gt labels for coord prediction
    gt_coords = []
    pre_coords_train = []
    for i, candidate_label_points in enumerate(batch_candidate_label_points):
        if candidate_label_points:
            pre_coords_train.append(pre_coords[i])
            # Convert from prediction value to image coordinate system
            tree = cKDTree(np.array(candidate_label_points).tolist())
            pre_coord = pre_coords[i].cpu().detach().numpy()
            crop_info = crop_infos[i]
            pre_coord = env.agent.train2world(pre_coord, crop_info=crop_info)
            # Find the closest gt pixels as the coord label for training
            _, ii = tree.query([pre_coord], k=[1])
            ii = ii[0]
            gt_coord = candidate_label_points[int(ii)].copy()
            gt_coord_world = gt_coord.copy()
            gt_coord[0] -= crop_info[4]
            gt_coord[1] -= crop_info[5]
            gt_coord = [x / (crop_size // 2) for x in gt_coord]
            gt_coords.append(gt_coord)

    # Define reward function for self-supervised RL
    def reward_function(pre_coords, gt_coords):
        r'''
            Reward function based on distance between predicted and ground-truth coordinates
        '''
        distances = np.linalg.norm(pre_coords - gt_coords, axis=1)
        rewards = -distances  # Negative distance as reward (closer is better)
        return rewards
    loss_stop =  0
    # Compute rewards
    if len(pre_coords_train):
        pre_coords = torch.stack(pre_coords_train).to(env.args.device)
        gt_coords = torch.FloatTensor(gt_coords).to(env.args.device)
        rewards = reward_function(pre_coords.cpu().numpy(), gt_coords.cpu().numpy())

        # Train with PPO (Proximal Policy Optimization)
        model = PPO('MlpPolicy', env, verbose=1)
        model.learn(total_timesteps=1000)  # Adjust timesteps as needed

        # Loss for stop actions
        gt_stop_actions = torch.LongTensor(gt_stop_actions).to(env.args.device)
        loss_stop = env.network.criterions['ce'](pre_stop_actions, gt_stop_actions)

        # Total loss
        loss = loss_stop
        env.network.loss = loss
        env.network.bp()
        return 0, loss_stop.item()
    else:
        loss = loss_stop
        env.network.loss = loss
        env.network.bp()
        return 0, loss_stop.item()


def run_train(env, data, iCurb_image_index):
    def train():
        r'''
            Train iCurb with self-supervised reinforcement learning
        '''
        loss_coord_ave = 0
        loss_stop_ave = 0
        dataset = DatasetDagger(env.DAgger_buffer)
        print(len(dataset))
        data_loader = DataLoader(dataset, batch_size=env.args.batch_size, shuffle=True,
                                 collate_fn=env.network.DAgger_collate)

        if len(dataset):
            for ii, data_explore in enumerate(data_loader):
                loss_coord, loss_stop = train_SSRL(env, data_explore, iCurb_image_index)
                loss_coord_ave = (loss_coord_ave * ii + loss_coord) / (ii + 1)
                loss_stop_ave = (loss_stop_ave * ii + loss_stop) / (ii + 1)
        return loss_coord_ave, loss_stop_ave

    env.training_step += 1
    train_len = env.network.train_len()

    # Get tiff image data
    seq, seq_lens, tiff, mask, name, init_points, end_points = data
    tiff = tiff.to(env.args.device)
    seq, seq_lens, mask, init_points, end_points = seq[0], seq_lens[0], mask[0], init_points[0], end_points[0]
    name = [n.split('\\')[-1].split('.')[0] for n in name]
    print(name)

    # Extract feature of the whole image to grow the graph
    _, fpn_feature_map = env.network.encoder(tiff)
    fpn_feature_map = F.interpolate(fpn_feature_map, size=(512, 512), mode='bilinear', align_corners=True)
    fpn_feature_tensor = fpn_feature_map

    # Running self-supervised RL training
    print('--------- Training iCurb with Self-Supervised RL ---------')
    loss_coord, loss_stop = train()

    # Clear the DAgger buffer
    env.DAgger_buffer = []

    # Time usage
    time_now = time.time()
    time_used = time_now - env.time_start
    time_remained = time_used / env.training_step * (env.training_image_number - env.training_step)
    speed = time_used / env.training_step
    print('Time usage: Speed {}s/im || Ut {}h || Rt {}h'.format(round(speed, 2), round(time_used / 3600, 2),
                                                                round(time_remained / 3600, 2)))

    # Print and training curve
    print('Epoch: {}/{} | Image: {}/{} | Loss: coord {}/stop {}'.format(
        env.epoch_counter, env.args.epochs, iCurb_image_index, train_len,
        round(loss_coord, 6), round(loss_stop, 6)))
    env.network.writer.add_scalar('Train/coord_loss', loss_coord, env.training_step)
    env.network.writer.add_scalar('Train/stop_loss', loss_stop, env.training_step)