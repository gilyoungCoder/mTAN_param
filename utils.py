# pylint: disable=E1101
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
from physionet import PhysioNet, get_data_min_max
from sklearn import model_selection
from sklearn import metrics
import random
def diversity_regularization(time_points, drate=0.1):
    """
    시간 점들 간의 다양성을 증진시키기 위한 레귤라리제이션 항을 계산
    
    :param time_points: 생성된 시간 점들의 텐서, 크기는 [batch_size, num_points].
    :param min_distance: 점들 간의 최소 거리 임곗값.
    :param penalty_scale: 레귤라리제이션의 스케일.
    :return: 레귤라리제이션 항.
    """
    # 시간 점들 간의 거리 계산
    steps = time_points.size(-1)
    min_distance = drate * 1 / steps
    diffs = time_points.unsqueeze(2) - time_points.unsqueeze(1)
    distances = torch.abs(diffs)
    
    # 임곗값보다 작은 거리에 대한 패널티 계산
    penalties = torch.relu(min_distance - distances)
    
    # 레귤라리제이션 항 계산
    regularization = penalties.sum() / (time_points.size(0) * time_points.size(1) * (time_points.size(1) - 1))
    
    return regularization

import torch

def batch_cosine_similarity_penalty(X):
    # Cosine similarity를 계산하기 전에 입력 X의 크기를 확인합니다.
    # X의 차원: (batch size, time point, feature)
    batch_size, time_point, num_features = X.shape
    
    # 각 배치에 대해 독립적으로 cosine similarity를 계산하기 위해 반복문을 사용합니다.
    penalties = []
    for batch in range(batch_size):
        # 현재 배치 선택: (time point, feature)
        current_batch = X[batch]
        
        # 정규화
        norm = torch.norm(current_batch, p=2, dim=1, keepdim=True)
        current_batch_norm = current_batch / (norm + 1e-6)
        
        # Cosine similarity 계산
        similarity_matrix = torch.matmul(current_batch_norm, current_batch_norm.T)
        
        # 자기 자신과의 similarity(1)를 제외하고 나머지 값들의 합을 구합니다.
        num_elements = time_point
        penalty = (similarity_matrix.sum() - num_elements) / (num_elements * (num_elements - 1))
        
        penalties.append(penalty)
    
    # 모든 배치에 대한 평균 penalty 계산
    average_penalty = sum(penalties) / batch_size
    
    return average_penalty

def spread_regularization_loss(hidden_states):
    batch_size, num_elements, _ = hidden_states.size()
    
    # Expand the hidden states to calculate differences
    expanded_states = hidden_states.unsqueeze(1) - hidden_states.unsqueeze(2)
    
    # Calculate the Euclidean distance for each pair of points
    distance_matrix = torch.sqrt(torch.sum(expanded_states ** 2, dim=-1) + 1e-9)
    
    # Calculate the inverse of distances
    inv_distances = 1.0 / distance_matrix
    
    # Create a mask to zero out the diagonals
    mask = torch.eye(num_elements).to(hidden_states.device)
    mask = mask.unsqueeze(0).expand(batch_size, num_elements, num_elements)
    inv_distances = inv_distances * (1 - mask)
    
    # Calculate the mean of the inverse distances
    loss = inv_distances.sum() / (batch_size * num_elements * (num_elements - 1))
    return loss

def efficient_spread_regularization_loss(hidden_states):
    batch_size, num_elements, _ = hidden_states.size()
    loss = 0.0

    for i in range(num_elements):
        diff = hidden_states - hidden_states[:, i:i+1, :]
        distance = torch.sqrt(torch.sum(diff ** 2, dim=-1) + 1e-9)
        inv_distance = 1.0 / distance
        inv_distance[:, i] = 0  # 대각선 요소를 0으로 설정
        loss += inv_distance.sum()

    loss /= (batch_size * num_elements * (num_elements - 1))
    return loss

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def log_normal_pdf(x, mean, logvar, mask):
    const = torch.from_numpy(np.array([2. * np.pi])).float().to(x.device)
    const = torch.log(const)
    return -.5 * (const + logvar + (x - mean) ** 2. / torch.exp(logvar)) * mask


def normal_kl(mu1, lv1, mu2, lv2):
    v1 = torch.exp(lv1)
    v2 = torch.exp(lv2)
    lstd1 = lv1 / 2.
    lstd2 = lv2 / 2.

    kl = lstd2 - lstd1 + ((v1 + (mu1 - mu2) ** 2.) / (2. * v2)) - .5
    return kl


def mean_squared_error(orig, pred, mask):
    error = (orig - pred) ** 2
    error = error * mask
    return error.sum() / mask.sum()


def compute_losses(dim, dec_train_batch, qz0_mean, qz0_logvar, pred_x, args, device):
    observed_data, observed_mask \
        = dec_train_batch[:, :, :dim], dec_train_batch[:, :, dim:2*dim]

    noise_std = args.std  # default 0.1
    noise_std_ = torch.zeros(pred_x.size()).to(device) + noise_std
    noise_logvar = 2. * torch.log(noise_std_).to(device)
    logpx = log_normal_pdf(observed_data, pred_x, noise_logvar,
                           observed_mask).sum(-1).sum(-1)
    pz0_mean = pz0_logvar = torch.zeros(qz0_mean.size()).to(device)
    analytic_kl = normal_kl(qz0_mean, qz0_logvar,
                            pz0_mean, pz0_logvar).sum(-1).sum(-1)
    if args.norm:
        logpx /= observed_mask.sum(-1).sum(-1)
        analytic_kl /= observed_mask.sum(-1).sum(-1)
    return logpx, analytic_kl


def evaluate_classifier(model, query, aug, dec, kl_coef, test_loader, args=None, classifier=None,
                        dim=41, device='cuda', reconst=False, num_sample=1):
    pred = []
    true = []
    test_loss = 0
    hidden_dim = args.sethidden
    for test_batch, label in test_loader:
        test_batch, label = test_batch.to(device), label.to(device)
        batch_len = test_batch.shape[0]
        observed_data, observed_mask, observed_tp \
            = test_batch[:, :, :dim], test_batch[:, :, dim:2*dim], test_batch[:, :, -1]
        with torch.no_grad():
            x_aug= aug(observed_tp*hidden_dim, torch.cat((observed_data, observed_mask), 2))
            ref = query(observed_tp*hidden_dim, torch.cat((observed_data, observed_mask), 2))
            out = model(x_aug, ref)

            if reconst:
                qz0_mean, qz0_logvar = out[:, :,
                                           :args.latent_dim], out[:, :, args.latent_dim:]
                epsilon = torch.randn(
                    num_sample, qz0_mean.shape[0], qz0_mean.shape[1], qz0_mean.shape[2]).to(device)
                z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean
                z0 = z0.view(-1, qz0_mean.shape[1], qz0_mean.shape[2])
                # print(f"z0: {z0.shape}, out: {out.shape}, observed_data: {observed_data.shape}, observed_tp: {observed_tp.shape}, pred_y: {pred_y.shape}")
                if args.classify_pertp:
                    pred_x = dec(z0, observed_tp[None, :, :].repeat(
                        num_sample, 1, 1).view(-1, observed_tp.shape[1]))
                    #pred_x = pred_x.view(num_sample, batch_len, pred_x.shape[1], pred_x.shape[2])
                    out = classifier(pred_x)
                else:
                    out = classifier(z0)
            if args.classify_pertp:
                N = label.size(-1)
                out = out.view(-1, N)
                label = label.view(-1, N)
                _, label = label.max(-1)
                test_loss += nn.CrossEntropyLoss()(out, label.long()).item() * batch_len * 50.
            else:
                label = label.unsqueeze(0).repeat_interleave(
                    num_sample, 0).view(-1)
                test_loss += nn.CrossEntropyLoss()(out, label).item() *  batch_len * num_sample
        pred.append(out.cpu().numpy())
        true.append(label.cpu().numpy())
    pred = np.concatenate(pred, 0)
    true = np.concatenate(true, 0)
    acc = np.mean(pred.argmax(1) == true)
    auc = metrics.roc_auc_score(
        true, pred[:, 1]) if not args.classify_pertp else 0.
    return test_loss/pred.shape[0], acc, auc


def normalize_masked_data(data, mask, att_min, att_max):
    # we don't want to divide by zero
    att_max[att_max == 0.] = 1.

    if (att_max != 0.).all():
        data_norm = (data - att_min) / att_max
    else:
        raise Exception("Zero!")

    if torch.isnan(data_norm).any():
        raise Exception("nans!")

    # set masked out elements back to zero
    data_norm[mask == 0] = 0

    return data_norm, att_min, att_max


def get_physionet_data(args, device, q, flag=1):
    train_dataset_obj = PhysioNet('data/physionet', train=True,
                                  quantization=q,
                                  download=True, n_samples=min(10000, args.n),
                                  device=device)
    # Use custom collate_fn to combine samples with arbitrary time observations.
    # Returns the dataset along with mask and time steps
    test_dataset_obj = PhysioNet('data/physionet', train=False,
                                 quantization=q,
                                 download=True, n_samples=min(10000, args.n),
                                 device=device)

    # Combine and shuffle samples from physionet Train and physionet Test
    total_dataset = train_dataset_obj[:len(train_dataset_obj)]

    if not args.classif:
        # Concatenate samples from original Train and Test sets
        # Only 'training' physionet samples are have labels.
        # Therefore, if we do classifiction task, we don't need physionet 'test' samples.
        total_dataset = total_dataset + \
            test_dataset_obj[:len(test_dataset_obj)]
    print(len(total_dataset)) 
    # Shuffle and split
    train_data, test_data = model_selection.train_test_split(total_dataset, train_size=0.8,
                                                             random_state=42, shuffle=True)

    record_id, tt, vals, mask, labels = train_data[0]

    # n_samples = len(total_dataset) 특성 숫자/텐서의 마지막 차원
    input_dim = vals.size(-1)
    data_min, data_max = get_data_min_max(total_dataset, device)
    batch_size = min(min(len(train_dataset_obj), args.batch_size), args.n)
    if flag:
        test_data_combined = variable_time_collate_fn(test_data, device, classify=args.classif,
                                                      data_min=data_min, data_max=data_max)

        if args.classif:
            train_data, val_data = model_selection.train_test_split(train_data, train_size=0.8,
                                                                    random_state=11, shuffle=True)
            train_data_combined = variable_time_collate_fn(
                train_data, device, classify=args.classif, data_min=data_min, data_max=data_max)
            val_data_combined = variable_time_collate_fn(
                val_data, device, classify=args.classif, data_min=data_min, data_max=data_max)
            num_tp = train_data_combined[0].size(1)

            print(train_data_combined[1].sum(
            ), val_data_combined[1].sum(), test_data_combined[1].sum())
            print(train_data_combined[0].size(), train_data_combined[1].size(),
                  val_data_combined[0].size(), val_data_combined[1].size(),
                  test_data_combined[0].size(), test_data_combined[1].size())

            train_data_combined = TensorDataset(
                train_data_combined[0], train_data_combined[1].long().squeeze())
            val_data_combined = TensorDataset(
                val_data_combined[0], val_data_combined[1].long().squeeze())
            test_data_combined = TensorDataset(
                test_data_combined[0], test_data_combined[1].long().squeeze())
        else:
            train_data_combined = variable_time_collate_fn(
                train_data, device, classify=args.classif, data_min=data_min, data_max=data_max)
            print(train_data_combined.size(), test_data_combined.size())

        train_dataloader = DataLoader(
            train_data_combined, batch_size=batch_size, shuffle=False)
        test_dataloader = DataLoader(
            test_data_combined, batch_size=batch_size, shuffle=False)

    else:
        # train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=False,
        #                               collate_fn=lambda batch: variable_time_collate_fn2(batch, args, device, data_type="train",
        #                                                                                  data_min=data_min, data_max=data_max))
        # test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False,
        #                              collate_fn=lambda batch: variable_time_collate_fn2(batch, args, device, data_type="test",
        #                                                                                 data_min=data_min, data_max=data_max))
        print("improper case -- no fn2")
    attr_names = train_dataset_obj.params
    data_objects = {"dataset_obj": train_dataset_obj,
                    "train_dataloader": train_dataloader,
                    "test_dataloader": test_dataloader,
                    "input_dim": input_dim,
                    "n_train_batches": len(train_dataloader),
                    "n_test_batches": len(test_dataloader),
                    "attr": attr_names,  # optional
                    "classif_per_tp": False,  # optional
                    "num_tp" : num_tp,
                    "n_labels": 1}  # optional
    if args.classif:
        val_dataloader = DataLoader(
            val_data_combined, batch_size=batch_size, shuffle=False)
        data_objects["val_dataloader"] = val_dataloader
    return data_objects

# 시간, 값들을 모두 0~1로 정규화
def variable_time_collate_fn(batch, device=torch.device("cpu"), classify=False, activity=False,
                             data_min=None, data_max=None):
    """
    Expects a batch of time series data in the form of (record_id, tt, vals, mask, labels) where
      - record_id is a patient id
      - tt is a 1-dimensional tensor containing T time values of observations.
      - vals is a (T, D) tensor containing observed values for D variables.
      - mask is a (T, D) tensor containing 1 where values were observed and 0 otherwise.
      - labels is a list of labels for the current patient, if labels are available. Otherwise None.
    Returns:
      combined_tt: The union of all time observations.
      combined_vals: (M, T, D) tensor containing the observed values.
      combined_mask: (M, T, D) tensor containing 1 where values were observed and 0 otherwise.
      M은 배치(batch)당 포함된 시계열 데이터 수를 의미
      위 세개를 연결하여 (M, T, D+D+1)형태로 반환
    """
    D = batch[0][2].shape[1]
    # number of labels
    N = batch[0][-1].shape[1] if activity else 1
    len_tt = [ex[1].size(0) for ex in batch]
    maxlen = np.max(len_tt)
    enc_combined_tt = torch.zeros([len(batch), maxlen]).to(device)
    enc_combined_vals = torch.zeros([len(batch), maxlen, D]).to(device)
    enc_combined_mask = torch.zeros([len(batch), maxlen, D]).to(device)
    if classify:
        if activity:
            combined_labels = torch.zeros([len(batch), maxlen, N]).to(device)
        else:
            combined_labels = torch.zeros([len(batch), N]).to(device)

    for b, (record_id, tt, vals, mask, labels) in enumerate(batch):
        currlen = tt.size(0)
        ## max 길이를 둬서 앞에서부터 채우고, 나머지는 0이 되게 하는 방식
        enc_combined_tt[b, :currlen] = tt.to(device)
        enc_combined_vals[b, :currlen] = vals.to(device)
        enc_combined_mask[b, :currlen] = mask.to(device)
        if classify:
            if activity:
                combined_labels[b, :currlen] = labels.to(device)
            else:
                combined_labels[b] = labels.to(device)

    if not activity:
        enc_combined_vals, _, _ = normalize_masked_data(enc_combined_vals, enc_combined_mask,
                                                        att_min=data_min, att_max=data_max)

    if torch.max(enc_combined_tt) != 0.:
        enc_combined_tt = enc_combined_tt / torch.max(enc_combined_tt)

    combined_data = torch.cat(
        (enc_combined_vals, enc_combined_mask, enc_combined_tt.unsqueeze(-1)), 2)
    if classify:
        return combined_data, combined_labels
    else:
        return combined_data