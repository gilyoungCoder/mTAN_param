#pylint: disable=E1101, E0401, E1102, W0621, W0221
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import time

from random import SystemRandom
import models
import utils

from setmodels import *

# from torch.utils.tensorboard import SummaryWriter
# experiment_id = 'classification'
# writer = SummaryWriter('runs/experiment_' + experiment_id)
# import vessl
# vessl.init()

parser = argparse.ArgumentParser()
parser.add_argument('--niters', type=int, default=2000)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--std', type=float, default=0.01)
parser.add_argument('--latent-dim', type=int, default=32)
parser.add_argument('--rec-hidden', type=int, default=32)
parser.add_argument('--gen-hidden', type=int, default=50)
parser.add_argument('--embed-time', type=int, default=128)
parser.add_argument('--save', type=int, default=1)
parser.add_argument('--enc', type=str, default='mtan_rnn')
parser.add_argument('--dec', type=str, default='mtan_rnn')
parser.add_argument('--fname', type=str, default=None)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--split', type=int, default=0)
parser.add_argument('--n', type=int, default=8000)
parser.add_argument('--batch-size', type=int, default=50)
parser.add_argument('--quantization', type=float, default=0.1, 
                    help="Quantization on the physionet dataset.")
parser.add_argument('--classif', action='store_true', 
                    help="Include binary classification loss")
parser.add_argument('--freq', type=float, default=10.)
parser.add_argument('--k-iwae', type=int, default=10)
parser.add_argument('--norm', action='store_true')
parser.add_argument('--kl', action='store_true')
parser.add_argument('--learn-emb', action='store_true')
parser.add_argument('--dataset', type=str, default='physionet')
parser.add_argument('--alpha', type=int, default=100)
parser.add_argument('--beta', type=int, default= 50)
parser.add_argument('--old-split', type=int, default=1)
parser.add_argument('--nonormalize', action='store_true')
parser.add_argument('--enc-num-heads', type=int, default=1)
parser.add_argument('--dec-num-heads', type=int, default=1)
parser.add_argument('--num-ref-points', type=int, default=128)
parser.add_argument('--classify-pertp', action='store_true')
parser.add_argument('--aug-ratio', type=int, default=2)
parser.add_argument('--drate', type=float, default=0.5)
parser.add_argument('--sethidden', type=int, default=42)


args = parser.parse_args()


if __name__ == '__main__':
    experiment_id = int(SystemRandom().random()*100000)
    print(args, experiment_id)
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')
    
    # if args.dataset == 'physionet':
    data_obj = utils.get_physionet_data(args, 'cpu', args.quantization)
    # elif args.dataset == 'mimiciii':
    #     data_obj = utils.get_mimiciii_data(args)
    
    train_loader = data_obj["train_dataloader"]
    test_loader = data_obj["test_dataloader"]
    val_loader = data_obj["val_dataloader"]
    dim = data_obj["input_dim"]
    num_tp = data_obj["num_tp"]
    print(f"num tp : {num_tp}")
    # if args.enc == 'enc_rnn3':
    #     rec = models.enc_rnn3(
    #         dim, torch.linspace(0, 1., 128), args.latent_dim, args.rec_hidden, 128, learn_emb=args.learn_emb).to(device)
    # elif args.enc == 'mtan_rnn':
    hidden_dim = args.sethidden

    rec = models.enc_mtan_rnn(
        hidden_dim, torch.linspace(0, 1., args.num_ref_points), args.latent_dim, args.rec_hidden, 
        embed_time=128, learn_emb=args.learn_emb, num_heads=args.enc_num_heads).to(device)

    # if args.dec == 'rnn3':
    #     dec = models.dec_rnn3(
    #         dim, torch.linspace(0, 1., 128), args.latent_dim, args.gen_hidden, 128, learn_emb=args.learn_emb).to(device)
    # elif args.dec == 'mtan_rnn':
    dec = models.dec_mtan_rnn(
        dim, torch.linspace(0, 1., args.num_ref_points), args.latent_dim, args.gen_hidden, 
        embed_time=128, learn_emb=args.learn_emb, num_heads=args.dec_num_heads).to(device)
        
    classifier = models.create_classifier(args.latent_dim, args.rec_hidden).to(device)

    aug = models.TimeSeriesAugmentation(dim*2+1, 256, hidden_dim, num_outputs=args.aug_ratio*num_tp).to(device)

    query = models.QueryMa(hidden_dim, 128).to(device)
    
    params = (list(rec.parameters()) + list(dec.parameters()) + list(classifier.parameters()) + list(aug.parameters()) + list(query.parameters()))
    print('parameters:', utils.count_parameters(rec), utils.count_parameters(dec), utils.count_parameters(classifier), utils.count_parameters(aug), utils.count_parameters(query))
    optimizer = optim.Adam(params, lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    if args.fname is not None:
        checkpoint = torch.load(args.fname)
        rec.load_state_dict(checkpoint['rec_state_dict'])
        dec.load_state_dict(checkpoint['dec_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print('loading saved weights', checkpoint['epoch'])

    best_val_loss = float('inf')
    total_time = 0.
    # beta = 0
    for itr in range(1, args.niters + 1):
        train_recon_loss, train_ce_loss, train_reg_loss = 0, 0, 0
        mse = 0
        train_n = 0
        train_acc = 0
        #avg_reconst, avg_kl, mse = 0, 0, 0
        if args.kl:
            wait_until_kl_inc = 10
            if itr < wait_until_kl_inc:
                kl_coef = 0.
            else:
                kl_coef = (1-0.99** (itr - wait_until_kl_inc))
        else:
            kl_coef = 1
        start_time = time.time()
        for train_batch, label in train_loader:
            train_batch, label = train_batch.to(device), label.to(device)
            batch_len  = train_batch.shape[0]
            ## data augmentation
            observed_data, observed_mask, observed_tp \
                = train_batch[:, :, :dim], train_batch[:, :, dim:2*dim], train_batch[:, :, -1]
            
            output_aug = aug(observed_tp, torch.cat((observed_data, observed_mask), 2))
            ref = query(output_aug)
            # x_aug_copy = x_aug.clone()

            # val = torch.where(mask == 1, x_aug, torch.zeros_like(x_aug))
            
            # reg_loss = utils.diversity_regularization(tp_aug, drate = args.drate)

            reg_loss = utils.efficient_spread_regularization_loss(output_aug) + utils.efficient_spread_regularization_loss(ref)

            out = rec(output_aug, ref)

            # x_aug, time_steps = aug(observed_tp, torch.cat((observed_data, observed_mask), 2))
        
            # # x_copy = x_aug.clone()
            
            # # x_aug_copy = x_aug.clone()
            # mask = torch.where(
            #     x_aug[:, :, dim:2*dim] < 0.5,  # 조건
            #     torch.zeros_like(x_aug[:, :, dim:2*dim]),  # 조건이 True일 때 적용할 값
            #     torch.ones_like(x_aug[:, :, dim:2*dim])  # 조건이 False일 때 적용할 값
            # )          

            # data = x_aug[:, :, :dim]
            # reg_loss = utils.diversity_regularization(time_steps, drate = 0.5)
            # val = torch.where(mask == 1, x_aug, torch.zeros_like(x_aug))
            
            # if random.random() < 0.01:
            #     print("ref ", ref[0])
            #     print("out ", output_aug[0])

            #     # print(f"alpha : {self.alpha}")
            #     # print(f"original tt : {combined_x[0, :, -1]}")
            #     print(f"mask_raw: {x_copy[0, :, self.dim:2*self.dim]}")
            #     print(f"mask : {mask.shape, mask[0]}")
                # print(f"augemented time : {tp_aug.shape, tp_aug[0]}")
            #     print(f"val : {val.shape, val[0, :, :self.dim]}")
            
            # out = rec(torch.cat((augmented_data, augmented_mask), 2), augmented_tp)
            # qz0_mean, qz0_logvar = out[:, :, :args.latent_dim], out[:, :, args.latent_dim:]
            # epsilon = torch.randn(args.k_iwae, qz0_mean.shape[0], qz0_mean.shape[1], qz0_mean.shape[2]).to(device)
            # z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean
            # z0 = z0.view(-1, qz0_mean.shape[1], qz0_mean.shape[2])
            # pred_y = classifier(z0)
            
            # out = rec(torch.cat((data, mask), 2), time_steps)
            qz0_mean, qz0_logvar = out[:, :, :args.latent_dim], out[:, :, args.latent_dim:]
            epsilon = torch.randn(args.k_iwae, qz0_mean.shape[0], qz0_mean.shape[1], qz0_mean.shape[2]).to(device)
            z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean
            z0 = z0.view(-1, qz0_mean.shape[1], qz0_mean.shape[2])
            pred_y = classifier(z0)
            
            
            # print(f"z0: {z0.shape}, out: {out.shape}, observed_data: {observed_data.shape}, observed_tp: {observed_tp.shape}, pred_y: {pred_y.shape}")
            pred_x = dec(
                z0, observed_tp[None, :, :].repeat(args.k_iwae, 1, 1).view(-1, observed_tp.shape[1]))
            pred_x = pred_x.view(args.k_iwae, batch_len, pred_x.shape[1], pred_x.shape[2]) #nsample, batch, seqlen, dim
            # z0: torch.Size([50(batch), 128(rftp), 20(ldim)]), out: torch.Size([50, 128, 40]), observed_data: torch.Size([50, 203, 41]), observed_tp: torch.Size([50, 203]), pred_y: torch.Size([50, 2])
            # compute loss
            logpx, analytic_kl = utils.compute_losses(
                dim, train_batch, qz0_mean, qz0_logvar, pred_x, args, device)
            recon_loss = -(torch.logsumexp(logpx - kl_coef * analytic_kl, dim=0).mean(0) - np.log(args.k_iwae))
            label = label.unsqueeze(0).repeat_interleave(args.k_iwae, 0).view(-1)
            ce_loss = criterion(pred_y, label)
            loss = recon_loss + args.alpha*ce_loss + args.beta*reg_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_ce_loss += ce_loss.item() * batch_len
            train_recon_loss += recon_loss.item() * batch_len
            train_reg_loss += reg_loss.item() * batch_len
            train_acc += (pred_y.argmax(1) == label).sum().item()/args.k_iwae
            train_n += batch_len
            mse += utils.mean_squared_error(observed_data, pred_x.mean(0), 
                                      observed_mask) * batch_len
            
        total_time += time.time() - start_time
        val_loss, val_acc, val_auc = utils.evaluate_classifier(
            rec, query, aug, dec, kl_coef, val_loader, args=args, classifier=classifier, reconst=True, num_sample=1, dim=dim)
        # vessl.log(step = itr, payload ={'Loss/Val': val_loss,
        #                                         'Accuracy/Val': val_acc,
        #                                         'AUC/Val': val_auc})
        if val_loss <= best_val_loss:
            best_val_loss = min(best_val_loss, val_loss)
            rec_state_dict = rec.state_dict()
            dec_state_dict = dec.state_dict()
            classifier_state_dict = classifier.state_dict()
            optimizer_state_dict = optimizer.state_dict()
        test_loss, test_acc, test_auc = utils.evaluate_classifier(
            rec, query, aug, dec, kl_coef, test_loader, args=args, classifier=classifier, reconst=True, num_sample=1, dim=dim)
        cur_reg_loss = args.beta*train_reg_loss/train_n
        print('Iter: {}, recon_loss: {:.4f}, ce_loss: {:.4f}, reg_loss: {:.4f}, acc: {:.4f}, mse: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}, test_acc: {:.4f}, test_auc: {:.4f}'
              .format(itr, train_recon_loss/train_n, args.alpha*train_ce_loss/train_n, cur_reg_loss,
                      train_acc/train_n, mse/train_n, val_loss, val_acc, test_acc, test_auc))
        
        
        if best_val_loss * 1.08 < val_loss:
            print("early stop")
            break

        
            
        # if itr % 100 == 0 and args.save:
        #     torch.save({
        #         'args': args,
        #         'epoch': itr,
        #         'rec_state_dict': rec_state_dict,
        #         'dec_state_dict': dec_state_dict,
        #         'optimizer_state_dict': optimizer_state_dict,
        #         'classifier_state_dict': classifier_state_dict,
        #         'loss': -loss,
        #     }, args.dataset + '_' + 
        #         args.enc + '_' + 
        #         args.dec + '_' + 
        #         str(experiment_id) +
        #         '.h5')

    print(best_val_loss)
    print(total_time)
