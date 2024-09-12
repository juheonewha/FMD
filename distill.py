import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils
from tqdm import tqdm
from utils_clean import get_dataset, get_network, get_eval_pool, evaluate_synset, get_time, DiffAugment, ParamDiffAug
import wandb
import copy
import random
from reparam_module import ReparamModule
from sample import get_strategy
from data import Data
import matplotlib.pyplot as plt
import scipy.stats as st
import math
import warnings
from sklearn.manifold import TSNE
from torch.utils.data import Dataset

warnings.filterwarnings("ignore", category=DeprecationWarning)

def manual_seed(seed=0):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)

def euclidean_dist(x, y):
        m, n = x.size(0), y.size(0)
        xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
        yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
        dist = xx + yy
        dist.addmm_(1, -2, x, y.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        return dist

def main(args):
    manual_seed(0)
    if args.zca and args.texture:
        raise AssertionError("Cannot use zca and texture together")

    if args.texture and args.pix_init == "real":
        print("WARNING: Using texture with real initialization will take a very long time to smooth out the boundaries between images.")

    if args.max_experts is not None and args.max_files is not None:
        args.total_experts = args.max_experts * args.max_files

    print("CUDNN STATUS: {}".format(torch.backends.cudnn.enabled))

    args.dsa = True if args.dsa == 'True' else False
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    eval_it_pool = np.arange(0, args.Iteration + 1, args.eval_it).tolist()
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, loader_train_dict, class_map, class_map_inv = get_dataset(args.dataset, args.data_path, args.batch_real, args.subset, args=args)
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)

    im_res = im_size[0]

    args.im_size = im_size

    accs_all_exps = dict() 
    for key in model_eval_pool:
        accs_all_exps[key] = []

    data_save = []

    if args.dsa:
      
        args.dc_aug_param = None

    args.dsa_param = ParamDiffAug()

    dsa_params = args.dsa_param
    if args.zca:
        zca_trans = args.zca_trans
    else:
        zca_trans = None

    wandb.init(sync_tensorboard=False,
               project="DatasetDistillation",
               job_type="CleanRepo",
               config=args,
               )

    args = type('', (), {})()

    for key in wandb.config._items:
        setattr(args, key, wandb.config._items[key])

    args.dsa_param = dsa_params
    args.zca_trans = zca_trans

    if args.batch_syn is None:
        args.batch_syn = num_classes * args.ipc

    args.distributed = torch.cuda.device_count() > 1


    print('Hyper-parameters: \n', args.__dict__)
    print('Evaluation model pool: ', model_eval_pool)

    ''' organize the real dataset '''
    images_all = []
    labels_all = []
    indices_class = [[] for c in range(num_classes)]
    print("BUILDING DATASET")
    for i in tqdm(range(len(dst_train))):
        sample = dst_train[i]
        images_all.append(torch.unsqueeze(sample[0], dim=0))
        labels_all.append(class_map[torch.tensor(sample[1]).item()])

    for i, lab in tqdm(enumerate(labels_all)):
        indices_class[lab].append(i)
    images_all = torch.cat(images_all, dim=0).to("cpu")
    labels_all = torch.tensor(labels_all, dtype=torch.long, device="cpu")

    for c in range(num_classes):
        print('class c = %d: %d real images'%(c, len(indices_class[c])))

    for ch in range(channel):
        print('real images channel %d, mean = %.4f, std = %.4f'%(ch, torch.mean(images_all[:, ch]), torch.std(images_all[:, ch])))

    def get_init_images(c,n):
        query_idxs= strategy_init.query(c,n)
        return query_idxs

    def get_images(c, n):
    
        idx_shuffle = np.random.permutation(indices_class[c])[:n]
        return idx_shuffle
    
    dataset = Data(images_all, labels_all)
    
    model = get_network(args.model, channel, num_classes, im_size,dist=False)

    strategy_init = get_strategy('KMeansSampling')(dataset, model)

    ''' initialize the synthetic data '''
    label_syn = torch.tensor([np.ones(args.ipc,dtype=np.int_)*i for i in range(num_classes)], dtype=torch.long, requires_grad=False, device=args.device).view(-1)

    if args.texture:
        image_syn_1 = torch.randn(size=(num_classes * args.ipc, channel, im_size[0]*args.canvas_size, im_size[1]*args.canvas_size), dtype=torch.float)
    else:
        image_syn_1 = torch.randn(size=(num_classes * args.ipc, channel, im_size[0], im_size[1]), dtype=torch.float)
        image_syn_2 = torch.randn(size=(num_classes * args.ipc, channel, im_size[0], im_size[1]), dtype=torch.float)
        image_embed = torch.randn(size=(num_classes * args.ipc, channel, im_size[0], im_size[1]), dtype=torch.float)
        img_real =  torch.randn(size=(num_classes * args.batch_real, channel, im_size[0], im_size[1]), dtype=torch.float)

    real_image=[]
    
    ''' select real subset '''
    class CustomDataset(Dataset):
        def __init__(self,data):
            self.data = data 
            self.labels = [x // args.batch_real for x in range(args.batch_real*num_classes)] 

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            image = self.data[idx]
            label = self.labels[idx]
            return image, label

    while True:
        for c in range(num_classes):
            idx_real = get_images(c, args.batch_real)
            img_real.data[c * args.batch_real:(c + 1) * args.batch_real] = images_all[idx_real].detach().data
        
        img_real=img_real.detach().to(args.device)
        subset=CustomDataset(img_real)
        sub_train=torch.utils.data.DataLoader(subset,batch_size=64,shuffle=True)
        sub_model=get_network(args.model, channel, num_classes, im_size).to(args.device)
        sub_model.load_state_dict(torch.load(args.ckpt_dir))
        
        acc_avg=0.0
        num_exp=0.0    
        for i,data in enumerate(sub_train):
            img=data[0].float().cuda()
            lab=data[1].long().cuda()
            with torch.no_grad():
                output=sub_model(img)
            acc = np.sum(np.equal(np.argmax(output.cpu().data.numpy(), axis=-1), lab.cpu().data.numpy()))
            acc_avg += acc
            n_b = lab.shape[0]
            num_exp += n_b
        acc_avg /= num_exp
        if acc_avg > args.acc:
            break

    '''initialize synthetic learning rate'''
    syn_lr_1 = torch.tensor(args.lr_teacher).to(args.device)
    syn_lr_2 = torch.tensor(args.lr_teacher).to(args.device)
    idx_list=[[1285, 2780, 2891, 1383, 1878, 1965, 1923, 2640, 2551, 1189],[ 6635, 10647,  8536,  3299,  7938,  7881,  4702,  9457,  3416,  8354],
              [15762, 14327, 13184, 13928, 12291, 14211, 13087, 13592, 12598, 15424],[16589, 16841, 16863, 16694, 16346, 16622, 16438, 16773, 16548, 15930]]

    if args.pix_init == 'real':
        print('initialize synthetic data from random real images')
        if args.texture:
            for c in range(num_classes):
                for i in range(args.canvas_size):
                    for j in range(args.canvas_size):
                        image_syn_1.data[c * args.ipc:(c + 1) * args.ipc, :, i * im_size[0]:(i + 1) * im_size[0],
                        j * im_size[1]:(j + 1) * im_size[1]] = torch.cat(
                            [get_images(c, 1).detach().data for s in range(args.ipc)])
        for c in range(num_classes):
            idx_shuffle=get_init_images(c,args.ipc)
            #idx_shuffle=torch.tensor(np.array(idx_list[c]))
            image_syn_1.data[c * args.ipc:(c + 1) * args.ipc] = images_all[idx_shuffle].detach().data
            image_syn_2.data[c * args.ipc:(c + 1) * args.ipc] = images_all[idx_shuffle].detach().data
            image_embed.data[c * args.ipc:(c + 1) * args.ipc] = images_all[idx_shuffle].detach().data

    else:
        print('initialize synthetic data from random noise')

   
    ''' training '''
    image_syn_1 = image_syn_1.detach().to(args.device).requires_grad_(True)
    image_syn_2 = image_syn_2.detach().to(args.device).requires_grad_(True)
    image_embed=image_embed.detach().to(args.device).requires_grad_(True)
    syn_lr_1 = syn_lr_1.detach().to(args.device).requires_grad_(True)
    syn_lr_2 = syn_lr_2.detach().to(args.device).requires_grad_(True)

    optimizer_embed=torch.optim.SGD([image_embed], lr=args.lr_feat, momentum=0.5)
    optimizer_img_1 = torch.optim.SGD([image_syn_1], lr=args.lr_img_1, momentum=0.5)
    optimizer_img_2 = torch.optim.SGD([image_syn_2], lr=args.lr_img_2, momentum=0.5)
    optimizer_lr_1 = torch.optim.SGD([syn_lr_1], lr=args.lr_lr, momentum=0.5)
    optimizer_lr_2 = torch.optim.SGD([syn_lr_2], lr=args.lr_lr, momentum=0.5)
    optimizer_embed.zero_grad()
    optimizer_img_1.zero_grad()
    optimizer_img_2.zero_grad()
    optimizer_lr_1.zero_grad()
    optimizer_lr_2.zero_grad()

    criterion = nn.CrossEntropyLoss().to(args.device)
    print('%s training begins'%get_time())

    expert_dir = os.path.join(args.buffer_path, args.dataset)
    if args.dataset == "ImageNet":
        expert_dir = os.path.join(expert_dir, args.subset, str(args.res))
    if args.dataset in ["CIFAR10", "CIFAR100"] and not args.zca:
        expert_dir += "_NO_ZCA"
    expert_dir = os.path.join(expert_dir, args.model)
    print("Expert Dir: {}".format(expert_dir))

    if args.load_all:
        buffer = []
        n = 1
        while os.path.exists(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n))):
            buffer = buffer + torch.load(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n)))
            n += 1
        if n == 1:
            raise AssertionError("No buffers detected at {}".format(expert_dir))

    else:
        expert_files = []
        n = 0
        while os.path.exists(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n))):
            expert_files.append(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n)))
            n += 1
        if n == 0:
            raise AssertionError("No buffers detected at {}".format(expert_dir))
        file_idx = 0
        expert_idx = 0
        random.shuffle(expert_files)
        if args.max_files is not None:
            expert_files = expert_files[:args.max_files]
        print("loading file {}".format(expert_files[file_idx]))
        buffer = torch.load(expert_files[file_idx])
        if args.max_experts is not None:
            buffer = buffer[:args.max_experts]
        random.shuffle(buffer)

    best_acc = {m: 0 for m in model_eval_pool}

    best_std = {m: 0 for m in model_eval_pool}

    '''omega according to parameter intervals'''
    q_list=[]
    for i in range(len(buffer)):
        weight_list=[]
        for j in range(args.max_start_epoch):
            starting_buffer= torch.cat([p.data.to(args.device).reshape(-1) for p in buffer[i][j]], 0)
            targeting_buffer=torch.cat([p.data.to(args.device).reshape(-1) for p in buffer[i][j+args.expert_epochs]], 0)
            weight_pertub_buffer = torch.norm(targeting_buffer-starting_buffer,2)
            weight_list.append(weight_pertub_buffer.detach().cpu())    
        
        df = len(weight_list) - 1
        weight_list=np.array(weight_list)
        mu = np.mean(weight_list)
        st_d=np.std(weight_list)
        q1=np.percentile(weight_list,q=25)
        q2=np.percentile(weight_list,q=75)
        
        q_list.append([q1,q2])
 
    '''real feature'''
    embed_model=get_network(args.model, channel, num_classes, im_size).to(args.device)
    embed_model.load_state_dict(torch.load(args.ckpt_dir))
  
    for param in list(embed_model.parameters()):
        param.requires_grad = False
    
    img_real=img_real.to(args.device)
    out_real=embed_model.module.embed(img_real).detach()
    
    '''distillation step begins'''
    for it in range(0, args.Iteration+1):
        save_this_it = False
  
        image=args.alpha*image_syn_1+args.beta*image_syn_2+args.gamma*image_embed
  
      

        wandb.log({"Progress": it}, step=it)
        ''' Evaluate synthetic data '''
        if it in eval_it_pool:
            for model_eval in model_eval_pool:
                print('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d'%(args.model, model_eval, it))
                if args.dsa:
                    print('DSA augmentation strategy: \n', args.dsa_strategy)
                    print('DSA augmentation parameters: \n', args.dsa_param.__dict__)
                else:
                    print('DC augmentation parameters: \n', args.dc_aug_param)

                accs_test = []
                accs_train = []
                for it_eval in range(args.num_eval):
                    net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device) # get a random model

                    eval_labs = label_syn
                    with torch.no_grad():
                        image_save = image
                    image_syn_eval, label_syn_eval = copy.deepcopy(image_save.detach()), copy.deepcopy(eval_labs.detach()) # avoid any unaware modification

                    args.lr_net = syn_lr.item()
                    _, acc_train, acc_test = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval, testloader, args, texture=args.texture)
                    accs_test.append(acc_test)
                    accs_train.append(acc_train)
                accs_test = np.array(accs_test)
                accs_train = np.array(accs_train)
                acc_test_mean = np.mean(accs_test)
                acc_test_std = np.std(accs_test)
                if acc_test_mean > best_acc[model_eval]:
                    best_acc[model_eval] = acc_test_mean
                    best_std[model_eval] = acc_test_std
                    save_this_it = True
                print('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------'%(len(accs_test), model_eval, acc_test_mean, acc_test_std))
                wandb.log({'Accuracy/{}'.format(model_eval): acc_test_mean}, step=it)
                wandb.log({'Max_Accuracy/{}'.format(model_eval): best_acc[model_eval]}, step=it)
                wandb.log({'Std/{}'.format(model_eval): acc_test_std}, step=it)
                wandb.log({'Max_Std/{}'.format(model_eval): best_std[model_eval]}, step=it)


        if it in eval_it_pool and (save_this_it or it % 1000 == 0):
            with torch.no_grad():
                image_save = image.cuda()

                save_dir = os.path.join(".", "logged_files", args.dataset, wandb.run.name)

                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                torch.save(image_save.cpu(), os.path.join(save_dir, "images_{}.pt".format(it)))
                torch.save(label_syn.cpu(), os.path.join(save_dir, "labels_{}.pt".format(it)))

                if save_this_it:
                    torch.save(image_save.cpu(), os.path.join(save_dir, "images_best.pt".format(it)))
                    torch.save(label_syn.cpu(), os.path.join(save_dir, "labels_best.pt".format(it)))

                wandb.log({"Pixels": wandb.Histogram(torch.nan_to_num(image.detach().cpu()))}, step=it)

                if args.ipc < 50 or args.force_save:
                    upsampled = image_save
                    if args.dataset != "ImageNet":
                        upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
                        upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
                    grid = torchvision.utils.make_grid(upsampled, nrow=10, normalize=True, scale_each=True)
                    wandb.log({"Synthetic_Images": wandb.Image(torch.nan_to_num(grid.detach().cpu()))}, step=it)
                    wandb.log({'Synthetic_Pixels': wandb.Histogram(torch.nan_to_num(image_save.detach().cpu()))}, step=it)

                    for clip_val in [2.5]:
                        std = torch.std(image_save)
                        mean = torch.mean(image_save)
                        upsampled = torch.clip(image_save, min=mean-clip_val*std, max=mean+clip_val*std)
                        if args.dataset != "ImageNet":
                            upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
                            upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
                        grid = torchvision.utils.make_grid(upsampled, nrow=10, normalize=True, scale_each=True)
                        wandb.log({"Clipped_Synthetic_Images/std_{}".format(clip_val): wandb.Image(torch.nan_to_num(grid.detach().cpu()))}, step=it)

                    if args.zca:
                        image_save = image_save.to(args.device)
                        image_save = args.zca_trans.inverse_transform(image_save)
                        image_save.cpu()

                        torch.save(image_save.cpu(), os.path.join(save_dir, "images_zca_{}.pt".format(it)))

                        upsampled = image_save
                        if args.dataset != "ImageNet":
                            upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
                            upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
                        grid = torchvision.utils.make_grid(upsampled, nrow=10, normalize=True, scale_each=True)
                        wandb.log({"Reconstructed_Images": wandb.Image(torch.nan_to_num(grid.detach().cpu()))}, step=it)
                        wandb.log({'Reconstructed_Pixels': wandb.Histogram(torch.nan_to_num(image_save.detach().cpu()))}, step=it)

                        for clip_val in [2.5]:
                            std = torch.std(image_save)
                            mean = torch.mean(image_save)
                            upsampled = torch.clip(image_save, min=mean - clip_val * std, max=mean + clip_val * std)
                            if args.dataset != "ImageNet":
                                upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
                                upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
                            grid = torchvision.utils.make_grid(upsampled, nrow=10, normalize=True, scale_each=True)
                            wandb.log({"Clipped_Reconstructed_Images/std_{}".format(clip_val): wandb.Image(
                                torch.nan_to_num(grid.detach().cpu()))}, step=it)

        wandb.log({"Synthetic_LR": syn_lr_1.detach().cpu()}, step=it)

        student_net = get_network(args.model, channel, num_classes, im_size, dist=False).to(args.device) 
       
        student_net = ReparamModule(student_net)

        if args.distributed:
            student_net = torch.nn.DataParallel(student_net)

        student_net.train()

        num_params = sum([np.prod(p.size()) for p in (student_net.parameters())])

        if args.load_all:
            buffer_index=np.random.randint(0, len(buffer))
            expert_trajectory = buffer[buffer_index]
        else:
            expert_trajectory = buffer[expert_idx]
            expert_idx += 1
            if expert_idx == len(buffer):
                expert_idx = 0
                file_idx += 1
                if file_idx == len(expert_files):
                    file_idx = 0
                    random.shuffle(expert_files)
                print("loading file {}".format(expert_files[file_idx]))
                if args.max_files != 1:
                    del buffer
                    buffer = torch.load(expert_files[file_idx])
                if args.max_experts is not None:
                    buffer = buffer[:args.max_experts]
                random.shuffle(buffer)
        
        start_epoch = np.random.randint(0, args.max_start_epoch)
        starting_params = expert_trajectory[start_epoch]
        target_params = expert_trajectory[start_epoch+args.expert_epochs]
        target_params = torch.cat([p.data.to(args.device).reshape(-1) for p in target_params], 0)
        student_params = [torch.cat([p.data.to(args.device).reshape(-1) for p in starting_params], 0).requires_grad_(True)]
        
        if start_epoch != 0:
            d= torch.normal(mean=0.5,std=1.0,size=(1,len(student_params[-1]))).to(args.device)
            d=d[-1]
            d = (d / (torch.norm(d))) * student_params[-1]
            student_params[-1] = student_params[-1] + 0.1*d
            
        starting_params = torch.cat([p.data.to(args.device).reshape(-1) for p in starting_params], 0)
        weight_pertub=torch.norm(target_params-starting_params,2)

        omega =q_list[buffer_index][0]
        omega2=q_list[buffer_index][1]
       
        syn_images1 = image_syn_1
        syn_images2 = image_syn_2
        y_hat = label_syn.to(args.device)

        param_loss_list = []
        param_dist_list = []
        indices_chunks = []
     
        if weight_pertub > omega:
            for step in range(args.syn_steps):
                if not indices_chunks:
                    indices = torch.randperm(len(syn_images1))
                    indices_chunks = list(torch.split(indices, args.batch_syn))

                these_indices = indices_chunks.pop()
                x = syn_images1[these_indices]
                this_y = y_hat[these_indices]

                if args.texture:
                    x = torch.cat([torch.stack([torch.roll(im, (torch.randint(im_size[0]*args.canvas_size, (1,)), torch.randint(im_size[1]*args.canvas_size, (1,))), (1,2))[:,:im_size[0],:im_size[1]] for im in x]) for _ in range(args.canvas_samples)])
                    this_y = torch.cat([this_y for _ in range(args.canvas_samples)])

                if args.dsa and (not args.no_aug):
                    x = DiffAugment(x, args.dsa_strategy, param=args.dsa_param)

                if args.distributed:
                    forward_params = student_params[-1].unsqueeze(0).expand(torch.cuda.device_count(), -1)
                else:
                    forward_params = student_params[-1]
                x = student_net(x, flat_param=forward_params)
                ce_loss = criterion(x, this_y)

                grad = torch.autograd.grad(ce_loss, student_params[-1], create_graph=True)[0]

                student_params.append(student_params[-1] - syn_lr_1 * grad)

        else:
            for step in range(args.syn_steps):

                if not indices_chunks:
                    indices = torch.randperm(len(syn_images2))
                    indices_chunks = list(torch.split(indices, args.batch_syn))

                these_indices = indices_chunks.pop()
                x = syn_images2[these_indices]
                this_y = y_hat[these_indices]

                if args.texture:
                    x = torch.cat([torch.stack([torch.roll(im, (torch.randint(im_size[0]*args.canvas_size, (1,)), torch.randint(im_size[1]*args.canvas_size, (1,))), (1,2))[:,:im_size[0],:im_size[1]] for im in x]) for _ in range(args.canvas_samples)])
                    this_y = torch.cat([this_y for _ in range(args.canvas_samples)])

                if args.dsa and (not args.no_aug):
                    x = DiffAugment(x, args.dsa_strategy, param=args.dsa_param)

                if args.distributed:
                    forward_params = student_params[-1].unsqueeze(0).expand(torch.cuda.device_count(), -1)
                else:
                    forward_params = student_params[-1]
                x = student_net(x, flat_param=forward_params)
                ce_loss = criterion(x, this_y)

                grad = torch.autograd.grad(ce_loss, student_params[-1], create_graph=True)[0]

                student_params.append(student_params[-1] - syn_lr_2 * grad)

        
        param_loss = torch.tensor(0.0).to(args.device)
        param_dist = torch.tensor(0.0).to(args.device)

        param_loss += torch.nn.functional.mse_loss(student_params[-1], target_params, reduction="sum")
        param_dist += torch.nn.functional.mse_loss(starting_params, target_params, reduction="sum")
      
        param_loss_list.append(param_loss)
        param_dist_list.append(param_dist)


        param_loss /= num_params
        param_dist /= num_params

        param_loss /= param_dist

        grand_loss = param_loss
       
        '''feature distribution matching'''
        out_syn=embed_model.module.embed(image_embed).to(args.device)
        embedding_center_r=torch.mean(out_real.reshape(num_classes, args.batch_real, -1), dim=1)
       
        embedding_center_s=torch.mean(out_syn.reshape(num_classes, args.ipc, -1), dim=1)
        euclidean_distance_r=euclidean_dist(embedding_center_r,out_real)
        euclidean_distance_s=euclidean_dist(embedding_center_s,out_syn)  

        embedding_loss = torch.sum((torch.mean(out_real.reshape(num_classes, args.batch_real, -1), dim=1) - torch.mean(out_syn.reshape(num_classes, args.ipc, -1), dim=1))**2)
      
        distillation_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(euclidean_distance_r, dim=1),F.softmax(F.pad(euclidean_distance_s,(0, args.batch_real*num_classes -(args.ipc*num_classes)),value=0)))
        
        loss_feature = embedding_loss + distillation_loss
        optimizer_embed.zero_grad()
        loss_feature.backward(retain_graph=False)
        optimizer_embed.step()
        

        '''parameter matching'''
        if weight_pertub > omega:
            optimizer_img_1.zero_grad()
            optimizer_lr_1.zero_grad()
         
            grand_loss.backward(retain_graph=False)
         
            optimizer_img_1.step()
            optimizer_lr_1.step()
          
        else:
            optimizer_img_2.zero_grad()
          
            optimizer_lr_2.zero_grad()
           
            grand_loss.backward(retain_graph=False)
           
            optimizer_img_2.step()
            optimizer_lr_2.step()
     

        wandb.log({"Grand_Loss": grand_loss.detach().cpu(),
                   "Start_Epoch": start_epoch})
        
        for _ in student_params:
            del _

        if it%10 == 0:
            print('%s iter = %04d, loss = %.4f' % (get_time(), it, grand_loss.item()))

    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')

    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')

    parser.add_argument('--subset', type=str, default='imagenette', help='ImageNet subset. This only does anything when --dataset=ImageNet')

    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    
    parser.add_argument('--ckpt_dir', type=str, default='./covid.pth', help='checkpoint direction for selecting real dataset')

    arser.add_argument('--acc', type=float, default=0.95, help='accuracy criteria for selecting a real dataset')

    arser.add_argument('--lr_feat', type=float, default=1.0, help='learning rate for feature distribution matching')

    arser.add_argument('--alpha', type=float, default=0.6, help='learning rate for feature distribution matching')

    arser.add_argument('--beta', type=float, default=0.2, help='learning rate for feature distribution matching')

    arser.add_argument('--gamma', type=float, default=0.2, help='learning rate for feature distribution matching')

    parser.add_argument('--ipc', type=int, default=10, help='image(s) per class')

    parser.add_argument('--eval_mode', type=str, default='S',
                        help='eval_mode, check utils.py for more info')

    parser.add_argument('--num_eval', type=int, default=2, help='how many networks to evaluate on')

    parser.add_argument('--eval_it', type=int, default=100, help='how often to evaluate')

    parser.add_argument('--epoch_eval_train', type=int, default=1000, help='epochs to train a model with synthetic data')

    parser.add_argument('--Iteration', type=int, default=5000, help='how many distillation steps to perform')

    parser.add_argument('--lr_img_1', type=float, default=100, help='learning rate for updating synthetic images 1')
    
    parser.add_argument('--lr_img_2', type=float, default=100, help='learning rate for updating synthetic images 2')

    parser.add_argument('--lr_lr', type=float, default=1e-05, help='learning rate for updating... learning rate')
    parser.add_argument('--lr_teacher', type=float, default=0.001, help='initialization for synthetic learning rate')

    parser.add_argument('--lr_init', type=float, default=0.001, help='how to init lr (alpha)')

    parser.add_argument('--batch_real', type=int, default=128, help='batch size for real data')
    
    parser.add_argument('--batch_syn', type=int, default=20, help='should only use this if you run out of VRAM')
    
    parser.add_argument('--batch_train', type=int, default=64, help='batch size for training networks')

    parser.add_argument('--pix_init', type=str, default='real', choices=["noise", "real"],
                        help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')

    parser.add_argument('--dsa', type=str, default='True', choices=['True', 'False'],
                        help='whether to use differentiable Siamese augmentation.')

    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate',
                        help='differentiable Siamese augmentation strategy')

    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--buffer_path', type=str, default='./buffer_128', help='buffer path')

    parser.add_argument('--expert_epochs', type=int, default=2, help='how many expert epochs the target params are')
    parser.add_argument('--syn_steps', type=int, default=60, help='how many steps to take on synthetic data')
    parser.add_argument('--max_start_epoch', type=int, default=20, help='max epoch we can start at')

    parser.add_argument('--zca', action='store_false', help="do ZCA whitening")

    parser.add_argument('--load_all', action='store_true', help="only use if you can fit all expert trajectories into RAM")

    parser.add_argument('--no_aug', type=bool, default=False, help='this turns off diff aug during distillation')

    parser.add_argument('--texture', action='store_false', help="will distill textures instead")
    parser.add_argument('--canvas_size', type=int, default=2, help='size of synthetic canvas')
    parser.add_argument('--canvas_samples', type=int, default=1, help='number of canvas samples per iteration')


    parser.add_argument('--max_files', type=int, default=None, help='number of expert files to read (leave as None unless doing ablations)')
    parser.add_argument('--max_experts', type=int, default=None, help='number of experts to read per file (leave as None unless doing ablations)')

    parser.add_argument('--force_save', action='store_true', help='this will save images for 50ipc')

    args = parser.parse_args()

    main(args)