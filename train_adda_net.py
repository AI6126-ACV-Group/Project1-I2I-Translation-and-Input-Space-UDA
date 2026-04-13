from __future__ import print_function
import os
from os.path import join
import torch
import torch.optim as optim
from cycada.data_loader_from_files import load_adda_data_aligned
from cycada.util import make_variable
from models.resnet18 import ResNet18Model
from models.adda_net import AddaNet

def train(loader_src, loader_tgt, net, opt_net, opt_dis, epoch):
   
    log_interval = 100 # specifies how often to display
  
    N = min(len(loader_src.dataset), len(loader_tgt.dataset)) 
    joint_loader = zip(loader_src, loader_tgt)
      
    net.train()
   
    last_update = -1
    for batch_idx, ((data_s, _), (data_t, _)) in enumerate(joint_loader):
        # log basic adda train info
        info_str = "[Train Adda] Epoch: {} [{}/{} ({:.2f}%)]".format(
            epoch, batch_idx*len(data_t), N, 100 * batch_idx / N)
   
        ########################
        # Setup data variables #
        ########################
        data_s = make_variable(data_s, requires_grad=False)
        data_t = make_variable(data_t, requires_grad=False)
        
        ##########################
        # Optimize discriminator #
        ##########################

        # zero gradients for optimizer
        opt_dis.zero_grad()

        # extract and concat features
        score_s = net.src_net(data_s)
        score_t = net.tgt_net(data_t)
        f = torch.cat((score_s, score_t), 0)
        #print(f.shape)
        
        # predict with discriminator
        pred_concat = net.discriminator(f)

        # prepare real and fake labels: source=1, target=0
        target_dom_s = make_variable(torch.ones(len(data_s)).long(), requires_grad=False)
        target_dom_t = make_variable(torch.zeros(len(data_t)).long(), requires_grad=False)
        label_concat = torch.cat((target_dom_s, target_dom_t), 0)

        # compute loss for disciminator
        loss_dis = net.gan_criterion(pred_concat, label_concat)
        loss_dis.backward()

        # optimize discriminator
        opt_dis.step()

        # compute discriminator acc
        pred_dis = torch.squeeze(pred_concat.max(1)[1])
        acc = (pred_dis == label_concat).float().mean()
        
        # log discriminator update info
        info_str += " acc: {:0.1f} D: {:.3f}".format(acc.item()*100, loss_dis.item())

        ###########################
        # Optimize target network #
        ###########################

        # only update net if discriminator is strong
        if acc.item() > 0.6:
            
            last_update = batch_idx
        
            # zero out optimizer gradients
            opt_dis.zero_grad()
            opt_net.zero_grad()

            # --- 正确写法：必须提取 feature ---
            score_t, feat_t = net.tgt_net(data_t, with_ft=True)  # 确保返回两个值
            pred_tgt = net.discriminator(feat_t)  # 喂给判别器的是 feat_t (512维)
            # 目标网的目标是让判别器误认为这是 Source (label=1)
            label_tgt = make_variable(torch.ones(pred_tgt.size(0)).long(), requires_grad=False)
            loss_gan_t = net.gan_criterion(pred_tgt, label_tgt)
            loss_gan_t.backward()
            opt_net.step()

            # log net update info
            info_str += " G: {:.3f}".format(loss_gan_t.item()) 

        ###########
        # Logging #
        ###########
        if batch_idx % log_interval == 0:
            print(info_str)

    return last_update


def train_adda_epoch(train_loader, net, opt_net, opt_dis, epoch):
    log_interval = 10
    # AddaDataset 已经处理了长度对齐，直接获取长度
    N = len(train_loader.dataset)

    net.train()
    last_update = -1

    # 注意：这里的 train_loader 每一个 iteration 直接返回 ((data_s, label_s), (data_t, label_t))
    # 对应你 data_loader.py 中 AddaDataset 的 __getitem__ 返回值
    for batch_idx, ((data_s, _), (data_t, _)) in enumerate(train_loader):

        # 基础日志信息
        batch_size = data_t.size(0)
        info_str = "[Train Adda] Epoch: {} [{}/{} ({:.2f}%)]".format(
            epoch, batch_idx * batch_size, N, 100 * batch_idx * batch_size / N)

        # 数据转为 Variable (适配你原有的 make_variable 工具)
        data_s = make_variable(data_s, requires_grad=False)
        data_t = make_variable(data_t, requires_grad=False)

        ##########################
        # 1. 优化判别器 (Discriminator)
        ##########################
        opt_dis.zero_grad()

        # 提取特征并拼接
        score_s, feat_s = net.src_net(data_s, with_ft=True)
        score_t, feat_t = net.tgt_net(data_t, with_ft=True)
        # 2. 拼接特征 (Feature-level alignment)
        # 这里的 f 应该是 (Batch*2, 512)
        f = torch.cat((feat_s, feat_t), 0)
        #print(f.shape)
        # 3. 这时再传给判别器就不会报错了
        pred_concat = net.discriminator(f)

        # 准备域标签：source=1, target=0
        target_dom_s = make_variable(torch.ones(len(data_s)).long(), requires_grad=False)
        target_dom_t = make_variable(torch.zeros(len(data_t)).long(), requires_grad=False)
        label_concat = torch.cat((target_dom_s, target_dom_t), 0)

        loss_dis = net.gan_criterion(pred_concat, label_concat)
        loss_dis.backward()
        opt_dis.step()

        # 计算判别器准确率
        pred_dis = torch.squeeze(pred_concat.max(1)[1])
        acc = (pred_dis == label_concat).float().mean()
        info_str += " acc: {:0.1f} D: {:.3f}".format(acc.item() * 100, loss_dis.item())

        ###########################
        # 2. 优化目标网络 (Target Encoder)
        ###########################
        if acc.item() > 0.6:
            last_update = batch_idx
            opt_dis.zero_grad()
            opt_net.zero_grad()

            # 【核心修改 1】：确保提取 feature (512维)
            score_t, feat_t = net.tgt_net(data_t, with_ft=True)
            # 【核心修改 2】：将 feat_t 而不是 score_t 喂给判别器
            pred_tgt = net.discriminator(feat_t)
            # 目标网的目标是让判别器误认为这是 Source (label=1)
            label_tgt = make_variable(torch.ones(pred_tgt.size(0)).long(), requires_grad=False)
            loss_gan_t = net.gan_criterion(pred_tgt, label_tgt)
            loss_gan_t.backward()
            opt_net.step()

            info_str += " G: {:.3f}".format(loss_gan_t.item())

        if batch_idx % log_interval == 0:
            print(info_str)

    return last_update

def train_adda(src_path, tgt_path, num_cls, num_epoch=200,
        batch=128, outdir="",
        src_weights=None, lr=1e-5,lr_d=1e-4, betas=(0.9,0.999),
        weight_decay=0, outfile_name="addanet.pth"):
    """Main function for training ADDA."""

    ###########################
    # Setup cuda and networks #
    ###########################

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # setup network
    src_model = ResNet18Model(num_cls=num_cls)
    tgt_model = ResNet18Model(num_cls=num_cls)

    print(f"Loading weights from {src_weights}")
    # 1. 载入原始 pth 文件
    raw_weights = torch.load(src_weights, map_location=device)
    if 'state_dict' in raw_weights:
        raw_weights = raw_weights['state_dict']
    fixed_weights = {}
    for k, v in raw_weights.items():
        fixed_weights[f"model.{k}"] = v
    # 4. 再次检查：打印第一个 key 看看是不是变成了 model.conv1.weight
    print("First fixed key:", list(fixed_weights.keys())[0])
    src_model.load_state_dict(fixed_weights, strict=True)
    tgt_model.load_state_dict(fixed_weights, strict=True)
    print("Weights loaded successfully!")

    for param in src_model.parameters():
        param.requires_grad = False

    net = AddaNet(src_model, tgt_model, num_cls=num_cls,
                  feature_dim=512, discrim_feat=True).to(device)
    
    # print network and arguments
    #print(net)

    #######################################
    # Setup data for training and testing #
    #######################################

    train_loader = load_adda_data_aligned(
        src_dir=src_path,
        tgt_dir=tgt_path,
        batch=batch,  # 建议保持 32
        load_size=128,  # 对应你 ResNet 代码的设置
        crop_size=128,  # 对应你 ResNet 代码的设置
    )

    ######################
    # Optimization setup #
    ######################
    net_param = net.tgt_net.parameters()
    opt_net = optim.Adam(net_param, lr=lr, weight_decay=weight_decay, betas=betas)
    opt_dis = optim.Adam(net.discriminator.parameters(), lr=lr_d,
            weight_decay=weight_decay, betas=betas)
    ##############
    # Train Adda #
    ##############
    for epoch in range(num_epoch):
        err = train_adda_epoch(train_loader, net, opt_net, opt_dis, epoch)
        if err == -1:
            print("No suitable discriminator")
            break
       
    ##############
    # Save Model #
    ##############
    os.makedirs(outdir, exist_ok=True)
    outfile = join(outdir, outfile_name)
    print('Saving to', outfile)
    net.save(outfile)


if __name__ == '__main__':
    src_path= "transformed_dataset/cyCADA/amazon2webcam"
    tgt_path= "original_datasets/office_31/webcam"

    num_epoch= 20
    num_cls=31
    batch=32
    outdir= "checkpoints/Resnet_amazon2webcam_cycada"
    src_weights='checkpoints/Resnet_REAL_amazon/resnet18_amazon.pth'
    lr = 1e-6
    lr_d = 1e-5
    weight_decay = 1e-5
    outfile_name = "addanet.pth"

    train_adda(src_path, tgt_path, num_cls=num_cls, num_epoch=num_epoch,
               batch=batch, outdir=outdir,
               src_weights=src_weights, lr=lr, lr_d=lr_d,betas=(0.9, 0.999),
               weight_decay=weight_decay, outfile_name=outfile_name)

    checkpoint_path = outdir+'/'+outfile_name
    checkpoint = torch.load(checkpoint_path)
    tgt_net_state_dict = {}
    for k, v in checkpoint.items():
        if k.startswith('tgt_net.'):
            # 去掉 'tgt_net.' 这 8 个字符
            new_key = k[8:]
            tgt_net_state_dict[new_key] = v

    torch.save(tgt_net_state_dict, outdir+"/resnet18_adda_optimized.pth")
    print("Successfully extracted tgt_net weights!")

