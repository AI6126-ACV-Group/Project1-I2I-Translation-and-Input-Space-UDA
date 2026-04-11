import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from torch.autograd import Variable
import numpy as np
import torch.distributed as dist

## this is the part I for cy CADA introduce the sem loss
## part II is to train the src_net and adda_net


class CycleGANSemanticModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For FG_CycleGAN, in addition to cycle GAN losses, introduce frequency loss
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss (pixel):  loss_cycle_A = L1 (rec_A, real_A) (default zero)
        Backward cycle loss (pixel): loss_cycle_B = L1 (rec_A, real_A) (default zero)
        Forward cycle loss (frequency) : Loss_freq_A = FrequencyLoss(rec_A, real_A)
        Backward cycle loss (frequency) : Loss_freq_B = FrequencyLoss(rec_B, real_B)

        Loss_G = Loss_G_A + Loss_G_B + (lambda_A * Loss_cycle_A) + (lambda_B * Loss_cycle_B) + (lambda_FA * Loss_freq_A) + (lambda_FB * Loss_freq_B) + (lambda_idt * lambda_B * Loss_idt_A) + (lambda_idt * lambda_A * Loss_idt_B)
        (Latex) $$\mathcal{L}_{G} = \mathcal{L}_{GAN} + \lambda_{A}\mathcal{L}_{cyc\_A} + \lambda_{B}\mathcal{L}_{cyc\_B} + \lambda_{FA}\mathcal{L}_{freq\_A} + \lambda_{FB}\mathcal{L}_{freq\_B} + \lambda_{idt}(\lambda_{B}\mathcal{L}_{idt\_A} + \lambda_{A}\mathcal{L}_{idt\_B})$$

        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)

        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            # default setting is only use frequency cycle loss
            parser.add_argument("--D_lr_weight", type=float, default=1.0, help="weight for generator loss")

            parser.add_argument("--lambda_A", type=float, default=10.0, help="weight for cycle loss (A -> B -> A)")
            parser.add_argument("--lambda_B", type=float, default=10.0, help="weight for cycle loss (B -> A -> B)")

            parser.add_argument("--lambda_GA", type=float, default=1.0, help="weight for generator loss")
            parser.add_argument("--lambda_GB", type=float, default=1.0, help="weight for generator loss")


            parser.add_argument(
                "--lambda_identity",
                type=float,
                default=0.2,
                help="use identity mapping. "
                     "Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. "
                     "For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1"
                     "If you only want to use frequency loss, please set lambda_identity = 0.0",
            )

        return parser


    def name(self):
        return 'CycleGANModel'

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """

        BaseModel.__init__(self, opt)

        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 
                'D_B', 'G_B', 'cycle_B', 'idt_B', 
                'sem_AB', 'sem_BA', 'CLS']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        if self.isTrain and self.opt.lambda_identity > 0.0:
            visual_names_A.append('idt_A')
            visual_names_B.append('idt_B')

        self.visual_names = visual_names_A + visual_names_B
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B', 'CLS']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']

        # load/define networks
        # The naming conversion is different from those used in the paper
        # Code (paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm, not opt.no_dropout,
                                        opt.init_type, opt.init_gain)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm, not opt.no_dropout,
                                        opt.init_type, opt.init_gain)

        if self.isTrain:
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.norm, opt.init_type,
                                            opt.init_gain)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.norm, opt.init_type,
                                            opt.init_gain)
            self.netCLS = networks.define_C(opt.output_nc, opt.ndf, 
                                            init_type=opt.init_type, out_feature_num=opt.out_feature_num)
            # load trained CLS network
            if not opt.continue_train:
                print("First time training: Loading pretrained CLS weights...")
                self.load_network(self.netCLS, 'CLS', 'latest')
                print("weight loaded")
            else:
                print("Continuing training: CLS will be loaded by setup().")
 
        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert opt.input_nc == opt.output_nc

            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            self.criterionCLS = torch.nn.modules.CrossEntropyLoss()
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
                                                lr=opt.lr * opt.D_lr_weight, betas=(opt.beta1, 0.999))
            self.optimizer_CLS = torch.optim.Adam(self.netCLS.parameters(), lr=1e-3, betas=(opt.beta1, 0.999))
            self.optimizers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        if 'A_label' in input and 'B_label' in input:
            self.input_A_label = input['A_label' if AtoB else 'B_label'].to(self.device)
            self.input_B_label = input['B_label' if AtoB else 'A_label'].to(self.device)
            #self.image_paths = input['B_paths'] # Hack!! forcing the labels to corresopnd to B domain

    def load_network(self, net, net_label, epoch_label):
        """
        加载单个指定的网络权重。
        :param net: 需要加载权重的网络对象 (例如 self.netCLS)
        :param net_label: 网络的标签名称 (例如 'CLS')
        :param epoch_label: epoch 标识 (例如 'latest')
        """
        # 1. 构造文件名，例如 "latest_net_CLS.pth"
        load_filename = f"{epoch_label}_net_{net_label}.pth"
        load_path = self.save_dir / load_filename
        # 2. 如果是分布式训练，解包模型
        if isinstance(net, torch.nn.parallel.DistributedDataParallel):
            net = net.module
        print(f"loading the model from {load_path}")
        # 3. 加载权重文件
        # 注意：确保设置 weights_only=True 以符合最新的安全实践
        state_dict = torch.load(load_path, map_location=str(self.device), weights_only=True)
        if hasattr(state_dict, "_metadata"):
            del state_dict._metadata
        # 4. 修复 InstanceNorm 的兼容性问题
        for key in list(state_dict.keys()):
            self._BaseModel__patch_instance_norm_state_dict(state_dict, net, key.split("."))
        # 5. 正式加载到网络中
        net.load_state_dict(state_dict)

    def forward(self):
        self.fake_B = self.netG_A(self.real_A)
        self.rec_A = self.netG_B(self.fake_B)

        self.fake_A = self.netG_B(self.real_B)
        self.rec_B = self.netG_A(self.fake_A)

        if self.isTrain:
           # Forward all four images through classifier
           # Keep predictions from fake images only
           self.pred_real_A = self.netCLS(self.real_A)
           _,self.gt_pred_A = self.pred_real_A.max(1)
           pred_real_B = self.netCLS(self.real_B)
           _,self.gt_pred_B = pred_real_B.max(1)
           self.pred_fake_A = self.netCLS(self.fake_A)
           self.pred_fake_B = self.netCLS(self.fake_B)

           _,pfB = self.pred_fake_B.max(1)
        

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D
    
    def backward_CLS(self):
        label_A = self.input_A_label
        # forward only real source image through semantic classifier
        pred_A = self.netCLS(self.real_A) 
        self.loss_CLS = self.criterionCLS(pred_A, label_A)
        self.loss_CLS.backward()

    def backward_D_A(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_G(self):
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed.
            self.idt_A = self.netG_A(self.real_B)

            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed.
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_A = 2 * self.criterionGAN(self.netD_A(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # Forward cycle loss
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # combined loss standard cyclegan
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B

        # semantic loss AB
        self.loss_sem_AB = self.criterionCLS(self.pred_fake_B, self.input_A_label)
        #self.loss_sem_AB = self.criterionCLS(self.pred_fake_B, self.gt_pred_A)
        # semantic loss BA
        self.loss_sem_BA = self.criterionCLS(self.pred_fake_A, self.gt_pred_B)
        
        # only use semantic loss when classifier has reasonably low loss
        #if True:
        if not hasattr(self, 'loss_CLS') or self.loss_CLS.detach().item() > 1.0:
            self.loss_sem_AB = 0 * self.loss_sem_AB 
            self.loss_sem_BA = 0 * self.loss_sem_BA 
      
        self.loss_G += self.loss_sem_BA + self.loss_sem_AB
        self.loss_G.backward()

    def optimize_parameters(self):
        # forward
        self.forward()
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)
        self.set_requires_grad([self.netG_A, self.netG_B], True)
        self.optimizer_G.zero_grad()
        self.optimizer_CLS.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()
        self.backward_D_A()
        self.backward_D_B()
        self.optimizer_D.step()
        # CLS
        self.set_requires_grad([self.netD_A, self.netD_B], False)
        self.set_requires_grad([self.netCLS], True)
        self.optimizer_CLS.zero_grad()
        self.backward_CLS()
        self.optimizer_CLS.step()
