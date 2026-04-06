import torch
import itertools
import torch.fft
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks

class FrequencyLoss(torch.nn.Module):
    # Referecne: Frequency-Domain-Based Structure Losses for CycleGAN-Based Cone-Beam Computed Tomography Translation
    # frequency loss to train cycle-gan
    # L_gfl(x, y) = (1 / (M * N)) * Σ [ w(u,v) * | tanh(|F_shift(DFT(x))|{u,v}) - tanh(|F_shift(DFT(y))|{u,v}) | ]
    # (Latex) $$\mathcal{L}_{GFL}(x, y) = \frac{1}{M \times N} \sum_{u=0}^{M-1} \sum_{v=0}^{N-1} w(u,v) \cdot \left| \tanh\left(\left| \mathcal{F}_{shift}(\text{DFT}(x)) \right|_{u,v}\right) - \tanh\left(\left| \mathcal{F}_{shift}(\text{DFT}(y)) \right|_{u,v}\right) \right|$$
    # (Latex) $$w(u,v) = \alpha \cdot (Dist_{u,v})^\beta$$

    def __init__(self, alpha=1.0, beta=-1.0):
        super(FrequencyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta #默认低通
        self.l1_loss = torch.nn.L1Loss(reduction='none')  # 逐像素加权需设为 none

    def forward(self, x, y):
        # 1. FFT 变换并移频
        x_fft = torch.fft.fftshift(torch.fft.fftn(x, dim=(-2, -1),norm="ortho"), dim=(-2, -1))
        y_fft = torch.fft.fftshift(torch.fft.fftn(y, dim=(-2, -1),norm="ortho"), dim=(-2, -1))

        # 2. 提取幅度谱并应用 tanh 非线性映射
        # 公式参考: tanh(|FFT(x)|)
        x_mag = torch.tanh(torch.abs(x_fft))
        y_mag = torch.tanh(torch.abs(y_fft))

        # 3. 生成基于距离的软权重 w(u,v) = alpha * dist^beta
        h, w = x_mag.shape[-2:]
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, h, device=x.device),
            torch.linspace(-1, 1, w, device=x.device),
            indexing='ij'
        )
        dist = torch.sqrt(grid_x ** 2 + grid_y ** 2)

        # 软掩码权重计算
        # epsilon (1e-8) 防止中心点距离为 0 时负幂次导致 NaN
        #当 $\beta < 0$ 时（低通倾向）： 距离中心越近（$Dist$ 越小），权重 $w$ 越大；距离中心越远，$w$ 迅速衰减。
        soft_mask = self.alpha * torch.pow(dist + 1e-8, self.beta)

        # 4. 计算加权 L1 损失
        diff = self.l1_loss(x_mag, y_mag)
        weighted_loss = diff * soft_mask

        return weighted_loss.mean()


class FGCycleGANModel(BaseModel):
    """
    This class implements the FG_CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    and a least-square GANs objective ('--gan_mode lsgan').
    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """
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
            parser.add_argument("--lambda_A", type=float, default=10.0, help="weight for cycle loss (A -> B -> A)")
            parser.add_argument("--lambda_B", type=float, default=10.0, help="weight for cycle loss (B -> A -> B)")

            parser.add_argument("--lambda_GA", type=float, default=1.0, help="weight for generator loss")
            parser.add_argument("--lambda_GB", type=float, default=1.0, help="weight for generator loss")

            parser.add_argument("--lambda_FA", type=float, default=10.0, help="weight for freq cycle loss (A -> B -> A)")
            parser.add_argument("--lambda_FB", type=float, default=10.0, help="weight for freq cycle loss (B -> A -> B)")

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

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """


        BaseModel.__init__(self, opt)
        self.loss_D_A = 0.5
        self.loss_D_B = 0.5
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ["D_A", "G_A", "cycle_A", "freq_A", "idt_A", "D_B", "G_B", "cycle_B", "freq_B", "idt_B"]
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ["real_A", "fake_B", "rec_A"]
        visual_names_B = ["real_B", "fake_A", "rec_B"]
        if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_B(A)
            visual_names_A.append("idt_B")
            visual_names_B.append("idt_A")

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ["G_A", "G_B", "D_A", "D_B"]
        else:  # during test time, only load Gs
            self.model_names = ["G_A", "G_B"]

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain)

        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain)

        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert opt.input_nc == opt.output_nc
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            self.criterionFreq = FrequencyLoss().to(self.device) # zy add frequency loss

            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == "AtoB"
        self.real_A = input["A" if AtoB else "B"].to(self.device)
        self.real_B = input["B" if AtoB else "A"].to(self.device)
        self.image_paths = input["A_paths" if AtoB else "B_paths"]

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG_A(self.real_A)  # G_A(A)
        self.rec_A = self.netG_B(self.fake_B)  # G_B(G_A(A))
        self.fake_A = self.netG_B(self.real_B)  # G_B(B)
        self.rec_B = self.netG_A(self.fake_A)  # G_A(G_B(B))

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_FA = self.opt.lambda_FA
        lambda_FB = self.opt.lambda_FB
        lambda_GA = self.opt.lambda_GA
        lambda_GB = self.opt.lambda_GB
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B

        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True) * lambda_GA
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True) * lambda_GB

        # Forward cycle loss  (pixel guided) by default it will be zero
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss  (pixel guided)
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B

        # Forward loss (frequency guided)
        self.loss_freq_A = self.criterionFreq(self.fake_A, self. real_A) * lambda_FA
        # Backward loss  (frequency guided)
        self.loss_freq_B = self.criterionFreq(self.fake_B, self. real_B) * lambda_FB

        # combined loss and calculate gradients
        self.loss_G = (self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B
                       + self.loss_freq_A + self.loss_freq_B
                       + self.loss_idt_A + self.loss_idt_B)

        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()  # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()  # calculate gradients for G_A and G_B
        self.optimizer_G.step()  # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()  # set D_A and D_B's gradients to zero
        # 只有当 D 的损失不够低时才计算梯度并更新
        # avoid over powered D net
        if self.loss_D_A > 0.13:
            self.backward_D_A()
        if self.loss_D_B > 0.13:
            self.backward_D_B()
        self.optimizer_D.step()  # update D_A and D_B's weights
