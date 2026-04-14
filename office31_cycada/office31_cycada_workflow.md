在 Amazon -> Webcam 这个任务里，CyCADA 做的事情可以压缩成一句话：

把有标签的源域 Amazon 先“变成看起来像 Webcam 的样子”，再用这些带标签的“伪 Webcam 源域图像”训练分类器，最后再用无标签 Webcam 做特征级对齐，让模型更适应真实 Webcam。

你现在的重写版如果按这个思路做，方向上就是完全对的。

先把方向彻底理顺
这里有两个域：

源域 Amazon：有标签，31 类
目标域 Webcam：训练时不使用标签，测试时才用标签评估
因此，风格转换方向一定是 Amazon -> Webcam-style，不是反过来。
原因很简单：你最终想让分类器在 Webcam 上表现好，所以应该把“有标签的 Amazon 图像”翻译成“长得像 Webcam 的图像”，这样这些图仍然保留原类别标签，但视觉风格更接近目标域。


所以在你的任务里，各数据的角色是：

amazon/images/...：源域原图，有标签
webcam/images/...：目标域原图，训练时无标签使用
amazon2webcam/...：把 amazon 翻译成 webcam 风格后的图，仍继承 amazon 的类别标签
webcam test：最终评估数据
CyCADA 在这个任务里到底做了什么
按方法逻辑，可以拆成 4 步。

第 1 步：训练图像级适配器
用 Amazon 和 Webcam 的无配对图像训练一个 CycleGAN 风格的图像翻译模型。

训练时喂给它的是：

域 A：Amazon 全部训练图像
域 B：Webcam 全部训练图像
这一阶段不需要类别标签参与主训练。
它学到的是两个映射：

G_A2W: Amazon -> Webcam-style
G_W2A: Webcam -> Amazon-style
但对你的分类任务，真正会被用到的是 G_A2W。

你可以把这一步理解成：

输入：一张商品白底图 Amazon
输出：一张内容还是同一个物体、但外观更像摄像头拍摄条件的图 Webcam-style
第 2 步：生成翻译后的源域训练集
拿上一步训练好的 G_A2W，把整个 Amazon 训练集都过一遍，得到：

amazon2webcam/images/backpack/...
amazon2webcam/images/bike/...
...
amazon2webcam/images/desk_lamp/...
这里最关键的一点是：

类别标签不变。

因为图像虽然被改了风格，但语义还是同一个物体类别。
所以：

原图 amazon/images/backpack/xxx.jpg 标签是 backpack
翻译图 amazon2webcam/images/backpack/xxx_fake.jpg 标签仍然是 backpack
这一步之后，你就得到一个“视觉上更像目标域，但仍然有源域标签”的训练集。

第 3 步：在 amazon2webcam 上训练 ResNet18 分类器
这一步对应老仓库里的 “Train Source Net” 思想，只不过它的 src 在真正的 CyCADA 设定里已经变成“翻译后的源域”。

README 其实已经表达了这个意思：


README.md
Lines 47-51
## Train feature adaptation following image adaptation
- Use the feature space adapt code with the data and models from image adaptation
- For example: to train for the SVHN to MNIST shift, set `src = 'svhn2mnist'` and `tgt = 'mnist'` inside `scripts/train_adda.py` 
- Either download the relevant images above or run image space adaptation code and extract transferred images
映射到你的任务就是：

src = amazon2webcam
tgt = webcam
这一阶段训练 ResNet18 时，用的数据是：

训练输入：amazon2webcam
监督标签：来自原 Amazon 的 31 类标签
也就是说，这时你的分类器学的是：

“在 Webcam 风格外观下，怎样分 31 类。”

这一步训练完，先别急着说已经是最终 CyCADA 结果。
它更像是“pixel-adapted source model”或“CycleGAN baseline + classifier”。

第 4 步：做 feature-level adaptation
这是 CyCADA 相比单纯 “CycleGAN + classifier” 多出来的关键一步。

你现在已经有一个在 amazon2webcam 上训练好的 ResNet18。
接下来你做：

复制一份 encoder 作为 source encoder
再复制一份作为 target encoder
初始化时两者参数相同，来自上一步训练好的分类器
分类头通常也从这个模型初始化
然后训练一个 domain discriminator，让它区分：

source feature：来自 amazon2webcam
target feature：来自真实 webcam
同时更新 target encoder 去“骗过” discriminator，使得：

target encoder(webcam) 提取的特征
看起来像 source encoder(amazon2webcam) 提取出来的特征分布
老仓库里这一步就是 train_adda.py -> train_adda_net.py 的思路：


train_adda.py
Lines 52-73
#######################
# 1. Train Source Net #
#######################
# ...
#####################
# 2. Train Adda Net #
#####################
# ...
train_adda(src, tgt, model, num_cls, num_epoch=adda_num_epoch, 
        batch=batch, datadir=datadir,
        outdir=outdir, src_weights=src_net_file, 
        lr=adda_lr, betas=betas, weight_decay=weight_decay)
而它在 feature adaptation 时，训练数据配对方式就是：

src 训练集
tgt 训练集

train_adda_net.py
Lines 142-150
#######################################
# Setup data for training and testing #
#######################################
train_src_data = load_data(src, 'train', batch=batch, 
    rootdir=join(datadir, src), num_channels=net.num_channels, 
    image_size=net.image_size, download=True, kwargs=kwargs)
train_tgt_data = load_data(tgt, 'train', batch=batch, 
    rootdir=join(datadir, tgt), num_channels=net.num_channels, 
    image_size=net.image_size, download=True, kwargs=kwargs)
对你来说，这一步应该是：

source side：amazon2webcam，有标签但此阶段主要用于提供 source feature distribution
target side：webcam，无标签
初始化权重：来自第 3 步训练好的 ResNet18
学习目标：让真实 webcam 的 feature 更贴近 amazon2webcam 的 feature
最终测试时到底用哪个模型
最终评估时，你不用 source encoder，而是用adapt 后的 target encoder + classifier 去测真实 Webcam test。

测试集是：

输入：webcam/images/...
标签：真实 Webcam 类别标签，仅用于评估 accuracy
所以你的完整结果应该至少有 3 个可比实验：

Source-only

训练：amazon
测试：webcam
CycleGAN approach

训练：amazon2webcam
测试：webcam
CyCADA

先在 amazon2webcam 上训练分类器
再用 amazon2webcam + webcam(unlabeled) 做 feature adaptation
测试：webcam
你可以把整个流程记成这个“数据流图”
阶段 A：图像翻译
输入：amazon + webcam
训练模型：CycleGAN
得到模型：G_A2W
阶段 B：生成伪目标域源数据
输入：amazon
用模型：G_A2W
输出数据：amazon2webcam
标签来源：继承原 amazon 标签
阶段 C：训练分类器
输入：amazon2webcam
标签：31 类
训练模型：ResNet18
输出模型：pixel-adapted classifier
阶段 D：特征级对齐
source 数据：amazon2webcam
target 数据：webcam 无标签
初始化：第 C 步的 ResNet18
训练模型：target encoder + domain discriminator
输出模型：adapted target classifier
阶段 E：评估
测试数据：webcam
测试模型：第 D 步的 adapted target model
指标：classification accuracy
你最容易混乱的几个点
1. 不是把 Webcam 变成 Amazon
不是。
你的任务方向是 Amazon -> Webcam，所以真正对分类有用的是：

把有标签源域 Amazon 变成目标域风格 Webcam
而不是把 Webcam 变回 Amazon。

2. amazon2webcam 仍然算“源域监督数据”
是的。
它虽然视觉风格更像目标域，但标签来自源域，所以本质上还是“带标签训练集”。

3. feature adaptation 用的 source 不是原始 amazon
在严格按 CyCADA 思路时，feature adaptation 阶段更应该用：

source = amazon2webcam
target = webcam
而不是 source = amazon。

4. ResNet18 不是最后单独加上去的
不是。
ResNet18 就是你的 task network，本来就应该贯穿：

分类器训练阶段
feature adaptation 初始化阶段
最终测试阶段
你后续实现时，建议直接按这个命名
为了避免自己把数据流搞混，建议你后续代码里统一用下面这些名字：

amazon_train: 原始源域有标签数据
webcam_train_unlabeled: 目标域训练图像
webcam_test: 目标域测试集
amazon2webcam_train: 翻译后的源域训练集
source_model: 在 amazon2webcam_train 上训练的 ResNet18
target_model: feature adaptation 后用于测试 webcam 的模型
一句最简版总结
对 Amazon -> Webcam，CyCADA 做的是：

先用 Amazon + Webcam 训练图像翻译器，把 Amazon 全部变成 Webcam 风格得到 amazon2webcam；再用 amazon2webcam + 标签 训练 ResNet18；然后拿这个模型初始化 target encoder，用 amazon2webcam 和无标签 webcam 做 feature adversarial adaptation；最后在真实 webcam test 上评估。

如果你愿意，我下一步可以继续帮你把这个流程再落成一份更具体的“代码实现清单”，直接对应到你要写的 5 个模块：

dataset 怎么设计
CycleGAN 输出如何整理成 amazon2webcam
ResNet18 分类器怎么封装
ADDA/DANN 训练循环怎么写
实验脚本应该按什么顺序跑