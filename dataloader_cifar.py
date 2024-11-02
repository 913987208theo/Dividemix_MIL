from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import json
import os
import torch
from torchnet.meter import AUCMeter

'''
这段代码主要是实现了一个自定义的数据加载器，目的是用于加载并处理CIFAR-10和CIFAR-100数据集，其中包含数据增强、噪声注入等功能。

# Mode           用途                     加载内容                   处理逻辑                                    特点
# test           模型测试和评估           测试数据及其标签           只加载测试数据，无训练数据处理              用于模型在测试集上的评估
# all            完整数据加载             所有训练数据和标签         加载完整训练数据和噪声标签                  提供完整数据，用于初始训练
# labeled        半监督学习的有标签部分     有标签的训练数据           筛选带标签样本，计算 AUC，记录日志         训练有标签的样本
# unlabeled      半监督学习的无标签部分     无标签的训练数据           筛选无标签样本，无需计算 AUC               用于训练无标签的样本

'''



'''该函数用于加载二进制格式的CIFAR数据文件。它使用pickle库将数据解码成字典格式，便于后续处理。——>反序列化'''
def unpickle(file):
    import _pickle as cPickle
    with open(file, 'rb') as fo: #'rb' 表示以二进制读模式打开文件。
        dict = cPickle.load(fo, encoding='latin1')  #数指定了字符编码为 latin1（ISO-8859-1），这在处理 Python 2 中序列化的数据时尤其有用。
    return dict
    '''
    函数最终返回加载的数据字典 dict，该字典包含了从 file 中读取的所有数据。
    CIFAR-10 的数据集文件通常是以二进制格式存储的，因此需要使用反序列化方法（如 unpickle 函数）来读取它们。
    '''


'''
cifar_dataset 继承了torch.utils.data.Dataset，是一个自定义数据集类。
主要用于根据不同模式（如测试集、带标签训练集和无标签训练集）加载数据，并实现数据增强、噪声注入等功能。
'''
class cifar_dataset(Dataset): 
    def __init__(self, dataset, r, noise_mode, root_dir, transform, mode, noise_file='', pred=[], probability=[], log=''): 
        '''
        dataset: 数据集类型（CIFAR-10 或 CIFAR-100）。
        r: 噪声率，表示注入噪声数据的比例。
        noise_mode: 噪声模式，sym表示对称噪声，asym表示非对称噪声。
        root_dir: 数据集存储的根目录。
        transform: 数据增强操作。
        mode: 加载模式，包括test（测试）、all（所有数据）、labeled（有标签数据）、unlabeled（无标签数据）。
        noise_file: 用于存储噪声标签的文件路径。
        '''
        self.r = r      # noise ratio
        self.transform = transform
        self.mode = mode
        self.transition = {0:0,2:0,4:7,7:7,1:1,9:1,3:5,5:3,6:6,8:8} # class transition for asymmetric noise
        #self.transition 是一个字典，用于定义非对称噪声下的类别标签转换关系。它的作用是根据给定的原始标签来确定噪声标签。比如，当原始标签为 4 时，噪声标签会被转换为 7。
        #此处可以自己定义，并且此处适用于十分类。


        #此处对数据集文件的打开可以修改。我们不是二进制，而是一个文件夹
        if self.mode=='test':
            if dataset=='cifar10':                
                test_dic = unpickle('%s/test_batch'%root_dir)  #调用unpickle进行反序列化，返回一个字典
                self.test_data = test_dic['data']
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))   # 10000 张图片，每张图片有 3 个通道（RGB）和 32x32 像素。
                self.test_data = self.test_data.transpose((0, 2, 3, 1))#通过 transpose 将数据从 (10000, 3, 32, 32) 转换为 (10000, 32, 32, 3) 的形状，使得数据可以与常用的图像处理库（如 PIL）兼容。
                self.test_label = test_dic['labels'] #将测试标签存储在 self.test_label 中。
            elif dataset=='cifar100':
                test_dic = unpickle('%s/test'%root_dir)
                self.test_data = test_dic['data']
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))  
                self.test_label = test_dic['fine_labels']                            

        else: #加载 CIFAR-10 或 CIFAR-100 数据集的训练数据
            train_data=[] #初始化训练数据和标签列表
            train_label=[]
            if dataset=='cifar10': 
                for n in range(1,6): #使用 for 循环遍历 1 到 5 的数字，这对应于 CIFAR-10 的五个训练数据文件（data_batch_1 到 data_batch_5）。
                    dpath = '%s/data_batch_%d'%(root_dir,n)
                    data_dic = unpickle(dpath) #调用 unpickle(dpath) 来读取数据，返回一个字典 data_dic，其中包含图像数据和标签。
                    train_data.append(data_dic['data'])
                    train_label = train_label+data_dic['labels']  #目的   是将从每个数据批次加载的标签合并到 train_label 列表中
                train_data = np.concatenate(train_data) #最后，使用 np.concatenate 将所有的训练数据合并为一个大数组。
            elif dataset=='cifar100':    
                train_dic = unpickle('%s/train'%root_dir)
                train_data = train_dic['data']
                train_label = train_dic['fine_labels']
            train_data = train_data.reshape((50000, 3, 32, 32))  #这里将 train_data 重新调整为 (50000, 3, 32, 32) 的形状
            train_data = train_data.transpose((0, 2, 3, 1)) #这里的 transpose 操作将 train_data 从 (50000, 3, 32, 32) 转换为 (50000, 32, 32, 3) 的形状。

            if os.path.exists(noise_file):  #如果指定的 noise_file 文件存在，直接从该文件中加载噪声标签。这样可以避免每次运行代码时重新生成噪声标签，从而保证结果的一致性。
                noise_label = json.load(open(noise_file,"r"))  #json.load 会将 noise_file 中的内容读取为 Python 列表 noise_label。
            else:    #inject noise     # 如果噪声文件不存在，则注入噪声
                noise_label = [] # 如果噪声文件不存在，则注入噪声
                idx = list(range(50000)) #生成一个包含从 0 到 49999 的索引列表，用于表示 50000 张训练图像的索引。
                random.shuffle(idx) #将索引顺序打乱，以随机选择要注入噪声的标签。
                num_noise = int(self.r*50000)   # 计算需要注入噪声的标签数量（由噪声率 self.r 决定）。
                noise_idx = idx[:num_noise] #从打乱后的索引中选择前 num_noise 个索引，这些索引将用于注入噪声。

                '''注入噪声标签'''
                for i in range(50000):
                    if i in noise_idx:  #如果 i 属于 noise_idx（即该索引需要注入噪声），则按照噪声模式决定如何生成噪声标签：
                        if noise_mode=='sym': # 对称噪声 (sym)：随机替换标签
                            if dataset=='cifar10': 
                                noiselabel = random.randint(0,9)  #CIFAR-10 中随机生成一个 [0, 9] 范围的整数作为新标签。
                            elif dataset=='cifar100':    
                                noiselabel = random.randint(0,99)
                            noise_label.append(noiselabel)
                        elif noise_mode=='asym':    #非对称噪声 (asym)
                            noiselabel = self.transition[train_label[i]] #使用 self.transition 字典中的映射规则将标签替换为另一类标签。
                            noise_label.append(noiselabel)
                    else:
                        noise_label.append(train_label[i])    #如果 i 不在 noise_idx 中，则保留原始标签。
                print("save noisy labels to %s ..."%noise_file)   #这行代码会输出一条提示信息，告知用户即将把生成的噪声标签保存到指定的文件路径。
                json.dump(noise_label,open(noise_file,"w"))    #函数用于将 Python 对象（例如列表或字典）以 JSON 格式写入文件。
            
            if self.mode == 'all':
                self.train_data = train_data
                self.noise_label = noise_label

            else:
                if self.mode == "labeled":
                    '''该代码片段主要用于筛选带标签的样本、计算 AUC（Area Under Curve，曲线下面积），并将相关信息写入日志文件。
                    AUC 是一个评估指标，用于衡量预测值与真实标签之间的匹配程度，尤其在不平衡数据集上有很好的表现。
                    '''
                    pred_idx = pred.nonzero()[0]  #pred.nonzero() 返回 pred 数组中非零元素的索引，[0] 表示选取第一个维度的索引。
                    self.probability = [probability[i] for i in pred_idx]  #使用列表推导从 probability 数组中提取 pred_idx 索引对应的概率值，并将其存储在 self.probability 中。
                    
                    clean = (np.array(noise_label)==np.array(train_label))  #生成一个布尔数组 clean，其中 True 表示噪声标签与原始标签匹配（即“干净”标签），False 表示噪声标签不匹配（即“带噪声”标签）。
                    auc_meter = AUCMeter()
                    auc_meter.reset() #reset() 方法将 auc_meter 重置，确保没有残留数据，便于重新计算。
                    auc_meter.add(probability,clean)
                    '''auc_meter.add(probability, clean) 将 probability 和 clean 传入 AUC 计算器，
                    其中 probability 是模型的预测概率，clean 是标记“干净”或“噪声”的布尔标签。
                    add方法计算 probability 与 clean 标签的关系，以估算模型的预测准确性。
                    !!模型的预测概率可以理解为模型对标签预测的“信任值”或“置信度”
                    '''
                    auc,_,_ = auc_meter.value()   #计算并返回 AUC 相关的数值。取值范围在 0 到 1 之间，越接近 1 表示模型性能越好，特别是在非平衡数据集中，AUC 可以更好地反映模型的实际表现。
                    log.write('Numer of labeled samples:%d   AUC:%.3f\n'%(pred.sum(),auc))
                    log.flush()  #用于立即将缓存在内存中的日志内容写入文件，确保日志内容实时更新。尤其在模型训练过程中，实时记录 AUC 值和样本信息便于监控模型性能。
                    
                elif self.mode == "unlabeled":
                    pred_idx = (1-pred).nonzero()[0]  #1 - pred 的操作会将 pred 中的 1 变为 0，0 变为 1，这样就反转了标签状态，即 1 表示无标签样本。
                
                self.train_data = train_data[pred_idx] #无标签数据：self.train_data = train_data[pred_idx] 将无标签样本的数据从 train_data 中筛选出来，并赋值给 self.train_data。
                self.noise_label = [noise_label[i] for i in pred_idx]     #无标签噪声标签
                print("%s data has a size of %d"%(self.mode,len(self.noise_label)))            



    '''该方法实现了根据索引返回一个样本的数据和标签，用于支持数据迭代。'''
    def __getitem__(self, index):
        if self.mode=='labeled': #返回两个经过数据增强的图像、标签和概率（用于表示模型对标签的置信度。）。
            img, target, prob = self.train_data[index], self.noise_label[index], self.probability[index]
            img = Image.fromarray(img)  #CIFAR 数据集中的图像通常以 NumPy 数组的形式存储。Image.fromarray(img) 将图像从 NumPy 格式转换为 PIL 图像格式，以便应用数据增强。
            img1 = self.transform(img)  #self.transform(img)：对图像进行数据增强（例如随机裁剪、旋转等操作），生成两个增强后的图像 img1 和 img2。
            img2 = self.transform(img) 
            return img1, img2, target, prob            
        elif self.mode=='unlabeled': #返回两个经过数据增强的图像。
            img = self.train_data[index]
            img = Image.fromarray(img)
            img1 = self.transform(img) 
            img2 = self.transform(img) 
            return img1, img2
        elif self.mode=='all':  #返回单个经过数据增强的图像和标签。
            img, target = self.train_data[index], self.noise_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)            
            return img, target, index        
        elif self.mode=='test':  #返回单个图像和测试标签。
            img, target = self.test_data[index], self.test_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)            
            return img, target


    '''返回数据集的大小。测试模式返回测试数据集的大小，训练模式返回训练数据集的大小'''
    def __len__(self):
        if self.mode!='test':
            return len(self.train_data)
        else:
            return len(self.test_data)         
        
'''该类封装了数据加载器，用于生成不同模式下的数据加载器（DataLoader），包括训练集、测试集、带标签数据和无标签数据的加载。 '''
class cifar_dataloader():  
    def __init__(self, dataset, r, noise_mode, batch_size, num_workers, root_dir, log, noise_file=''):
        self.dataset = dataset #数据集名称，指定要加载的数据集类型。
        self.r = r #噪声比例（noise ratio）
        self.noise_mode = noise_mode  #指定噪声的类型，通常为 'sym' 或 'asym'。
        self.batch_size = batch_size #批处理大小。
        self.num_workers = num_workers  #工作线程数。
        self.root_dir = root_dir  #数据集根目录。
        self.log = log  #日志文件对象
        self.noise_file = noise_file #噪声标签文件路径。
        if self.dataset=='cifar10':  #用于训练集的数据增强和标准化操作。定义了一系列的图像变换步骤，以提升模型的泛化能力：
            self.transform_train = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),  #随机裁剪大小为 32x32 的图像，并添加 4 像素的填充。
                    transforms.RandomHorizontalFlip(), #以 50% 的概率随机水平翻转图像。
                    transforms.ToTensor(),  #将图像从 PIL 格式转换为 PyTorch 的张量格式。同时将图像像素值从 [0, 255] 缩放到 [0, 1] 的范围，方便模型处理。
                    transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),#对图像的每个通道（R、G、B）进行归一化处理。
                        #将每个通道的均值和标准差调整为指定的值，使像素值更加标准化，有助于模型更快地收敛。
                            '''均值：(0.4914, 0.4822, 0.4465)  
                               标准差：(0.2023, 0.1994, 0.2010)
                            '''

                ]) 
            self.transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),  #cifar10公认的标准值
                ])    
        elif self.dataset=='cifar100': #用于测试集的标准化操作。与训练集不同，测试集仅进行归一化处理，不做数据增强：
            self.transform_train = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
                ]) 
            self.transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
                ])


    '''该方法根据传入的 mode 参数，生成对应模式的数据加载器。'''
    def run(self,mode,pred=[],prob=[]):
        if mode=='warmup':
            all_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_train, mode="all",noise_file=self.noise_file)                
            trainloader = DataLoader(
                dataset=all_dataset, 
                batch_size=self.batch_size*2,
                shuffle=True,
                num_workers=self.num_workers)             
            return trainloader
                                     
        elif mode=='train':
            labeled_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_train, mode="labeled", noise_file=self.noise_file, pred=pred, probability=prob,log=self.log)              
            labeled_trainloader = DataLoader(
                dataset=labeled_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)   
            
            unlabeled_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_train, mode="unlabeled", noise_file=self.noise_file, pred=pred)                    
            unlabeled_trainloader = DataLoader(
                dataset=unlabeled_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)     
            return labeled_trainloader, unlabeled_trainloader
        
        elif mode=='test':
            test_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_test, mode='test')      
            test_loader = DataLoader(
                dataset=test_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)          
            return test_loader
        
        elif mode=='eval_train':
            eval_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_test, mode='all', noise_file=self.noise_file)      
            eval_loader = DataLoader(
                dataset=eval_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)          
            return eval_loader        
