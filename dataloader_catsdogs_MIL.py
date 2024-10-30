from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import json
import os
import torch
from torchnet.meter import AUCMeter


class CatsDogsDataset(Dataset):
    def __init__(self, dataset, r, noise_mode, root_dir, transform, mode, noise_file='', pred=[], probability=[],
                 log=''):
        self.r = r  # noise ratio
        self.transform = transform
        self.mode = mode
        self.root_dir = root_dir
        self.classes = ['cats', 'dogs']  # class names for cats and dogs
        self.noise_mode = noise_mode
        self.dataset = dataset

        if self.mode == 'test':
            self.test_data, self.test_labels = self._load_images(os.path.join(root_dir, 'test'))

        else:  # training data
            self.train_data, self.train_labels = self._load_images(os.path.join(root_dir, 'train'))

            if os.path.exists(noise_file):
                noise_label = json.load(open(noise_file, "r"))
            else:
                noise_label = self._inject_noise(self.train_labels)
                print("Saving noisy labels to %s ..." % noise_file)
                json.dump(noise_label, open(noise_file, "w"))

            if self.mode == 'all':
                self.noise_label = noise_label
            else:
                if self.mode == "labeled":
                    pred_idx = pred.nonzero()[0]
                    self.probability = [probability[i] for i in pred_idx]
                    clean = (np.array(noise_label) == np.array(self.train_labels))
                    auc_meter = AUCMeter()
                    auc_meter.reset()
                    auc_meter.add(probability, clean)
                    auc, _, _ = auc_meter.value()
                    log.write('Number of labeled samples:%d   AUC:%.3f\n' % (pred.sum(), auc))
                    log.flush()

                elif self.mode == "unlabeled":
                    pred_idx = (1 - pred).nonzero()[0]

                self.train_data = [self.train_data[i] for i in pred_idx]
                self.noise_label = [noise_label[i] for i in pred_idx]
                print("%s data has a size of %d" % (self.mode, len(self.noise_label)))

    def _load_images(self, folder):
        data = []
        labels = []
        for idx, class_name in enumerate(self.classes):
            class_folder = os.path.join(folder, class_name)
            for file_name in os.listdir(class_folder):
                file_path = os.path.join(class_folder, file_name)
                data.append(file_path)
                labels.append(idx)
        return data, labels

    def _inject_noise(self, labels):
        noise_label = []
        idx = list(range(len(labels)))
        random.shuffle(idx)
        num_noise = int(self.r * len(labels))
        noise_idx = idx[:num_noise]

        for i in range(len(labels)):
            if i in noise_idx:
                if self.noise_mode == 'sym':
                    noiselabel = 1 - labels[i]  # flip class for symmetric noise
                elif self.noise_mode == 'asym':
                    noiselabel = labels[i]  # asymmetric noise doesn't change class for binary case
                noise_label.append(noiselabel)
            else:
                noise_label.append(labels[i])
        return noise_label

    def __getitem__(self, index):
        if self.mode == 'labeled':
            img_path, target, prob = self.train_data[index], self.noise_label[index], self.probability[index]
            img = Image.open(img_path).convert('RGB')
            img1 = self.transform(img)
            img2 = self.transform(img)
            return img1, img2, target, prob

        elif self.mode == 'unlabeled':
            img_path = self.train_data[index]
            img = Image.open(img_path).convert('RGB')
            img1 = self.transform(img)
            img2 = self.transform(img)
            return img1, img2

        elif self.mode == 'all':
            img_path, target = self.train_data[index], self.noise_label[index]
            img = Image.open(img_path).convert('RGB')
            img = self.transform(img)
            return img, target, index

        elif self.mode == 'test':
            img_path, target = self.test_data[index], self.test_labels[index]
            img = Image.open(img_path).convert('RGB')
            img = self.transform(img)
            return img, target

    def __len__(self):
        if self.mode != 'test':
            return len(self.train_data)
        else:
            return len(self.test_data)


class CatsDogsDataloader():
    def __init__(self, dataset, r, noise_mode, batch_size, num_workers, root_dir, log, noise_file=''):
        self.dataset = dataset
        self.r = r
        self.noise_mode = noise_mode
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root_dir = root_dir
        self.log = log
        self.noise_file = noise_file

        self.transform_train = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        self.transform_test = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def run(self, mode, pred=[], prob=[]):
        if mode == 'warmup':
            all_dataset = CatsDogsDataset(
                dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir,
                transform=self.transform_train, mode="all", noise_file=self.noise_file
            )
            trainloader = DataLoader(
                dataset=all_dataset,
                batch_size=self.batch_size * 2,
                shuffle=True,
                num_workers=self.num_workers
            )
            return trainloader

        elif mode == 'train':
            labeled_dataset = CatsDogsDataset(
                dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir,
                transform=self.transform_train, mode="labeled", noise_file=self.noise_file,
                pred=pred, probability=prob, log=self.log
            )
            labeled_trainloader = DataLoader(
                dataset=labeled_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers
            )

            unlabeled_dataset = CatsDogsDataset(
                dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir,
                transform=self.transform_train, mode="unlabeled", noise_file=self.noise_file, pred=pred
            )
            unlabeled_trainloader = DataLoader(
                dataset=unlabeled_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers
            )
            return labeled_trainloader, unlabeled_trainloader

        elif mode == 'test':
            test_dataset = CatsDogsDataset(
                dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir,
                transform=self.transform_test, mode='test'
            )
            test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers
            )
            return test_loader

        elif mode == 'eval_train':
            eval_dataset = CatsDogsDataset(
                dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir,
                transform=self.transform_test, mode='all', noise_file=self.noise_file
            )
            eval_loader = DataLoader(
                dataset=eval_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers
            )
            return eval_loader
