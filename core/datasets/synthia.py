import os
import os.path as osp
import numpy as np
from torch.utils import data
from PIL import Image
import pickle
import cv2
# import imageio



class synthiaDataSet(data.Dataset):
    """
        original resolution at 1280x760
    """
    def __init__(
            self,
            data_root,
            data_list,
            max_iters=None,
            num_classes=16,
            split="train",
            transform=None,
            ignore_label=255,
            debug=False,
    ):
        self.split = split
        self.NUM_CLASS = num_classes
        self.data_root = data_root
        self.data_list = []
        with open(data_list, "r") as handle:
            content = handle.readlines()
        self.img_ids = [i_id.strip() for i_id in content]

        if max_iters is not None:
            self.label_to_file, self.file_to_label = pickle.load(
                open(osp.join(data_root, "synthia_label_info.p"), "rb"))
            self.img_ids = []
            SUB_EPOCH_SIZE = 3000
            tmp_list = []
            ind = dict()
            for i in range(self.NUM_CLASS):
                ind[i] = 0
            for e in range(int(max_iters / SUB_EPOCH_SIZE) + 1):
                cur_class_dist = np.zeros(self.NUM_CLASS)
                for i in range(SUB_EPOCH_SIZE):
                    if cur_class_dist.sum() == 0:
                        dist1 = cur_class_dist.copy()
                    else:
                        dist1 = cur_class_dist / cur_class_dist.sum()
                    w = 1 / np.log(1 + 1e-2 + dist1)
                    w = w / w.sum()
                    c = np.random.choice(self.NUM_CLASS, p=w)

                    if ind[c] > (len(self.label_to_file[c]) - 1):
                        # np.random.shuffle(self.label_to_file[c])
                        ind[c] = ind[c] % (len(self.label_to_file[c]) - 1)

                    c_file = self.label_to_file[c][ind[c]]
                    tmp_list.append(c_file)
                    ind[c] = ind[c] + 1
                    cur_class_dist[self.file_to_label[c_file]] += 1

            self.img_ids = tmp_list

        for name in self.img_ids:
            self.data_list.append(
                {
                    "img": os.path.join(self.data_root, "RAND_CITYSCAPES/RGB/%s" % name),
                    "label": os.path.join(self.data_root, "RAND_CITYSCAPES/synthia_mapped_to_cityscapes/%s" % name),
                    # change
                    "depth": os.path.join(self.data_root, "RAND_CITYSCAPES/Depth/Depth/%s" % name),
                    "name": name,
                    
                }
            )

        if max_iters is not None:
            self.data_list = self.data_list * int(np.ceil(float(max_iters) / len(self.data_list)))

        self.id_to_trainid = {
            3: 0,
            4: 1,
            2: 2,
            21: 3,
            5: 4,
            7: 5,
            15: 6,
            9: 7,
            6: 8,
            1: 9,
            10: 10,
            17: 11,
            8: 12,
            19: 13,
            12: 14,
            11: 15,
        }
        self.trainid2name = {
            0: "road",
            1: "sidewalk",
            2: "building",
            3: "wall",
            4: "fence",
            5: "pole",
            6: "light",
            7: "sign",
            8: "vegetation",
            9: "sky",
            10: "person",
            11: "rider",
            12: "car",
            13: "bus",
            14: "motocycle",
            15: "bicycle",
        }
        self.transform = transform
        self.ignore_label = ignore_label
        self.debug = debug

    def __len__(self):
        return len(self.data_list)
    
    def get_depth(self, file):
        depth = np.array(cv2.imread(file, flags=cv2.IMREAD_ANYDEPTH), dtype=np.float32)/100.
        depth = 655.36 / (depth + 0.01)
        return depth

    def __getitem__(self, index):
        if self.debug:
            index = 0
        datafiles = self.data_list[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        label = np.array(Image.open(datafiles["label"]),dtype=np.float32)#[:, :, 0]
        # change
        depth = self.get_depth(datafiles["depth"])

        # re-assign labels to match the format of Cityscapes
        label_copy = self.ignore_label * np.ones(label.shape, dtype=np.float32)
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v

        label = Image.fromarray(label_copy)

        if self.transform is not None:
            image, label, depth = self.transform(image, label, depth)

        return image, label, depth, datafiles['name'] 
    



if __name__ == '__main__':

 # import build_dataset
    import torch 
    import torchvision
    import matplotlib.pyplot as plt
    from torch.utils import data
    from PIL import Image
    from torchvision import transforms
    from torchvision.transforms import functional as F

    class Compose(object):
        def __init__(self, transforms):
            self.transforms = transforms
        def __call__(self, image, label, depth):
            for t in self.transforms:
                image, label, depth = t(image, label, depth)
            return image, label, depth
    class ToTensor(object):
        def __call__(self, image, label, depth):
            return F.to_tensor(image), F.to_tensor(label).squeeze(), F.to_tensor(depth).squeeze()

    transform = Compose([
        # you can add other transformations in this list
        ToTensor()
    ])

    DATASET_DIR = "../../datasets"
    DATASETS = {
        "synthia_train": {
            "data_dir": "synthia",
            "data_list": "synthia_train_list.txt"
        }
    }
    data_dir = DATASET_DIR
    attrs = DATASETS["synthia_train"]
    args = dict(
        root=os.path.join(data_dir, attrs["data_dir"]),
        data_list=os.path.join(data_dir, attrs["data_list"]),
    )
    # print(args["root"],args["data_list"])
    src_train_data = synthiaDataSet(args["root"], args["data_list"], max_iters=20, num_classes=16,
                            split='train', transform=transform)
    train_loader = torch.utils.data.DataLoader(src_train_data,
                                               batch_size=2,
                                               shuffle=None,
                                               num_workers=1,
                                               pin_memory=None,
                                               sampler=None,
                                               drop_last=False)
    for i, (src_input, src_label, src_depth) in enumerate(train_loader):
        if i ==0 : 
            # print(name)
            print(src_depth)