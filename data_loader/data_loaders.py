from torchvision import datasets, transforms
from base import BaseDataLoader
import dataset
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import default_collate
import image_body_dataset
import video_dataset

class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)



def my_collate(batch):
    batch = filter (lambda x:x is not None, batch)
    return torch.utils.data.dataloader.default_collate(batch)

def pad_collate(batch):
    batch = [x for x in batch if x is not None]
    # xx, face, hands_left, hands_right paths, targets, targets_continuous = zip(*batch)
    # xx, laban_body_component, embedding, places_features, paths, targets, targets_continuous = zip(*batch)
    xx, laban_body_component, targets, targets_continuous = zip(*batch)

    x_lens = [len(x) for x in xx]
    xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
    ll_pad = pad_sequence(laban_body_component, batch_first=True, padding_value=0)

    # xx_pad_face = pad_sequence(face, batch_first=True, padding_value=0)
    # xx_pad_hands_left = pad_sequence(hands_left, batch_first=True, padding_value=0)
    # xx_pad_hands_right = pad_sequence(hands_right, batch_first=True, padding_value=0)

    return xx_pad, ll_pad, default_collate(targets), default_collate(targets_continuous), torch.tensor(x_lens)

class BoLDDataLoader(BaseDataLoader):
    """
    BoLD data loading demo using BaseDataLoader
    """
    def __init__(self, mode, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        self.dataset = dataset.BoLD(mode=mode)
        if mode == "val":
            shuffle = False
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=pad_collate)

class BoLDDataLoaderImage(BaseDataLoader):
    """
    BoLD data loading demo using BaseDataLoader
    """
    def __init__(self, mode, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        self.dataset = image_body_dataset.BoLD(mode=mode)
        if mode == "val":
            shuffle = False
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class BoLDDataLoaderVideo(BaseDataLoader):
    """
    BoLD data loading demo using BaseDataLoader
    """
    def __init__(self, mode, batch_size, shuffle=True, validation_split=0.0, num_workers=1, **kwargs):
        self.dataset = video_dataset.VideoDataset(mode, **kwargs)
        if mode == "val":
            shuffle = False
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

