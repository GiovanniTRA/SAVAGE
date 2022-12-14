import torch
from torch_geometric.data import InMemoryDataset, Data
import os
import os.path as osp


class CustomGraphDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["data_x.pt", "data_y.pt"]

    @property
    def processed_file_names(self):
        return ["data_all.pt"]

    @property
    def raw_dir(self):
        return osp.join(self.root, "raw")

    def process(self):
        # Read data into huge `Data` list.
        raw_files = sorted(os.listdir(self.raw_dir))
        x = torch.load(osp.join(self.raw_dir, raw_files[0]))
        y = torch.load(osp.join(self.raw_dir, raw_files[1]))
        data_list = [Data(x=x, edge_index=y)]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

