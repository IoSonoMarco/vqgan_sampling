from torch.utils.data import Dataset
from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch
from lib import filepaths


@dataclass
class ImageTokenDatasetOutput:
    token_ids: torch.LongTensor
    label_id: torch.LongTensor


class ImageTokenDataset(Dataset):
    def __init__(self):

        self.data = np.load(
            filepaths.DATASET_IMAGE_TOKENS, 
            allow_pickle=True
        ).item()

        self.filepaths = list(self.data.keys())
        self.token_ids = np.array(list((self.data.values())))

        self.name_labels = [" ".join(x.split("/")[-2].split("_")[1:]) for x in self.filepaths]
        self.cat_labels = pd.Series(self.name_labels).astype("category").cat.codes

        self.cat_to_name_labels_mapping = {k:v for k,v in pd.DataFrame(dict(c=self.cat_labels, n=self.name_labels)).drop_duplicates().values}

        self.n_classes = len(self.cat_to_name_labels_mapping)

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):

        token_ids = self.token_ids[idx]
        
        label = self.cat_labels[idx]

        return ImageTokenDatasetOutput(
            token_ids=torch.tensor(token_ids).long(),
            label_id=torch.tensor(label).long()
        )
    

class ImageTokenDatasetAbstractCategories(ImageTokenDataset):
    def __init__(self):
        super().__init__()

        metadata = pd.read_table(filepaths.DATASET_IMAGE_METADATA)
        metadata.uniqueID = metadata.uniqueID.apply(lambda x: " ".join(x.split("_")))
        metadata = metadata[["uniqueID", "All Bottom-up Categories"]]
        metadata.columns = ["name", "bottomup_category"]

        # match labels order between metadata and dataset
        d_merged = metadata.merge(pd.DataFrame(dict(name=self.cat_to_name_labels_mapping)))
        d_merged = d_merged.set_index("name").loc[pd.Series(self.cat_to_name_labels_mapping)].reset_index()
        # filter out labels belonging to under-represented abstract categories (keep > 5 exemplars)
        d_filtered = d_merged[d_merged.bottomup_category.map(d_merged.bottomup_category.value_counts()) > 5]
        d_filtered_map = {k:v for k,v in d_filtered.values}

        # extract indices related to name labels belonging to abstract categories
        idx = np.where(pd.Series(self.name_labels).isin(d_filtered.name))[0]
        self.filepaths = np.array(self.filepaths)[idx].tolist()
        self.name_labels = [d_filtered_map[n] for n in np.array(self.name_labels)[idx]]
        self.token_ids = self.token_ids[idx]
        self.cat_labels = pd.Series(self.name_labels).astype("category").cat.codes
        self.cat_to_name_labels_mapping = {k:v for k,v in pd.DataFrame(dict(c=self.cat_labels, n=self.name_labels)).drop_duplicates().values}
        self.cat_to_name_labels_mapping = dict(sorted(self.cat_to_name_labels_mapping.items()))

        self.n_classes = len(self.cat_to_name_labels_mapping)

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):

        token_ids = self.token_ids[idx]
        
        label = self.cat_labels[idx]

        return ImageTokenDatasetOutput(
            token_ids=torch.tensor(token_ids).long(),
            label_id=torch.tensor(label).long()
        )

