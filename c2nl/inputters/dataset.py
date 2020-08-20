# src: https://github.com/facebookresearch/DrQA/blob/master/drqa/reader/data.py
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler

from c2nl.inputters.vector import vectorize


# ------------------------------------------------------------------------------
# PyTorch dataset class for SQuAD (and SQuAD-like) data.
# ------------------------------------------------------------------------------


# class CommentDataset(Dataset):
#     def __init__(self, examples, model):
#         self.model = model
#         self.examples = examples

#     def __len__(self):
#         return len(self.examples)

#     def __getitem__(self, index):
#         return vectorize(self.examples[index], self.model)

#     def lengths(self):
#         return [(len(ex['code'].tokens), len(ex['summary'].tokens))
#                 for ex in self.examples]

###### larget dataset modification

import _pickle as pkl

class CommentDataset(Dataset):
    def __init__(self, examples, model, data_type = 'nottrain' ):
        self.model = model
        self.alllengths = [(len(ex['code'].tokens), len(ex['summary'].tokens)) for ex in examples]
        self.data_type = data_type
        self.dlength = len(examples)
        self.split = 50
        if self.data_type == 'train':
            divided = [ [] for i in range(self.split) ] 
            for i in range(self.dlength):
                chunk = i%self.split
                divided[chunk].append( examples[i] )
            
            for i in range(self.split):
                with open('../../temp_data/train' + str(i) + '.pkl', 'wb') as file:
                    file.write(pkl.dumps( divided[i] ))
        else:
            self.examples = examples
       
   

    def __len__(self):
        return self.dlength

    def __getitem__(self, index):
        if self.data_type == 'train':
            chunk = int(index%self.split)
            ind = int(index/self.split)
            with open('../../temp_data/train' + str(chunk) + '.pkl', 'rb') as file:
                data = pkl.load(file) 
                return vectorize(data[ind], self.model)
        else:
            return vectorize(self.examples[index], self.model)
        

    def lengths(self):
        return self.alllengths

# ------------------------------------------------------------------------------
# PyTorch sampler returning batched of sorted lengths (by doc and question).
# ------------------------------------------------------------------------------


class SortedBatchSampler(Sampler):
    def __init__(self, lengths, batch_size, shuffle=True):
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        lengths = np.array(
            [(-l[0], -l[1], np.random.random()) for l in self.lengths],
            dtype=[('l1', np.int_), ('l2', np.int_), ('rand', np.float_)]
        )
        indices = np.argsort(lengths, order=('l1', 'l2', 'rand'))
        batches = [indices[i:i + self.batch_size]
                   for i in range(0, len(indices), self.batch_size)]
        if self.shuffle:
            np.random.shuffle(batches)
        return iter([i for batch in batches for i in batch])

    def __len__(self):
        return len(self.lengths)
