from pfl.data.dataset import Dataset
from pfl.data.federated_dataset import FederatedDataset
from pfl.data.sampling import get_user_sampler
import numpy as np
import json

def make_dataset_fn(user_id):
    data = json.load(open('{}.json'.format(user_id), 'r'))
    features = np.array(data['x'])
    labels = np.eye(2)[data['y']] # Make one-hot encodings
    return Dataset(raw_data=[features, labels])

user_ids = ['user1', 'user2', 'user3']
sampler = get_user_sampler('minimize_reuse', user_ids)

for _ in range(5):
    print('sampled', sampler())

dataset = FederatedDataset(make_dataset_fn, sampler)


next(dataset).raw_data