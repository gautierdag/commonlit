import math
import torch
from torch.utils.data.sampler import RandomSampler


class BatchSchedulerSampler(torch.utils.data.sampler.Sampler):
    """
    Iterate over tasks and provide a random batch per task in each mini-batch
    """

    def __init__(self, dataset, batch_size, chunk_task_batches=1, max_size=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.number_of_datasets = len(dataset.datasets)
        self.chunk_task_batches = (
            chunk_task_batches  # number of task batches to show consecutively
        )
        if max_size:
            self.largest_dataset_size = max_size
        else:
            self.largest_dataset_size = max(
                [
                    len(cur_dataset.df)
                    for cur_dataset in dataset.datasets
                    if not isinstance(cur_dataset, torch.utils.data.dataset.Subset)
                ]
            )
        self.total_num_examples = sum(
            [
                len(cur_dataset.df)
                for cur_dataset in dataset.datasets
                if not isinstance(cur_dataset, torch.utils.data.dataset.Subset)
            ]
        )

    def __len__(self):
        return (
            self.batch_size
            * math.ceil(self.largest_dataset_size / self.batch_size)
            * len(self.dataset.datasets)
        )
        # total = sum(
        #     [
        #         math.floor(self.total_num_examples / self.batch_size)
        #         for cur_dataset in self.dataset.datasets
        #         if not isinstance(cur_dataset, torch.utils.data.dataset.Subset)
        #     ]
        # )
        # return self.batch_size * total

    def __iter__(self):
        samplers_list = []
        sampler_iterators = []
        for dataset_idx in range(self.number_of_datasets):
            cur_dataset = self.dataset.datasets[dataset_idx]
            sampler = RandomSampler(cur_dataset)
            samplers_list.append(sampler)
            cur_sampler_iterator = sampler.__iter__()
            sampler_iterators.append(cur_sampler_iterator)

        push_index_val = [0] + self.dataset.cumulative_sizes[:-1]
        step = self.batch_size * self.number_of_datasets
        samples_to_grab = self.batch_size

        # for this case we want to get all samples in dataset, this force us to resample from the smaller datasets
        epoch_samples = (
            self.largest_dataset_size
            * self.number_of_datasets
            # self.total_num_examples
            // self.chunk_task_batches
        )

        final_samples_list = []  # this is a list of indexes from the combined dataset
        for _ in range(0, epoch_samples, step):
            for i in range(self.number_of_datasets):
                for _ in range(self.chunk_task_batches):
                    cur_batch_sampler = sampler_iterators[i]
                    cur_samples = []
                    full_batch = True
                    for _ in range(samples_to_grab):
                        try:
                            cur_sample_org = cur_batch_sampler.__next__()
                            cur_sample = cur_sample_org + push_index_val[i]
                            cur_samples.append(cur_sample)
                        except StopIteration:
                            # got to the end of iterator - restart the iterator and continue to get samples
                            # until reaching "epoch_samples"
                            sampler_iterators[i] = samplers_list[i].__iter__()
                            cur_batch_sampler = sampler_iterators[i]
                            cur_sample_org = cur_batch_sampler.__next__()
                            cur_sample = cur_sample_org + push_index_val[i]
                            cur_samples.append(cur_sample)
                            # full_batch = False
                            # continue
                    # if full_batch:
                    final_samples_list.extend(cur_samples)

        return iter(final_samples_list)
