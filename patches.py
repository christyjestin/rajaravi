import bisect
import numpy as np
from heapq import *
from tqdm import trange, tqdm


def weighted_average(color1: np.ndarray, color2: np.ndarray, count1: int, count2: int):
    alpha = count1 / (count1 + count2)
    return color1 * alpha + color2 * (1 - alpha)

def dist(a: np.ndarray, b: np.ndarray):
    # divide before computing norm to avoid numerical overflow
    return np.linalg.norm((a - b) / 256) * 256

# return pair such that the first element is the lesser of the two
# helper function for maintaining i < j invariant
def reorder_pair(a, b):
    return (a, b) if a < b else (b, a)

class Patches:
    def __init__(self, img) -> None:
        assert isinstance(img, np.ndarray) and len(img.shape) == 3 and img.shape[2] == 3, \
            "Input is not an RGB image"

        self.height, self.width, _ = img.shape
        num_patches = self.height * self.width

        # all of these data structures have the patch id as the key/index
        self.patch_counts = {i: 1 for i in range(num_patches)}
        self.patch_colors = img.reshape((-1, 3)).astype(np.float16)
        # vals are n x 2 arrays
        self.patch_indices = {i * self.width + j: np.array([[i, j]]) for i in range(self.height) 
                              for j in range(self.width)}
        self.patch_adjacencies = {i: self.get_adjacencies(i) for i in range(num_patches)}

        # maps img indices to patch ids
        self.img_patches = np.arange(num_patches).reshape((self.height, self.width))
        self.existing_patches = set(range(num_patches))

        # cache negative distances: we negate because the heapq sorts least to greatest
        # and we want to remove the greatest distances when we use the heapq
        # see the method `cull` for context
        self.neg_dists = dict()
        for i in trange(num_patches, desc = "Computing cache"):
            for j in self.patch_adjacencies[i]:
                if i < j: # only add each pair once
                    self.neg_dists[(i, j)] = self.compute_neg_dist(i, j)

    @property
    def num_adjacencies(self):
        return len(self.neg_dists)

    @property
    def num_patches(self):
        return len(self.existing_patches)

    def compute_neg_dist(self, i, j):
        return -dist(self.patch_colors[i], self.patch_colors[j])

    # given a (row first) array index, return the set of adjacent patches
    def get_adjacencies(self, i):
        r, c = i // self.width, i % self.width
        adj = set()
        if r > 0: adj.add(i - self.width) # patch above
        if r + 1 < self.height: adj.add(i + self.width) # patch below
        if c > 0: adj.add(i - 1) # patch to the left
        if c + 1 < self.width: adj.add(i + 1) # patch to the right
        return adj

    # merge while keeping set i and deleting j
    # N.B. this method does not compute neg_dists; it's done later to avoid recomputes
    def merge(self, i, j):
        assert i < j, f"Does not maintain the invariant of i < j: {i}, {j}"
        assert i in self.existing_patches, f"This patch was already deleted: {i}"
        assert j in self.existing_patches, f"This patch was already deleted: {j}"
        self.patch_colors[i] = weighted_average(self.patch_colors[i], self.patch_colors[j], 
                                                self.patch_counts[i], self.patch_counts[j])
        self.patch_counts[i] += self.patch_counts[j]
        self.patch_indices[i] = np.concatenate((self.patch_indices[i], self.patch_indices[j]))
        assert self.patch_counts[i] == self.patch_indices[i].shape[0]
        self.patch_adjacencies[i] = self.patch_adjacencies[i] | self.patch_adjacencies[j]
        self.patch_adjacencies[i] -= {i, j} # remove adjacencies to self
        row, col = np.split(self.patch_indices[j], 2, axis = -1)
        self.img_patches[row, col] = i

        # fix references to j
        for other in self.patch_adjacencies[j]:
            if other != i:
                # replace with reference to i
                self.patch_adjacencies[other].remove(j)
                self.patch_adjacencies[other].add(i)
                # remove from our cache of neg_dists
                del self.neg_dists[reorder_pair(j, other)]

        # compute and cache new neg_dists
        for new_j in self.patch_adjacencies[i]:
            self.neg_dists[reorder_pair(i, new_j)] = self.compute_neg_dist(i, new_j)

        # clean up the data structures
        del self.patch_counts[j]
        # N.B. self.patch_colors is an array, so can't delete row
        del self.patch_indices[j]
        del self.patch_adjacencies[j]
        self.existing_patches.remove(j)
        del self.neg_dists[(i, j)]

    def construct_img(self):
        return np.rint(self.patch_colors[self.img_patches]).astype(np.uint8)

    # finds the closest n% pairs of adjacent patches to merge
    # returns the entire heap i.e. a list of items of the form (neg_dist, (i, j))
    def cull_by_fraction(self, cull_fraction):
        print(cull_fraction)
        cull_count = round(cull_fraction * self.num_patches)

        heap = []
        for pair, neg_dist in self.neg_dists.items():
            if len(heap) < cull_count:
                heappush(heap, (neg_dist, pair))
            # add to heap if the current neg_dist is greater than the neg_dist of the root
            elif neg_dist > heap[0][0]:
                heapreplace(heap, (neg_dist, pair))
        return heap

    # force all patches smaller than the threshold size to merge with the closest adjacent patch
    def cull_by_size(self, threshold):
        # convert to list to get a static copy
        for i in list(self.existing_patches):
            # skip patches if they've already been removed or if they're sufficiently large
            if i not in self.existing_patches or self.patch_counts[i] > threshold:
                continue
            closest, closest_dist = None, -np.inf
            for j in self.patch_adjacencies[i]:
                pair = reorder_pair(i, j)
                if self.neg_dists[pair] > closest_dist:
                    closest, closest_dist = j, self.neg_dists[pair]
            self.merge(*reorder_pair(closest, i))

    def run(self):
        output = []

        # Phase 1: Bulk merge patches
        cull_target = 2000
        round = 0
        initial = self.num_patches
        t = tqdm(total=initial - cull_target, desc="Merging patches:", miniters=1)
        while self.num_patches > cull_target:
            # run through the merges
            cull_fraction = 0.8 if self.num_patches > 100000 else 0.5
            for _, (i, j) in self.cull_by_fraction(cull_fraction):
                # check in case one of the patches has already been merged and deleted
                if i in self.existing_patches and j in self.existing_patches:
                    self.merge(i, j)
            cull_by_size_start = 2
            if round >= cull_by_size_start:
                self.cull_by_size(2 ** (round - cull_by_size_start))
            round += 1
            output.append((self.num_patches, self.construct_img()))
            t.update(initial - self.num_patches)
        t.close()

        # Phase 2: Merge one pair of patches at a time
        # setup queue for single merges
        # we use the number of patches as a timestamp
        self.patch_last_update = {p: self.num_patches for p in self.existing_patches}
        self.neg_dists = dict()
        for i in tqdm(self.existing_patches, desc = "Recomputing cache"):
            for j in self.patch_adjacencies[i]:
                if i < j: # only add each pair once
                    self.neg_dists[(i, j)] = self.compute_neg_dist(i, j)
        self.queue = [(neg_dist, *pair, self.num_patches) for pair, neg_dist in self.neg_dists.items()]
        self.queue.sort()

        for _ in trange(self.num_patches - 1):
            # pop until you find a valid pair
            while True:
                _, i, j, last_update = self.queue.pop()
                valid_patches = i in self.existing_patches and j in self.existing_patches
                # compare the pair's last update to each patch's last update
                fresh = (last_update == min(self.patch_last_update[i], self.patch_last_update[j]))
                if valid_patches and fresh:
                    break
            self.merge(i, j)
            self.patch_last_update[i] = self.num_patches
            # add new items to queue
            for new_j in self.patch_adjacencies[i]:
                dist = self.compute_neg_dist(i, new_j)
                bisect.insort(self.queue, (dist, *reorder_pair(i, new_j), self.num_patches))
            if self.num_patches % 100 == 0:
                output.append((self.num_patches, self.construct_img()))

        self.patch_process = output
        return output

    def post_process(self):
        pass