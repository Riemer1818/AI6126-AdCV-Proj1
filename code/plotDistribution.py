import numpy as np

class FashionNet_Dataset():

    def __init__(self, root, txt, dataset):
        self.labels = [[] for _ in range(len(num_subattributes))]

        with open(txt) as f:
            for line in f:
                # make dummy label for test set
                if 'test' in txt:
                    for i in range(len(num_subattributes)):
                        self.labels[i].append(0)

        if 'test' not in txt:
            with open(txt.replace('.txt', '_attr.txt')) as f:
                for line in f:
                    attrs = line.split()
                    for i in range(len(num_subattributes)):
                        self.labels[i].append(int(attrs[i]))

    def __len__(self):
        return len(self.labels[0])

    def __getitem__(self, index):
        label = np.array([self.labels[i][index] for i in range(len(num_subattributes))])
        label = label.astype(np.float32)

        return label

def count_abundances():
    # Initialize the array with zeros
    counts = np.zeros((7, 6))

    # Automatically calculate bin counts and fill in the array
    for i, _ in enumerate(num_subattributes):
        bincount = np.bincount(trainset.labels[i])
        length = min(len(bincount), 7)  # Ensure we don't exceed the number of rows in thisOne
        counts[:length, i] = bincount[:length]

    relative_abundances = counts / counts.sum(axis=0)
    relative_abundances = 1 / relative_abundances

    return relative_abundances

if __name__ == '__main__':
    global num_subattributes
    num_subattributes = [7, 3, 3, 4, 6, 3]

    testlist = "/home/riemer/Downloads/FashionDataset/split/test.txt"
    trainlist = "/home/riemer/Downloads/FashionDataset/split/train.txt" 
    vallist = "/home/riemer/Downloads/FashionDataset/split/val.txt"

    global trainset, valset, testset
    trainset = FashionNet_Dataset("./FashionDataset", trainlist, "train")
    valset = FashionNet_Dataset("./FashionDataset", vallist, "val")
    testset = FashionNet_Dataset("./FashionDataset", testlist, "test")

    import matplotlib.pyplot as plt

    class_distribution = count_abundances()

    # Set the color map to include a different color for infinities
    cmap = plt.cm.get_cmap('viridis')
    cmap.set_bad('white')

    plt.imshow(class_distribution, cmap=cmap, interpolation='nearest')
    plt.colorbar()
    plt.savefig('./imgs/distribution.png')
    plt.show()
    

