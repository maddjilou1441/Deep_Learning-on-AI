from utils import *


class DatasetCatDog(data.Dataset):
    def __init__(self, root, transform=None):
        images = []
        for filename in os.listdir(root):
            if filename.endswith('jpg'):
                images.append('{}'.format(filename))

        self.root = root
        self.imgs = images
        self.transform = transform
        #self.train_path = path + '/*.jpg'

    def __getitem__(self, index):
        filename = self.imgs[index]
        img = Image.open(os.path.join(self.root, filename))
        if self.transform is not None:
            img = self.transform(img)
        return img, filename

    def __len__(self):
        return len(self.imgs)
    
    def splitData(self, train_path, shuffle_data = True):
        
        # read addresses and labels from the 'train' folder
        addrs = glob.glob(train_path)
        labels = [ [1,0] if 'cat' in addr else [0,1] for addr in addrs]  # 1 = Cat, 0 = Dog
        # to shuffle data
        if shuffle_data:
            c = list(zip(addrs, labels))
            shuffle(c)
            addrs, labels = zip(*c)
            #print(labels[0:10])

        # Divide the hata into 60% train, 20% validation, and 20% test
        train_addrs = addrs[0:int(0.6*len(addrs))]
        train_labels = labels[0:int(0.6*len(labels))]
        #train_addrs.size

        val_addrs = addrs[int(0.6*len(addrs)):int(0.8*len(addrs))]
        val_labels = labels[int(0.6*len(addrs)):int(0.8*len(addrs))]

        test_addrs = addrs[int(0.8*len(addrs)):]
        test_labels = labels[int(0.8*len(labels)):]

        return train_addrs, train_labels, test_addrs, val_addrs, labels
    
    def resizeImage(self, file, labels, n):
        data = []
        for i in range(len(file[:n])):
            # read an image and resize to (64, 64)
            # cv2 load images as BGR, convert it to RGB
            addr = file[i]
            img = cv2.imread(addr)
            img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_CUBIC)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            data.append([np.array(img), np.array(labels[i])])
        shuffle(data)
        #np.save(filename+'.npy', data)
        return data