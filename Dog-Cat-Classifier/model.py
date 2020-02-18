from utils import *


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        #self.n_feature = n_feature
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=50, kernel_size=5)
        self.conv2 = nn.Conv2d(50, 100, kernel_size=7)
        self.fc1 = nn.Linear(100 * 12 * 12, 120)
        self.fc2 = nn.Linear(120, 100)
        self.fc3 = nn.Linear(100, 2)
        
    def forward(self, x, verbose=False):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = x.view(-1, 100 * 12 * 12)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.sigmoid(x)
        return x
    
class CNN2(nn.Module):
    def __init__(self):
        super(CNN2, self).__init__()
        #self.n_feature = n_feature
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=50, kernel_size=5)
        self.conv2 = nn.Conv2d(50, 100, kernel_size=5)
        self.conv3 = nn.Conv2d(100, 80, kernel_size=7)
        self.conv4 = nn.Conv2d(80, 100, kernel_size=7)
        self.fc1 = nn.Linear(100 * 8 * 8, 120)
        self.fc2 = nn.Linear(120, 100)
        self.fc3 = nn.Linear(100, 2)
        
    def forward(self, x, verbose=False):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = x.view(-1, 100 * 8 * 8)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.sigmoid(x)
        return x