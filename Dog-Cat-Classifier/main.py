from utils import *
from dataset import *
from model import *



def main(path):
    data = DatasetCatDog(path)
    train_path = path + '/*.jpg'
    train_addrs, train_labels, test_addrs, val_addrs, labels = data.splitData(train_path)
    
    
    train_data = data.resizeImage(train_addrs, train_labels, 1000)
    test_data = data.resizeImage(test_addrs, labels, 500)
    val_data = data.resizeImage(val_addrs, labels, 500)
    
    X = np.array([i[0] for i in train_data]).reshape(-1,64,64,3)
    X = Variable(torch.Tensor(X))
    X = X.reshape(-1,64,64,3)
    X = X.permute(0,3,1,2)

    Y = np.array([i[1] for i in train_data])
    target = Variable(torch.Tensor(Y))
    target = target.type(torch.LongTensor)
    
    test = np.array([i[0] for i in test_data]).reshape(-1,64,64,3)
    test = Variable(torch.Tensor(test))
    test = test.reshape(-1,64,64,3)
    test = test.permute(0,3,1,2)

    tlabels = np.array([i[1] for i in test_data])
    tlabels = Variable(torch.Tensor(tlabels))
    tlabels = tlabels.type(torch.long)
    
    #cnn = CNN()
    cnn = CNN2()
    
    criterian = nn.CrossEntropyLoss()
    optimizer = optim.SGD(cnn.parameters(), lr = 0.0001, momentum = 0.9)
    
    for epoch in range(10):
        running_loss  = 0.0
        optimizer.zero_grad() #zero the parameter gradients
        output = cnn(X)

        loss = criterian(output, torch.max(target, 1)[1])

        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        print('Epoch: ', epoch, ' loss: ', running_loss)
    
    print('**********************************************************************')
    
    correct = 0
    total = 0
    with torch.no_grad():
        for data in zip(X,target):
            images, labels = data
            images = images.reshape(1,3,64,64)
            outputs = cnn(images)
            _, predicted = torch.max(outputs, 1)
            #total += labels.size(0)
            if((predicted == 0 and labels[0] == 1) or (predicted == 1 and labels[1]==1) ):
                correct+=1
            #correct += (predicted == labels).sum().item()
            #print(outputs,labels)
    total = X.shape[0]
    print('Train accuracy of the network on the ' + str(total) +  ' train images: %f %%' % (
        100 * (correct*1.0) / total) )
    print(correct, total)
    
    print('**********************************************************************')
    
    correct = 0
    total = 0
    with torch.no_grad():
        for data in zip(test,tlabels):
            images, labels = data
            images = images.reshape(1,3,64,64)
            outputs = cnn(images)
            _, predicted = torch.max(outputs, 1)
            #total += labels.size(0)
            if((predicted == 0 and labels[0] == 1) or (predicted == 1 and labels[1]==1) ):
                correct += 1

    total = test.shape[0]
    print('Test accuracy of the network on the ' + str(total) +  ' test images: %f %%' % (
        100 * (correct*1.0) / total) )
    print(correct, total)
    

if __name__ == '__main__':
    path = '/home/aims/Documents/Dog-Cat-Classifier/train'
    main(path)
    
        
        
    
