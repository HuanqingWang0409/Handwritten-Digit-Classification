
def preview(previewNum, data):
    '''
    Plot the digit images in the dataset, with a row of 5 images
    Input:
    previewNum: The number of images to preview
    data: The data set containing digit images
    Output:
    Plots of digit images
    '''
    nrows = previewNum//5
    ncols = 5
    for i in range(nrows*ncols):
        plt.subplot(nrows, ncols, i+1)
        plt.imshow(data[i].reshape((28,28)))
        plt.axis('off')
    plt.show()
    
    
    
    
    
    
def digit_image(digit,X,y,previewNum = 25):
    '''
    Plot a certain digit images in the data set
    Input:
    digit: The digit (0-9) image one would like to preview from a data set
    X: The data set
    y: The lebel set
    previewNum: The number of images to preview
    Output:
    Plots of digit images
    '''
    data = Xtrain[np.where(ytrain==digit)]
    preview(previewNum,data)
    
    
    
    
    
    
       
    
def pca_trans(X, n_components):
    '''
    Perform Principal Component Analysis on data to transform a high number of components to a given number of components
    Input:
    X: Data with a high number of components
    n_components: Number of components after transformation
    Output:
    pca: The PCA object
    X_pca: Data after PCA transformation
    '''
    X = StandardScaler().fit_transform(X)
    pca = PCA(n_components)
    X_pca = pca.fit_transform(X)
    return pca, X_pca

def plot_components(X, y):
    '''
    Plot the PCA-transformed data in a 2D graph to view PCA seperation status
    Input:
    X: The PCA transformed data set containing digit images with 2 components
    y: The data set containing digit values
    Output:
    A plot of the PCA-transformed data in a 2D graph with the correct digit values
    '''
    X = StandardScaler().fit_transform(X)
    plt.figure(figsize=(12, 8))
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(y[i]), color=plt.cm.Set1(y[i]), fontdict={'size': 15})

    plt.xlim([np.min(X[:,0])*1.1, 1.1*np.max(X[:,0])])
    plt.ylim([1.1*np.min(X[:,1]),1.1*np.max(X[:,1])])
    plt.xticks([])
    plt.yticks([])
    plt.xlabel("First Principal Component")
    plt.ylabel("Second Principal Component")
    plt.title("PCA Transformed Data with 2 Components")
    plt.show()  
    
def random_forest(trainX, trainY, testX, testY, print_output=True):
    '''
    Fit a simple random forest classifier on the data and return the confusion matrix
    Input:
    trainX: The training data set containing digit images
    trainY: The training data set containing digit values
    testX: The test data set containing digit images
    testY: The test data set containing digit values
    print_output: True if the confusion matrix is printed
    Output:
    The percentage of corrected classified records in the test set
    '''

    rf = RandomForestClassifier()
    rf.fit(trainX, trainY)

    pred = rf.predict(testX)
    correct = accuracy_score(testY, pred)
    
    if print_output:
        confusion = confusion_matrix(testY, pred)
        plt.figure(figsize=(10,6))
        sns.heatmap(confusion, annot=True, cmap='bwr', linewidths=0.5)

        print(f'Train data set has {len(trainX)} records.')
        print(f'Test data set has {len(testX)} records.')
        print('Misclassification rate is {:2.2%}\n'.format(1-correct))
        print("Confusion Matrix:")
        print(confusion)
    
    return correct




class MNIST_Net(nn.Module):
    def __init__(self,pars):
        super(MNIST_Net, self).__init__()
                ks=pars.kernel_size
        ps=np.int32(pars.pool_size)
        self.mid_layer=pars.mid_layer
        # Two successive convolutional layers.
        # Two pooling layers that come after convolutional layers.
        # Two dropout layers.
        self.conv1 = nn.Conv2d(1, 32, kernel_size=ks[0],padding=ks[0]//2)
        self.pool1=nn.MaxPool2d(kernel_size=[ps],stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=ks[1],padding=ks[1]//2)
        self.drop2 = nn.Dropout2d(pars.dropout)
        self.pool2=nn.MaxPool2d(kernel_size=2,stride=2)
        self.drop_final=nn.Dropout(pars.dropout)
       
        
        # Run the network one time on one dummy data point of the same 
        # dimension as the input images to get dimensions of fully connected 
        # layer that comes after second convolutional layers
        self.first=True
        if self.first:
            self.forward(torch.zeros((1,)+pars.inp_dim))
            
        # Setup the optimizer type and send it the parameters of the model
        if pars.minimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.parameters(), lr = pars.step_size)
        else:
            self.optimizer = torch.optim.SGD(self.parameters(), lr = pars.step_size)
         
        self.criterion=nn.CrossEntropyLoss()
        
    def forward(self, x):
        
        # Apply relu to a pooled conv1 layer.
        x = F.relu(self.pool1(self.conv1(x)))
        if self.first:
            print('conv1',x.shape)
        # Apply relu to a pooled conv2 layer with a drop layer inbetween.
        x = self.drop2(F.relu(self.pool2(self.conv2(x))))
        if self.first:
            print('conv2',x.shape)
        
        if self.first:
            self.first=False
            self.inp=x.shape[1]*x.shape[2]*x.shape[3]
            # Compute dimension of output of x and setup a fully connected layer with that input dim 
            # pars.mid_layer output dim. Then setup final 10 node output layer.
            print('input dimension to fc1',self.inp)
            if self.mid_layer is not None:
                self.fc1 = nn.Linear(self.inp, self.mid_layer)
                self.fc_final = nn.Linear(self.mid_layer, 10)
            else:
                self.fc1=nn.Identity()
                self.fc_final = nn.Linear(self.inp, 10)
            # Print out all network parameter shapes and compute total:
            tot_pars=0
            for k,p in self.named_parameters():
                tot_pars+=p.numel()
                print(k,p.shape)
            print('tot_pars',tot_pars)
        x = x.reshape(-1, self.inp)
        x = F.relu(self.fc1(x))
        x = self.drop_final(x)
        x = self.fc_final(x)
        return x
    
    # Run the network on the data, compute the loss, compute the predictions and compute classification rate/
    def get_acc_and_loss(self, data, targ):
        output = self.forward(data)
        loss = self.criterion(output, targ)
        pred = torch.max(output,1)[1]
        correct = torch.eq(pred,targ).sum()
        
        return loss,correct
    
    # Compute classification and loss and then do a gradient step on the loss.
    def run_grad(self,data,targ):
    
        loss, correct=self.get_acc_and_loss(data,targ)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss, correct

def run_epoch(net,epoch,train,pars,num=None,ttype="train"):
    
    if ttype=='train':
        t1=time.time()
        n=train[0].shape[0]
        if (num is not None):
            n=np.minimum(n,num)
        ii=np.array(np.arange(0,n,1))
        np.random.shuffle(ii)
        tr=train[0][ii]
        y=train[1][ii]
        train_loss=0; train_correct=0

        for j in trange(0,n,pars.batch_size):
                
                # Transfer the batch from cpu to gpu (or do nothing if you're on a cpu)
                data=torch.torch.from_numpy(tr[j:j+pars.batch_size]).to(pars.device)
                targ=torch.torch.from_numpy(y[j:j+pars.batch_size]).type(torch.long).to(pars.device)
                
                # Implement SGD step on batch
                loss, correct = net.run_grad(data,targ) 
                
                train_loss += loss.item()
                train_correct += correct.item()
                

        train_loss /= len(y)
        print('\nTraining set epoch {}: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.2f}%), error rate: {:.2f}% \n'.format(epoch,
            train_loss, train_correct, len(y),
            100. * train_correct / len(y), 100. * (1-train_correct / len(y))))
        error_train.append(100. * (1-train_correct / len(y)))
        

def net_test(net,val,pars,ttype='val'):
    net.eval()
    with torch.no_grad():
                test_loss = 0
                test_correct = 0
                vald=val[0]
                yval=val[1]
                for j in np.arange(0,len(yval),pars.batch_size):
                    data=torch.from_numpy(vald[j:j+pars.batch_size]).to(device)
                    targ = torch.from_numpy(yval[j:j+pars.batch_size]).type(torch.long).to(pars.device)
                    loss,correct=net.get_acc_and_loss(data,targ)

                    test_loss += loss.item()
                    test_correct += correct.item()

                test_loss /= len(yval)
                SSS='Validation'
                if (ttype=='test'):
                    SSS='Test'
                print('\n{} set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.2f}%), error rate: {:.2f}% \n'.format(SSS,
                    test_loss, test_correct, len(yval),
                    100. * test_correct / len(yval), 100. * (1-test_correct / len(yval))))
                error_val.append(100. * (1-test_correct / len(yval)))

# An object containing the relevant parameters for running the experiment.
class par(object):
    def __init__(self):
        self.batch_size=1000
        self.step_size=.001
        self.num_epochs=20
        self.numtrain=10000
        self.minimizer="Adam"
        self.data_set="mnist"
        self.model_name="model"
        self.dropout=0.
        self.dim=32
        self.pool_size=2
        self.kernel_size=5
        self.mid_layer=256
        self.use_gpu=False

    
    
