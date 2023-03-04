from PixelCNN_MNIST_model import PixelCNN_MNIST,device

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model = model.to(device)
    list_loss = []
    print("test")
    for batch, (X, y) in enumerate(dataloader):
        target = Variable(X[:,:,:,:])
        X = X.to(device)
        target = target.to(device)
        # Compute prediction and loss
        pred = model(X)
        pred = pred[:,:,:,:] 
        #print(pred)
        
        loss = loss_fn(pred,target)
        list_loss.append(loss)
        # Backpropagation (always in three steps)
        optimizer.zero_grad() # a toujours mettre pour pas accumuler les gradients
        loss.backward()
        optimizer.step()
    return np.mean(list_loss)