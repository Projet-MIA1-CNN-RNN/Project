import streamlit as st
from MaskedConv import *
from PixelCNN_MNIST_model import PixelCNN_MNIST
from training_loop import train_loop


header = st.container()

description = st.container()

selecting_model = st.container()

training_model = st.container()

generating_image = st.container()

with header:
    st.title("PixelCNN and GatedPixelCNN application")
    st.markdown('''This application is here to illustrate our implementation of PixelCNN and 
    GatedPixelCNN ,two deep learning algorithm created for image generation an image completion. \n
For more detail about the algorithms, you can read our full report on the topic''')
    
with selecting_model:

    st.header("We will train a model for MNIST")
    st.markdown('''You can choose several parameters including for the network,
      the optimizer and the batch size''')
    st.markdown("Default chosen parameters are the one used in the PixelCNN original publication")
    st.markdown('''Warning: choosing a high number of residual block , epochs or a  large number for h may result in a larger computation time.
      The batch size will also impact computation time.''')
    
    sel_col, disp_col = st.columns(2)
    h_channels_user = st.slider("Choose the h parameter",min_value=2, max_value=256,value=32,step=2)
    nb_layer_blocks_user = st.slider("Choose the number of residual block",min_value=1, max_value=40,value=12,step=1)
    epoch_user = st.slider("Choose the number of epoch",min_value=1, max_value=30,value=5,step=1)

    batch_size_user = st.slider("Choose a batch size for the training of the model",min_value=2, max_value=512,value=16,step=2)
    optimizer_choice = st.selectbox("Choose an optimizer for the network", options=(["Adam","Adagrad", "Adamax","RMSprop","AdaDelta"]), index = 3)
    
    user_model_mnist = PixelCNN_MNIST(h_channels=h_channels_user, nb_layer_block = nb_layer_blocks_user)
    user_loss_fn = nn.BCEWithLogitsLoss()

    if optimizer_choice == "RMSprop":
        lr = st.selectbox("Choose a value for the learning rate", options=[0.00001, 0.0001, 0.001, 0.01, 0.1], index = 2)
        alpha =st.slider("Chose a value for the alpha parameter", min_value = 0.5, max_value=0.99,value=0.9, step = 0.01) 
        user_optimizer = torch.optim.RMSprop(user_model_mnist.parameters(),lr=lr,alpha=alpha)

    elif optimizer_choice == "Adagrad":
        lr = st.selectbox("Choose a value for the learning rate", options=[0.00001, 0.0001, 0.001, 0.01, 0.1], index = 2)
        st.markdown("The following parameters are optional")
        lr_decay = st.selectbox("Choose a value for the learning rate decay", options=[0.00001, 0.0001, 0.001, 0.01,0.1,0], index = 5)
        weight_decay = st.selectbox("choose a value for the weight decay", options=[0.00001, 0.0001, 0.001, 0.01,0.1,0], index = 5)
        user_optimizer = torch.optim.Adagrad(user_model_mnist.parameters(),lr=lr,lr_decay=lr_decay, weight_decay=weight_decay)

    elif optimizer_choice == "AdaDelta":
        lr = st.selectbox("Choose a value for the learning rate", options=[0.00001, 0.0001, 0.001, 0.01, 0.1], index = 2)
        st.markdown("The following parameters are optional")
        rho = st.slider("Choose a value for the rho parameter", min_value = 0.5, max_value=0.99,value=0.9, step = 0.01)
        weight_decay = st.selectbox("choose a value for the weight decay", options=[0.00001, 0.0001, 0.001, 0.01,0.1,0], index = 5)
        user_optimizer = torch.optim.Adadelta(user_model_mnist.parameters(),lr=lr,rho=rho, weight_decay= weight_decay)

    elif optimizer_choice == "Adam":
        lr = st.selectbox("Choose a value for the learning rate", options=[0.00001, 0.0001, 0.001, 0.01, 0.1], index = 2)
        st.markdown("The following parameter is optional")
        weight_decay = st.selectbox("choose a value for the weight decay", options=[0.00001, 0.0001, 0.001, 0.01,0.1,0], index = 5)
        user_optimizer = torch.optim.Adam(user_model_mnist.parameters(),lr=lr, weight_decay= weight_decay)

    else:
        lr = st.selectbox("Choose a value for the learning rate", options=[0.00001, 0.0001, 0.001, 0.01, 0.1], index = 2)
        st.markdown("The following parameter is optional")
        weight_decay = st.selectbox("choose a value for the weight decay", options=[0.00001, 0.0001, 0.001, 0.01,0.1,0], index = 5)
        user_optimizer = torch.optim.Adamax(user_model_mnist.parameters(),lr=lr, weight_decay= weight_decay)




with training_model:
    st.header("Training of the model")
    st.markdown('''Here you can train the model. Please note that is might be time consuming.
                During training you will have the mean loss for each epoch. When the training will 
                finish an aditionnal plot will show how the loss evolves ''')
    st.markdown("""Once you are have selected the desired parameters, you can train the model by selecting 'training mode' down below. 'Standby mode' is here by default to let you choose the parameters and stop algorithm if necessary """)
    begin_train = st.selectbox("Select the mode you want: standby mode or training mode", options = ["standby mode","training mode"],index=0)

    if begin_train=="training mode":
        list_epoch = []
        train_loader_user = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,)),
                        lambda x: x>0,
                        lambda x: x.float(),
                ])),batch_size=batch_size_user, shuffle=True,pin_memory=True)
        print(type(train_loader_user))
        for t in range(epoch_user):
            print(f"Epoch {t+1}\n-------------------------------")
            accuracy = train_loop(train_loader_user, user_model_mnist, user_loss_fn, user_optimizer)
            list_epoch.append(accuracy)
            print(accuracy)
        print("Done!")
        
        fig,ax = plt.subplot()
        ax.plot(range(epoch_user,list_epoch))
        plt.xlabel("Number of the epoch")
        plt.ylabel("Loss")
        plt.title("Evolution of the loss during training")
        st.pyplot(fig)

    st.subheader("Saving the model")

    st.markdown(''' If you are satisfied with the best accuracy achieved by the model, you can save it.
    Please note that you can save only one model at the time.
    The weights of the model will be saved inside the 'user_model_weights.pth' file. ''')

    st.markdown(""" Your custom model you will usable for generating images in the next section.""")
    save_model = st.selectbox("Do you want to save the model weights ?", options = ["Yes","No"],index=1)

    if save_model == "yes":
        PATH = './user_model_weights.pth'
        torch.save(user_model_mnist.state_dict(), PATH)