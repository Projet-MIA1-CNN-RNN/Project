
with training_model:
    st.header("Training of the model")
    st.markdown('''Here you can train the model. Please note that is might be time consuming.
                During training you will have the mean loss for each epoch. When the training will 
                finish an aditionnal plot will show how the loss evolves ''')
    st.markdown("""Once you are have selected the desired parameters, you can train the model by selecting 'training mode' down below. 'Standby mode' is here by default to let you choose the parameters """)
    begin_train = st.selectbox("Select the mode you want: standby mode, training mode, stop mode", options = ["standby mode","training mode"],index=0)

    if begin_train=="training mode":
        list_epoch = []
        best_loss = 1
        train_loader_user = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,)),
                        lambda x: x>0,
                        lambda x: x.float(),
                ])),batch_size=batch_size_user, shuffle=True,pin_memory=True)
        
        for t in range(epoch_user):
            print(f"Epoch {t+1}\n-------------------------------")
            accuracy,best_loss = train_loop(train_loader_user, user_model_mnist, user_loss_fn, user_optimizer,best_loss)
            list_epoch.append(accuracy)
            print(f"Mean loss for this epoch: {accuracy}")
            print(f"Minimal loss reached so far: {best_loss}")
        print("Done!")
        
        fig,ax = plt.subplots()
        ax.plot(range(epoch_user),list_epoch)
        plt.xlabel("Number of the epoch")
        plt.ylabel("Loss")
        plt.title("Evolution of the loss during training")
        st.pyplot(fig)

    st.subheader("Saving the model")

    st.markdown(''' If you are satisfied with the best accuracy achieved by the model, you can save it.
    Please note that you can save only one model at the time.
    The weights of the model will be saved inside the 'mnist_user_model.pth' file. ''')

    st.markdown(""" Your custom model you will usable for generating images in the next section.""")
    save_model = st.selectbox("Do you want to save the model weights ?", options = ["Yes","No"],index=1)

    if save_model == "yes":
        PATH = './mnist_user_model.pth'
        torch.save(user_model_mnist.state_dict(), PATH)