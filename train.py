import torch
from utils import EarlyStopping

def train(model, train_loader, valid_loader, criterion, optimizer, scheduler, device ,num_train_data, num_valid_data, saved_model, num_epochs=100, do_early_stopping =True):
    print('==> Training model..')
    if do_early_stopping:
        early_stopping = EarlyStopping(saved_model = saved_model, patience = 2, verbose =1 )

    model.to(device)
    for epoch in range(num_epochs):
        running_loss = 0.0
        running_corrects = 0
        val_running_loss = 0.0
        val_running_correct =0
        
        model.train()
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                loss.backward()
                optimizer.step()
                
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
        scheduler.step()
        epoch_loss = running_loss / num_train_data
        epoch_acc = running_corrects.double() / num_train_data * 100
        
        #validate!

        model.eval()
        with torch.no_grad():

            for val_inputs, val_labels in valid_loader:
                val_inputs = val_inputs.to(device)
                val_labels = val_labels.to(device)
                val_outputs = model(val_inputs)
                val_loss = criterion(val_outputs, val_labels)

                _, val_preds = torch.max(val_outputs, 1)
                val_running_loss += val_loss.item()
                val_running_correct += torch.sum(val_preds == val_labels.data)
            
            val_epoch_loss = val_running_loss / num_valid_data
            val_epoch_acc = val_running_correct.double() / num_valid_data * 100

            if do_early_stopping:
                if early_stopping.validate(val_epoch_loss, model):
                    break

        
        print("===================================================")
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print("training loss: {:.5f}, acc: {:5f}".format(epoch_loss, epoch_acc))
        print("validation loss: {:.5f}, acc: {:5f}".format(val_epoch_loss, val_epoch_acc))

        if epoch == num_epochs-1:
            torch.save(model.state_dict(), '{}_best.pkl'.format(saved_model)) #if early stopping didn't occur
