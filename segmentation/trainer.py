import torch
import time

class Solver:
    def __init__(self, model_name, model, train_loader, val_loader, device, num_classes=35):
        self.model_name = model_name
        self.num_classes = num_classes
        self.model = model.to(device)
        self.device = device
        self.trainLoader = train_loader
        self.valLoader = val_loader
        self.trainSize = len(train_loader)*train_loader.batch_size
        self.valSize = len(val_loader)*val_loader.batch_size
        
    def train(self, criterion, optimizer, lr_scheduler=None, num_epochs=25):
        self.model.train()  # put model in training mode
        self.model.to(self.device) # ensure correct device

        train_loss_history = []
        val_loss_history = []

        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            total_load_time = 0
            total_backprop_time = 0

            train_loss = 0
            num_points = 0
            load_start_time = time.time()
            for idx, (images, targets) in enumerate(self.trainLoader):
                load_end_time = time.time()
                total_load_time += load_end_time - load_start_time
                # Get images and targets
                images = images.to(self.device)  # Load Images into correct device
                targets = targets.to(self.device)  # Load Targets into correct device

                # zero parameter gradients
                for param in self.model.parameters():
                    param.grad = None
                
                # Forward pass the network
                scores = self.model(images)

                loss_start_time = time.time()
                # scores = torch.nn.functional.softmax(scores, dim=1)
                
                # Calculate loss and backpropogate
                loss = criterion(scores, targets)
                loss.backward()
                optimizer.step()

                loss_end_time = time.time()
                total_backprop_time += loss_end_time - loss_start_time

                # Per iteration Logging functionality
                train_loss += loss
                pct_done = (idx+1)/len(self.trainLoader)*100
                print("\rTraining {0:0.2f}%: Loss {1:0.5f}".format(pct_done, train_loss/idx), end="")
                load_start_time = time.time()
            # Per Epoch Functionality
            if lr_scheduler:
                lr_scheduler.step()

            # Check Validation
            with torch.no_grad():
                self.model.eval()
                val_loss = 0
                for images, labels in self.valLoader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    scores = self.model(images)
                    val_loss += criterion(scores, labels)

            train_loss = train_loss / len(self.trainLoader)
            val_loss = val_loss / len(self.valLoader)
            val_loss_history.append(val_loss)
            train_loss_history.append(train_loss)
            epoch_end_time = time.time()
            elapsed_time = epoch_end_time - epoch_start_time
            print("\nEpoch {0} Completed in {1:0.2f} Seconds".format(epoch, elapsed_time))
            print("Training Loss: {0:0.5f} Validation Loss: {1:0.5f}".format(train_loss, val_loss))
            print("load time: ", total_load_time)
            print("loss time: ", total_backprop_time)
        return train_loss_history, val_loss_history
    
    def save_model(filename):
        torch.save(self.model, filename)