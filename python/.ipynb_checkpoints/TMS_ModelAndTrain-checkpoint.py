#Script that declares our GNN model, loads our files, and then trains it. Right now it isn't crazy configurable from batch.
#In the near future, will break this into a python library so parts can be edited individually, but for dev purposes nice to have stuff here. 

#Kieran Wall - University of Virginia - August 2025
#I apologize to any CS folks who may have to read this

#Run - python3 ModelAndTrain "graphs directory" "weight save directory"

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
#Imports
import numpy as np
import matplotlib.pyplot as plt
import awkward as ak
import torch
from scipy.spatial import Delaunay
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch.utils.data import ConcatDataset
import os
import glob
import torch.nn.functional as F
from torch_geometric.nn import GCNConv as gcn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch_geometric.nn import EdgeConv 
import csv
import sys

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
#Functions

#Models:
#Arthur1

class Arthur1(torch.nn.Module):
    def __init__(self, in_channels, hc1, hc2, hc3, fc1, fc2, fc3, out_channels): # we define all of the convolutions and layers here
        super(Arthur1, self).__init__()
        self.convolution1 = gcn(in_channels, hc1) 
        self.bn1 = torch.nn.BatchNorm1d(hc1)
        self.convolution2 = gcn(hc1, hc2) 
        self.bn2 = torch.nn.BatchNorm1d(hc2)
        self.convolution3 = gcn(hc2, hc3)
        self.bn3 = torch.nn.BatchNorm1d(hc3)
        self.linear1 = torch.nn.Linear(hc3, fc1) 
        self.linear2 = torch.nn.Linear(fc1, fc2)
        self.linear3 = torch.nn.Linear(fc2, fc3)
        self.linear4 = torch.nn.Linear(fc3, out_channels)

        self.parameter_init()

    def forward(self, x, edge_index): #then apply them in the forward defintion. 
        x = self.convolution1(x, edge_index) #apply first convolution. Here we are actually applying the inputs. 
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        
        x = self.convolution2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        
        x = self.convolution3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)

        x = self.linear1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        
        x = self.linear2(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        
        x = self.linear3(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        
        x = self.linear4(x)
        return(x) #going to change the model to apply the log softmax at inference. 

    def parameter_init(self):
        for module in self.modules():
            if isinstance(module, (torch.nn.Linear)):
                torch.nn.init.kaiming_uniform_(module.weight, nonlinearity='relu') #set weights to random from uniform weighted on Relu
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias) #set biases to zero (if there are any)

class BerthaStatic(torch.nn.Module):
    def __init__(self, in_channels, hc1, hc2, hc3, fc1, fc2, fc3, out_channels): # we define all of the convolutions and layers here
        super(BerthaStatic, self).__init__()
        
        self.mlp1 = torch.nn.Sequential(
            torch.nn.Linear(in_channels * 2, hc1),
            torch.nn.ReLU(),
            torch.nn.Linear(hc1, hc1)
        )
        self.mlp2 = torch.nn.Sequential(
            torch.nn.Linear(hc1 * 2, hc2),
            torch.nn.ReLU(),
            torch.nn.Linear(hc2, hc2)
        ) 
        self.mlp3 = torch.nn.Sequential(
            torch.nn.Linear(hc2 * 2 , hc3),
            torch.nn.ReLU(),
            torch.nn.Linear(hc3, hc3)
        )  
        
        self.edgeconv1 = EdgeConv(self.mlp1, aggr='max')
        self.bn1 = torch.nn.BatchNorm1d(hc1)
        
        self.edgeconv2 = EdgeConv(self.mlp2, aggr='max') 
        self.bn2 = torch.nn.BatchNorm1d(hc2)
        
        self.edgeconv3 = EdgeConv(self.mlp3, aggr='max')
        self.bn3 = torch.nn.BatchNorm1d(hc3)
        
        self.linear1 = torch.nn.Linear(hc3, fc1) 
        self.linear2 = torch.nn.Linear(fc1, fc2)
        self.linear3 = torch.nn.Linear(fc2, fc3)
        self.linear4 = torch.nn.Linear(fc3, out_channels)

        self.parameter_init()

    def forward(self, x, edge_index): #then apply them in the forward defintion. 
        x = self.edgeconv1(x, edge_index) #apply first convolution. Here we are actually applying the inputs. 
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        x = self.edgeconv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        x = self.edgeconv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.linear1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        x = self.linear2(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        x = self.linear3(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        x = self.linear4(x)
        return(x) #going to change the model to apply the log softmax at inference. 

    def parameter_init(self):
        for module in self.modules():
            if isinstance(module, (torch.nn.Linear)):
                torch.nn.init.kaiming_uniform_(module.weight, nonlinearity='relu') #set weights to random from uniform weighted on Relu
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias) #set biases to zero (if there are any)


def run_train(model, training_loader_, validation_loader_, num_iterations=100, 
              log_dir='/sdf/data/neutrino/summer25/ktwall/Arthur/logs_Arthur1', 
              log_prefix='FC_run1', optimizer='SGD', lr=0.001):

    # Detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()
    print(f"Using device: {device} ({num_gpus} GPUs available)")

    # Multi-GPU setup if available - will typically request 2, if I don't get 2 then it'll fall back to previous way.  
    if device.type == "cuda" and num_gpus > 1:
        print("Wrapping model with DataParallel...")
        model = torch.nn.DataParallel(model)

    # Move model to device (after wrapping, or just alone if it hasn't been wrapped)
    model = model.to(device)

    # Loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer_fn = getattr(torch.optim, optimizer)
    optimizer = optimizer_fn(model.parameters(), lr=lr)

    iteration = 0
    epoch_at_valid = []
    validation_acc = []

    #adding a step to make the log. directory if it does not exist. 
    os.makedirs(log_dir, exist_ok=True)
    train_log_name = f'{log_dir}/{log_prefix}train.csv'
    val_log_name = f'{log_dir}/{log_prefix}val.csv'
    
    with open(train_log_name, 'w', newline='') as trainfile, \
         open(val_log_name, 'w', newline='') as valfile:
        train_writer = csv.writer(trainfile)
        val_writer = csv.writer(valfile)
        train_writer.writerow(['iter', 'epoch', 'loss'])
        val_writer.writerow(['iter', 'epoch', 'loss', 'accuracy'])

        max_epochs = 40

        epoch = 0
        while epoch < max_epochs:
            model.train()
            epoch += 1 
            for training_data in training_loader_:
                training_data = training_data.to(device)
                
                # Training step
                optimizer.zero_grad()
                loss = criterion(model(training_data.x, training_data.edge_index), training_data.y)
                loss.backward()
                optimizer.step()

                train_writer.writerow([iteration, iteration/len(training_loader_), loss.item()])
                iteration += 1

            # Do a validation every epoch (its just easier)
            model.eval()
            with torch.no_grad():
                total_correct = 0
                total_samples = 0
                total_loss = 0.0
                for validation_data in validation_loader_:
                    validation_data = validation_data.to(device)
                    out = model(validation_data.x, validation_data.edge_index)
                    val_loss = criterion(out, validation_data.y)
                    total_loss += val_loss.item()
                    n_correct = torch.sum(out.argmax(dim=1) == validation_data.y)
                    total_correct += n_correct.item()
                    total_samples += len(validation_data.y)
                
                acc = total_correct / total_samples
                avg_loss = total_loss / len(validation_loader_) 
                print(f"Validation at epoch {epoch}, "
                              f"accuracy = {acc:.4f}, loss = {avg_loss:.4f}")
                
                val_writer.writerow([iteration, epoch, avg_loss, acc]) 


#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
#Main Function - Take arguments, does useful printing

def main():
    #not a ton of stuff going on here command line wise since this is mostly for development. 
    graph_file_directory = sys.argv[1] #load in the directory from which to pull our graphs
    weights_dir = sys.argv[2]
    graph_file_paths = glob.glob(os.path.join(graph_file_directory, "*.pt")) 
    #Load in our files - this will take quite a bit!!
    SEED = 2026
    np.random.seed(SEED)    # Setting the seed for reproducibility
    torch.manual_seed(SEED) 
    
    #Lets create a training set 
    train_file_paths = graph_file_paths[0:90]
    training_datasets_list = []
    for i, train_path in enumerate(train_file_paths):
        training_datasets_list.append(torch.load(train_path))
        if (i % 10 == 0):
            print(f'training loaded {i}')
    
    training_dataset = ConcatDataset(training_datasets_list)
    print('loaded training dataset')

    #Validation set that contains 5 files worth of graphs
    validation_file_paths = graph_file_paths[90:95]
    validation_datasets_list = []
    for validation_path in validation_file_paths:
        validation_datasets_list.append(torch.load(validation_path))

    validation_dataset = ConcatDataset(validation_datasets_list)
    print('loaded validation dataset')

    #Testing set that contains 5 files worth of graphs
    testing_file_paths = graph_file_paths[95:100]
    testing_datasets_list = []
    for testing_path in testing_file_paths:
        testing_datasets_list.append(torch.load(testing_path))

    testing_dataset = ConcatDataset(testing_datasets_list)
    print('loaded testing dataset')


    #Let's create our data loaders here
    batch = 1000 #may need to decrease batch size.
    #train
    training_data_loader = DataLoader(training_dataset, batch_size = batch, drop_last = True, shuffle=True, num_workers=2)
    print(f'Training on {len(training_data_loader)*training_data_loader.batch_size} total graphs')

    #validate
    validation_data_loader = DataLoader(validation_dataset, batch_size = batch, drop_last = True, shuffle=False)
    print(f'Validating on {len(validation_data_loader)*validation_data_loader.batch_size} total graphs')

    #test
    testing_data_loader = DataLoader(testing_dataset, batch_size = batch, drop_last = True, shuffle=True)
    print(f'Training on {len(testing_data_loader)*testing_data_loader.batch_size} total graphs')


    #Now ready to run training
    #declare the model
    selected_model = BerthaStatic(in_channels = 3, hc1 = 20, hc2 = 40, hc3 = 50, fc1 = 25, fc2 = 12, fc3 = 6, out_channels = 2)
    #pull validation information from the training loop
    print('about to start training')
    run_train(selected_model, training_data_loader, validation_data_loader, num_iterations = 50000, optimizer='Adam', log_dir = '/sdf/data/neutrino/summer25/ktwall/logs_Bertha', log_prefix= 'FC-BerthaStatic-GPU-batch1000-p02')
    #now lets save the weights, we will be doing plotting etc elsewhere. 
    model_weights = selected_model.state_dict()
    torch.save(model_weights, weights_dir + 'FC-BerthaStatic-GPU-train90-batch1000-p02.pth')
    
main()
    

    
























