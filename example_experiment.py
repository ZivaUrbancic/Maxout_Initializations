###
# Prep work
###
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random
import itertools
from log_classes import *
exec(open("initialisation.py").read())
np.set_printoptions(threshold=np.inf)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


###
# Experiment setup
###
dataset = "mnist"
models = [ReLUNet3(input_dimension,10,10,output_dimension),
          ReLUNet3(input_dimension,9,9,output_dimension),
          ReLUNet3(input_dimension,8,8,output_dimension)]
optimisers = ["sgd", "adam"]
strats = [{"adjust_regions": False, "adjust_variance": False},
          {"adjust_regions": False, "adjust_variance": True},
          {"adjust_regions": True, "adjust_variance": True}]
num_runs = 24
num_epochs = 24
batch_size = 100



###
# Running experiments
###
experiment_number = str(random.randint(0,999999999))
FileLog = Log()
train_dataset, test_dataset, input_dimension, output_dimension = load_data(dataset)

for model_uncopied,optimiser_name,strat,run in itertools.product(models,optimisers,strats,range(num_runs)):
    runlog = RunLog(dataset,"small","relu",strat["adjust_regions"],strat["adjust_variance"],experiment_number=experiment_number)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    X, Y = sample_dataset(train_dataset, train_loader, data_sample_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    Xtest, Ytest = sample_dataset(test_dataset, test_loader, data_sample_size)

    model = deepcopy_network(model_uncopied,experiment_number)
    cost_vector = reinitialise_network(model, X, Y,
                                       return_cost_vector=True,
                                       adjust_regions=strat["adjust_regions"],
                                       adjust_variance=strat["adjust_variance"])
    cost_vector_test = reinitialise_network(model, Xtest, Ytest,
                                            return_cost_vector = True,
                                            adjust_regions = False,
                                            adjust_variance = False)
    model = model.to(device)
    runlog.record_cost_vector(0,0,[cost_vector,cost_vector_test])

    criterion = nn.CrossEntroyLoss()
    optimiser = load_optimiser(model,optimiser_name)

    n_total_steps = len(train_loader)

    imagesTest = torch.tensor([],dtype=torch.long)
    labelsTest = torch.tensor([],dtype=torch.long)
    for images, labels in test_loader:
        imagesTest = torch.cat((imagesTest,images))
        labelsTest = torch.cat((labelsTest,labels))
    imagesTest = imagesTest.to(device)
    labelsTest = labelsTest.to(device)

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            if (i+1) % 60 == 0:
                # logging accuracies and losses
                with torch.no_grad():
                    # computing train accuracy
                    n_correct = 0
                    n_samples = 0
                    n_class_correct = [0 for i in range(10)]
                    n_class_samples = [0 for i in range(10)]
                    for images, labels in train_loader:
                        images = images.to(device)
                        labels = labels.to(device)
                        outputs = model(images)
                        _, predicted = torch.max(outputs, 1)
                        n_samples += labels.size(0)
                        n_correct += (predicted == labels).sum().item()
                        for j in range(labels.size(0)):
                            label = labels[j]
                            pred = predicted[j]
                            n_class_correct[label] += (label == pred)
                            n_class_samples[label] += 1
                    acc_train = [round(n_correct / n_samples,3)]
                    acc_train += [round(n_class_correct[j] / n_class_samples[j],3) for j in range(10)]

                    # computing test accuracy
                    n_correct = 0
                    n_samples = 0
                    n_class_correct = [0 for i in range(10)]
                    n_class_samples = [0 for i in range(10)]
                    for images, labels in test_loader:
                        images = images.to(device)
                        labels = labels.to(device)
                        outputs = model(images)
                        _, predicted = torch.max(outputs, 1)
                        n_samples += labels.size(0)
                        n_correct += (predicted == labels).sum().item()
                        for j in range(labels.size(0)):
                            label = labels[j]
                            pred = predicted[j]
                            n_class_correct[label] += (label == pred)
                            n_class_samples[label] += 1
                    acc_test = [round(n_correct / n_samples,3)]
                    acc_test += [round(n_class_correct[j] / n_class_samples[j],3) for j in range(10)]

                    # recording accuracies
                    runlog.record_accuracies(epoch,i+1,[acc_train,acc_test])

                    # computing test loss
                    outputsTest = model(imagesTest)
                    lossTest = criterion(outputsTest, labelsTest)
                    # recording losses
                    runlog.record_losses(epoch,i+1,[round(loss.item(),3),round(lossTest.item(),3)])


            if (i+1) % 150 == 0:
                # logging region costs
                print("run ",run+1,"/ ",num_runs,";  epoch ",epoch+1," / ",num_epochs)
                print("    logging linear regions at step ",i+1)
                cost_vector_train = reinitialise_network(model, X, Y, True, False, False)
                cost_vector_test = reinitialise_network(model, Xtest, Ytest, True, False, False)
                runlog.record_cost_vector(epoch,i+1,[cost_vector_train,cost_vector_test])
                ModelNetworks.append(deepcopy_network(model))


            # Backward and optimize
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

    FileLog.add_runlog(runlog)

FileLog.save(experiment_number)
torch.save(ModelNetworks,str(experiment_number)+'.pt')
print("Experiment number: ", experiment_number)
