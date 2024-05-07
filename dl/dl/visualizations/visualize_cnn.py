import matplotlib.pyplot as plt

######################### READ DATA #####################################
with open('data/results/train_accuracies_initial_cnn.txt', 'r') as f: 
    lines = f.read().splitlines()
    train_acc = [float(line) for i,line in enumerate(lines)]
    
with open('data/results/validation_accuracies_initial_cnn.txt', 'r') as f: 
    lines = f.read().splitlines()
    val_acc = [float(line) for i,line in enumerate(lines)] 
    
with open('data/results/train_losses_initial_cnn.txt', 'r') as f: 
    lines = f.read().splitlines()
    train_loss = [float(line) for i,line in enumerate(lines)]

with open('data/results/validation_losses_initial_cnn.txt', 'r') as f: 
    lines = f.read().splitlines()
    val_loss = [float(line) for i,line in enumerate(lines)]


x = list(range(1,15))

######################### VISUALIZE ACCURACY #####################################

plt.figure(figsize=(10, 5))
plt.title("Training and Validation Accuracy initial CNN")
plt.xlabel('Epoch')
plt.ylabel("Accuracy")
plt.plot(x, train_acc, label = 'Training Accuracy')
plt.plot(x, val_acc, label = 'Validation Accuracy')
plt.legend()
plt.savefig('data/results/figures/val_and_train_acc_cnn.png')

######################### VISUALIZE LOSS #####################################

plt.figure(figsize=(10, 5))
plt.title("Training and Validation Loss initial CNN")
plt.xlabel('Epoch')
plt.ylabel("Loss")
minposs = 14
plt.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')
plt.plot(x, train_loss, label = 'Training Loss')
plt.plot(x, val_loss, label = 'Validation Loss')
plt.legend()
plt.savefig('data/results/figures/val_and_train_loss_cnn.png')

######################### VISUALIZE F1 #####################################
# val_f1 = [float(x)*100 for x in val_f1]
# train_f1 = [float(x)*100 for x in train_f1]

# plt.figure(figsize=(10, 5))
# plt.title("Training and Validation F1 AlexNet")
# plt.xlabel('Epoch')
# plt.ylabel("F1")
# plt.plot(x, train_f1, label = 'Training F1 score')
# plt.plot(x, val_f1, label = 'Validation F1 score')
# plt.legend()
# plt.savefig('dl_test/plots/val_and_train_f1.png')