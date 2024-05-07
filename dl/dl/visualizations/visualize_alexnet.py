import matplotlib.pyplot as plt


if __name__ == '__main__':
    ######################### READ DATA #####################################
    with open('data/results/accuracy_alexnet.txt', 'r') as f: 
        lines = f.read().splitlines()
        train_acc = [line for i,line in enumerate(lines) if i%2 == 0]
        val_acc = [line for i,line in enumerate(lines) if i%2 != 0] 
        
    with open('data/results/log_loss_alexnet.txt', 'r') as f: 
        lines = f.read().splitlines()
        train_loss = [line for i,line in enumerate(lines) if i%2 == 0]
        val_loss = [line for i,line in enumerate(lines) if i%2 != 0] 
        
    # with open('data/results/f1.txt', 'r') as f: 
    #     lines = f.read().splitlines()
    #     train_f1 = [line for i,line in enumerate(lines) if i%2 == 0]
    #     val_f1 = [line for i,line in enumerate(lines) if i%2 != 0] 
        
    x = list(range(1,19))

    ######################### VISUALIZE ACCURACY #####################################
    val_acc = [float(x)*100 for x in val_acc]
    train_acc = [float(x)*100 for x in train_acc]

    plt.figure(figsize=(10, 5))
    plt.title("Training and Validation Accuracy AlexNet")
    plt.xlabel('Epoch')
    plt.ylabel("Accuracy")
    plt.plot(x, train_acc, label = 'Training Accuracy')
    plt.plot(x, val_acc, label = 'Validation Accuracy')
    plt.legend()
    plt.savefig('data/results/figures/val_and_train_acc_alexnet.png')

    ######################### VISUALIZE LOSS #####################################
    val_loss = [float(x) for x in val_loss]
    train_loss = [float(x) for x in train_loss]

    plt.figure(figsize=(10, 5))
    plt.title("Training and Validation Loss AlexNet")
    plt.xlabel('Epoch')
    plt.ylabel("Loss")
    minposs = 18
    plt.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')
    plt.plot(x, train_loss, label = 'Training Loss')
    plt.plot(x, val_loss, label = 'Validation Loss')
    plt.legend()
    plt.savefig('data/results/figures/val_and_train_loss_alexnet.png')

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
    # plt.savefig('data/results/figures/val_and_train_f1_alexnet.png')