import matplotlib.pyplot as plt
def plot_history(history, model_title):

    train_loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    train_acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    # Sample data (replace with your actual data)
    epochs = range(1, len(train_loss) + 1)
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), sharex=True)

    # Plot loss on the upper subplot
    ax1.set_ylabel('Loss', color='tab:red')
    ax1.plot(epochs, train_loss, color='tab:blue', label='Training Loss')
    ax1.plot(epochs, val_loss, color='tab:red', label='Validation Loss')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    ax1.legend(loc='upper left')

    # Plot accuracy on the lower subplot
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy', color='tab:green')
    ax2.plot(epochs, [acc * 100 for acc in train_acc], color='tab:blue', label='Training Accuracy (%)')
    ax2.plot(epochs, [acc * 100 for acc in val_acc], color='tab:red', label='Validation Accuracy (%)')
    ax2.tick_params(axis='y', labelcolor='tab:green')
    ax2.set_ylim(0, 100)
    ax2.legend(loc='upper left')

    ax2.yaxis.set_label_position('right')  # Set the y-axis label to the right
    ax2.yaxis.set_ticks_position('right')  # Set the y-axis ticks to the right

    # Add a title
    plt.suptitle(f'Training and Validation Loss & Accuracy Over {epochs}: Model:{model_title}')

    # Display the plot
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()