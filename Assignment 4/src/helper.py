import torch
def read_file(path : str) -> str:
    """
    It opens a file, reads it, and returns the contents

    :param path: The path to the file you want to read
    :return: The text of the file.
    """
    with open(path) as f:
        txt = f.read()
    return txt

def save_state(model_save_path, model, optimizer, total_batches, total_count, validation_accuracy, training_accuracy, train_loss, validation_loss):
    """
    Save the model's state, the optimizer's state, the total number of batches, the total number of
    images, the validation accuracy, the training accuracy, the training loss, and the validation loss

    :param model_save_path: The path to save the model to
    :param model: the model object
    :param optimizer: the optimizer used to train the model
    :param total_batches: The total number of batches that have been trained on
    :param total_count: The total number of images that have been trained on
    :param validation_accuracy: The accuracy of the model on the validation set
    :param training_accuracy: The accuracy of the model on the training set
    :param train_loss: The loss of the model on the training set
    :param validation_loss: The loss of the model on the validation set
    """    
    state = {
    "total_batches": total_batches,
    "total_count": total_count,
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "validation_acc": validation_accuracy,
    "train_acc": training_accuracy,
    "train_loss": train_loss,
    "valid_loss": validation_loss
    }
    torch.save(obj=state, f=model_save_path)

def load_state(save_path, model, optimizer, total_batches, total_count, validation_accuracy, training_accuracy, train_loss, validation_loss):
    """
    It loads the model, optimizer, and all the other variables that we need.

    :param save_path: the path to the saved model
    :param model: the model
    :param optimizer: the optimizer used to train the model
    :param total_batches: The total number of batches that have been trained on
    :param total_count: total number of batches trained on
    :param validation_accuracy: A list of the validation accuracy at each epoch
    :param training_accuracy: A list of the training accuracy at each epoch
    :param train_loss: A list of the training loss at each epoch
    :param validation_loss: A list of the validation loss at each epoch
    :return: The model, optimizer, total_batches, total_count, validation_accuracy, training_accuracy,
    train_loss, validation_loss
    """
    state = torch.load(save_path)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    total_batches = state["total_batches"]
    total_count = state["total_count"]
    validation_accuracy = state["validation_acc"]
    training_accuracy = state["train_acc"]
    train_loss = state["train_loss"]
    validation_loss = state["valid_loss"]
    return model, optimizer, total_batches, total_count, validation_accuracy, training_accuracy, train_loss, validation_loss

def counter(text : str):
  """
  It counts the number of sentences, the average length of a sentence, and the number of characters in
  a text
  
  :param text: the text to be analyzed
  """
  count_sentence = text.count('?')
  if count_sentence != 0:
    print("Num sentences", count_sentence)
    counter = 0
    reset_counter = 1
    for i in text:
      if i != '?':
        reset_counter += 1
      else:
        counter += reset_counter
        reset_counter = 1
    print("Average question length:", round(counter/count_sentence))
  else:
    counter = 0
    reset_counter = 0
    for i in text:
      if i != '\n':
        reset_counter += 1
      else:
        counter += reset_counter
        reset_counter = 0
    print("Average answer length:", round(counter/len(text.split('\n'))))
  print("Num chars", len(text))