## Running YOLO Folder

### efficient_run_train_test.py:

Allows for running multiple trials of YOLO with different combinations of training, supplemental, and validation data (this is performed by creating a Dataset object based on file path conventions)

### run_save_train_test.py:

Takes in arguments for training, supplemental, validation data, and key hyperparameters (epochs, batch size, GPU device), makes a data file, and runs the training and testing scripts (within the YOLO folder).

### train_mixed_batch.py:

Similar to train.py in YOLO with slight changes to introduce mixed batch training (which ensures a fixed batch size of supplemental vs training data in a given training batch)
