
from .data_collection import minibatch_idx_list, minibatch_idx
import numpy as np

def cross_validation(session, loss, optimize, save_op, restore_op, dct, validation_ratio=0.1, max_epochs=500, batch_size=64, epochs_early_stopping=25):
    '''
    Minimizes a loss function with cross-validation.
    See demo_cv.py for an example.
    '''

    # Split data training / validation
    nb_data = next(iter(dct.values())).shape[0]
    idx_test = minibatch_idx(int(np.ceil(nb_data*validation_ratio)), nb_data)
    idx_train = np.delete(np.arange(nb_data), idx_test)
    dct_test = {}
    dct_train = {}
    for key in dct:
        dct_test[key] = dct[key][idx_test,:]
        dct_train[key] = dct[key][idx_train,:]

    # Save current loss on validation set and current variables
    last_loss_ok = session.run(loss, dct_test)
    session.run(save_op)
    epochs_no_decrease = 0

    loss_train_history = np.zeros((max_epochs,1))
    loss_test_history = np.zeros((max_epochs,1))

    # Learn
    for epoch in range(max_epochs):
        # Perform one epoch of gradient descent on the whole training dataset, divided into mini-batches
        dct_batch = {}
        for batch_idx in minibatch_idx_list(batch_size, idx_train.shape[0]):
            for key in dct:
                dct_batch[key] = dct[key][idx_train[batch_idx],:]
            session.run(optimize, dct_batch)

        # Check if loss is decreasing, if so save variables
        current_loss = session.run(loss, dct_test)
        if current_loss < last_loss_ok:
            epochs_no_decrease = 0
            last_loss_ok = current_loss
            session.run(save_op)
        else:
            epochs_no_decrease += 1

        loss_train_history[epoch] = session.run(loss, dct_train)
        loss_test_history[epoch] = current_loss

        # Terminal condition and restore variables
        if epochs_no_decrease >= epochs_early_stopping:
            session.run(restore_op)
            break

    return loss_test_history[:epoch], loss_train_history[:epoch]
