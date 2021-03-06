from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have column dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    dW = np.zeros(W.shape) # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            #here we want to seperate the ground truth class to have a score greater or equal to 1 than other scores
            #i.e. x*wj - x*wy +1
            if margin > 0:
                loss += margin
                dW[:, y[i]] += np.transpose(X[i])*-1
                dW[:, j] += np.transpose(X[i])

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss.
    loss += 0.5*reg * np.sum(W * W)
    dW += reg*W

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather than first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    return loss, dW



def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    scores = X.dot(W)
    num_classes = W.shape[1]
    num_train = X.shape[0]
    correct_class_score = scores[np.arange(num_train), y]
    correct_class_score = np.reshape(correct_class_score, (num_train, -1))
    margins = scores - correct_class_score +1.0
    margins[np.arange(num_train), y] = 0.0
    margins[margins <= 0] = 0.0
    loss += np.sum(margins)/num_train
    loss += 0.5*reg*np.sum(W**2)
    # scores = X.dot(W)
    # num_train = X.shape[0]
    # num_classes = W.shape[1]
    # scores_correct = scores[np.arange(num_train), y]
    # scores_correct = np.reshape(scores_correct, (num_train, 1))
    # margins = scores - scores_correct + 1.0
    # margins[np.arange(num_train), y] = 0.0
    # margins[margins <= 0] = 0.0
    # loss += np.sum(margins) / num_train
    # loss += 0.5 * reg * np.sum(W * W)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    dscore = np.zeros_like(margins)
    dscore[margins>0] = 1.0
    #find those entries that involved in the computation of loss
    temp = np.sum(dscore, axis=1)
    dscore[np.arange(num_train),y] -= temp
    #noted that each entry generate a derivative of -1
    #and the derivative of the correct_classes should be the sum of those -1s
    dW = np.transpose(X).dot(dscore)
    #use the size of matrix to find out
    dW = dW/num_train + reg*W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
