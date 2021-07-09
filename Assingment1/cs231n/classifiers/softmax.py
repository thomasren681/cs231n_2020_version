from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.
    here D is 3073 ; C is 10; N is the number of training examples in a minibatch

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
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    num_classes = W.shape[1]
    scores = X.dot(W)
    max_score = np.max(X, axis=1)
    max_score = np.reshape(max_score, [-1,1])
    scores -= max_score
    #all the entries in the matrix will be less or equal than zero
    #so that the entries in exp(scores)will range from zero to one

    for i in range(num_train):
        correct_score = scores[i,y[i]]
        exp_sum = np.sum(np.exp(scores[i]))
        loss += np.log(exp_sum) - correct_score

        for j in range(num_classes):
            #the dW is different for correct scores and non-correct scores
            dW[:,j] += (np.exp(scores[i,j])/exp_sum)*X[i]
        dW[:,y[i]] -= X[i]

    loss = loss/num_train + 0.5*reg*np.sum(W*W)
    dW = dW/num_train + reg*W


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_train = X.shape[0]
    num_classes = W.shape[1]
    scores = X.dot(W)
    max_score = np.max(X, axis=1)
    max_score = np.reshape(max_score, [-1, 1])
    scores -= max_score
    correct_score = scores[np.arange(num_train),y]
    loss += np.sum(np.log(np.sum(np.exp(scores),axis=1))) - np.sum(correct_score)
    loss = loss/num_train + 0.5*reg*np.sum(W*W)


    # temp =  (np.exp(scores)/np.sum(np.exp(scores),axis=1)[:,np.newaxis])
    # dW = X.T.dot(temp)
    # dW[:,y] -= X.T
    # dW = dW/num_train + reg*W

    exp_scores = np.exp(scores)
    sum_exp_scores = np.sum(exp_scores, axis=1)
    exp_scores = exp_scores / sum_exp_scores[:, np.newaxis]
    for i in xrange(num_train):
        dW += exp_scores[i] * X[i][:, np.newaxis]
        dW[:, y[i]] -= X[i]
    dW /= num_train
    dW += reg * W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
