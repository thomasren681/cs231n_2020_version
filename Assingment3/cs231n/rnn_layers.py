from __future__ import print_function, division
from builtins import range
import numpy as np


"""
This file defines layer types that are commonly used for recurrent neural
networks.
"""


def rnn_step_forward(x, prev_h, Wx, Wh, b):
    """
    Run the forward pass for a single timestep of a vanilla RNN that uses a tanh
    activation function.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Inputs:
    - x: Input data for this timestep, of shape (N, D).
    - prev_h: Hidden state from previous timestep, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - cache: Tuple of values needed for the backward pass.
    """
    next_h, cache = None, None
    ##############################################################################
    # TODO: Implement a single forward step for the vanilla RNN. Store the next  #
    # hidden state and any values you need for the backward pass in the next_h   #
    # and cache variables respectively.                                          #
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    z = prev_h.dot(Wh) + x.dot(Wx) + b#z.size() is (N,H)
    next_h = np.tanh(z)
    cache = (next_h,Wh,Wx,prev_h,x)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return next_h, cache


def rnn_step_backward(dnext_h, cache):
    """
    Backward pass for a single timestep of a vanilla RNN.

    Inputs:
    - dnext_h: Gradient of loss with respect to next hidden state, of shape (N, H)
    - cache: Cache object from the forward pass

    Returns a tuple of:
    - dx: Gradients of input data, of shape (N, D)
    - dprev_h: Gradients of previous hidden state, of shape (N, H)
    - dWx: Gradients of input-to-hidden weights, of shape (D, H)
    - dWh: Gradients of hidden-to-hidden weights, of shape (H, H)
    - db: Gradients of bias vector, of shape (H,)
    """
    dx, dprev_h, dWx, dWh, db = None, None, None, None, None
    ##############################################################################
    # TODO: Implement the backward pass for a single step of a vanilla RNN.      #
    #                                                                            #
    # HINT: For the tanh function, you can compute the local derivative in terms #
    # of the output value from tanh.                                             #
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    next_h, Wh, Wx, prev_h, x = cache
    dtanh = dnext_h*(1-next_h**2)#the chain rule of the derivative of the tanh function
    #dtanh has the shape (N,H)
    dWh = prev_h.T.dot(dtanh)
    dx = dtanh.dot(Wx.T)
    dWx = x.T.dot(dtanh)
    dprev_h = dtanh.dot(Wh.T)
    db = np.sum(dtanh, axis=0)


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dx, dprev_h, dWx, dWh, db


def rnn_forward(x, h0, Wx, Wh, b):
    """
    Run a vanilla RNN forward on an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The RNN uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the RNN forward, we return the hidden states for all timesteps.

    Inputs:
    - x: Input data for the entire timeseries, of shape (N, T, D).
    - h0: Initial hidden state, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - h: Hidden states for the entire timeseries, of shape (N, T, H).
    - cache: Values needed in the backward pass
    """
    h, cache = None, None
    ##############################################################################
    # TODO: Implement forward pass for a vanilla RNN running on a sequence of    #
    # input data. You should use the rnn_step_forward function that you defined  #
    # above. You can use a for loop to help compute the forward pass.            #
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #for the vanilla version we can use a for loop to enumerate each data of the whole sequence
    #first we should define the shape of h then we can use the for loop

    N, T, D = x.shape
    H = h0.shape[1]
    h = np.zeros((N,T,H))
    prev_h = h0#the first prev_h is initialized to be h0
    #the latter prev_h is setted to be the next_h from the previous output
    for i in range(T):
        next_h, _ = rnn_step_forward(x[:,i,:], prev_h, Wx, Wh, b)
        prev_h = next_h
        h[:,i,:] = next_h

    cache = (h, Wh, Wx, b, x, h0)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return h, cache


def rnn_backward(dh, cache):
    """
    Compute the backward pass for a vanilla RNN over an entire sequence of data.

    Inputs:
    - dh: Upstream gradients of all hidden states, of shape (N, T, H). 
    
    NOTE: 'dh' contains the upstream gradients produced by the 
    individual loss functions at each timestep, *not* the gradients
    being passed between timesteps (which you'll have to compute yourself
    by calling rnn_step_backward in a loop).

    Returns a tuple of:
    - dx: Gradient of inputs, of shape (N, T, D)
    - dh0: Gradient of initial hidden state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
    - db: Gradient of biases, of shape (H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    ##############################################################################
    # TODO: Implement the backward pass for a vanilla RNN running an entire      #
    # sequence of data. You should use the rnn_step_backward function that you   #
    # defined above. You can use a for loop to help compute the backward pass.   #
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    h, Wh, Wx, b, x, h0 = cache
    #the same step as the forward process

    N,T, H = dh.shape
    D = x.shape[2]
    dx = np.zeros((N,T,D))
    dh0 = np.zeros((N,H))
    dWx = np.zeros((D,H))
    dWh = np.zeros((H,H))
    db = np.zeros(H)
    dprev_h = np.zeros((N,H))#this is a temporary variable to store the gradient the links different steps

    for i in range(T-1,-1,-1):
        prev_h = h0 if i == 0 else h[:,i-1,:]
        cachei = (h[:, i, :], Wh, Wx, prev_h, x[:, i, :])
        dx[:,i,:], dprev_h, dWx_temp, dWh_temp, db_temp = rnn_step_backward(dh[:,i,:] + dprev_h, cachei)
        dWh += dWh_temp
        dWx += dWx_temp
        db += db_temp
    dh0 = dprev_h


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dx, dh0, dWx, dWh, db


def word_embedding_forward(x, W):
    """
    Forward pass for word embeddings. We operate on minibatches of size N where
    each sequence has length T. We assume a vocabulary of V words, assigning each
    word to a vector of dimension D.

    Inputs:
    - x: Integer array of shape (N, T) giving indices of words. Each element idx
      of x muxt be in the range 0 <= idx < V.
    - W: Weight matrix of shape (V, D) giving word vectors for all words.

    Returns a tuple of:
    - out: Array of shape (N, T, D) giving word vectors for all input words.
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    ##############################################################################
    # TODO: Implement the forward pass for word embeddings.                      #
    #                                                                            #
    # HINT: This can be done in one line using NumPy's array indexing.           #
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    out = W[x]
    cache = (x, W.shape)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return out, cache


def word_embedding_backward(dout, cache):
    """
    Backward pass for word embeddings. We cannot back-propagate into the words
    since they are integers, so we only return gradient for the word embedding
    matrix.

    HINT: Look up the function np.add.at

    Inputs:
    - dout: Upstream gradients of shape (N, T, D)
    - cache: Values from the forward pass

    Returns:
    - dW: Gradient of word embedding matrix, of shape (V, D).
    """
    dW = None
    ##############################################################################
    # TODO: Implement the backward pass for word embeddings.                     #
    #                                                                            #
    # Note that words can appear more than once in a sequence.                   #
    # HINT: Look up the function np.add.at                                       #
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    """
        np.ufunc.at(a, indices, b=None)
        Performs unbuffered in place operation on operand ‘a’ for elements specified by ‘indices’. 
        For addition ufunc, this method is equivalent to a[indices] += b, 
        except that results are accumulated for elements that are indexed more than once. 
        For example, a[[0,0]] += 1 will only increment the first element once because of buffering, 
        whereas add.at(a, [0,0], 1) will increment the first element twice.
    """

    x, dW_shape = cache
    dW = np.zeros(dW_shape)
    np.add.at(dW, x, dout)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dW


def sigmoid(x):
    """
    A numerically stable version of the logistic sigmoid function.
    """
    pos_mask = x >= 0
    neg_mask = x < 0
    z = np.zeros_like(x)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x)
    top[neg_mask] = z[neg_mask]
    return top / (1 + z)


def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
    """
    Forward pass for a single timestep of an LSTM.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Note that a sigmoid() function has already been provided for you in this file.

    Inputs:
    - x: Input data, of shape (N, D)
    - prev_h: Previous hidden state, of shape (N, H)
    - prev_c: previous cell state, of shape (N, H)
    - Wx: Input-to-hidden weights, of shape (D, 4H)
    - Wh: Hidden-to-hidden weights, of shape (H, 4H)
    - b: Biases, of shape (4H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - next_c: Next cell state, of shape (N, H)
    - cache: Tuple of values needed for backward pass.
    """
    next_h, next_c, cache = None, None, None
    #############################################################################
    # TODO: Implement the forward pass for a single timestep of an LSTM.        #
    # You may want to use the numerically stable sigmoid implementation above.  #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, H = prev_h.shape
    a = x.dot(Wx) + prev_h.dot(Wh) + b
    #then slice the a into four part
    ai = a[:,:H]
    af = a[:,H:2*H]
    ao = a[:,H*2:H*3]
    ag = a[:,H*3:H*4]
    #then compute i,f,o,g based on ai,af,ao,ag
    i = sigmoid(ai)
    f = sigmoid(af)
    o = sigmoid(ao)
    g = np.tanh(ag)
    next_c = f*prev_c + i*g
    next_h = o * np.tanh(next_c)
    cache = (x, prev_h, prev_c, i, f, o, g, Wx, Wh, next_c, a)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return next_h, next_c, cache


def lstm_step_backward(dnext_h, dnext_c, cache):
    """
    Backward pass for a single timestep of an LSTM.

    Inputs:
    - dnext_h: Gradients of next hidden state, of shape (N, H)
    - dnext_c: Gradients of next cell state, of shape (N, H)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient of input data, of shape (N, D)
    - dprev_h: Gradient of previous hidden state, of shape (N, H)
    - dprev_c: Gradient of previous cell state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    dx, dprev_h, dprev_c, dWx, dWh, db = None, None, None, None, None, None
    #############################################################################
    # TODO: Implement the backward pass for a single timestep of an LSTM.       #
    #                                                                           #
    # HINT: For sigmoid and tanh you can compute local derivatives in terms of  #
    # the output value from the nonlinearity.                                   #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x, prev_h, prev_c, i, f, o, g, Wx, Wh, next_c, a = cache
    N,H = dnext_h.shape
    ai = a[:, :H]
    af = a[:, H:2 * H]
    ao = a[:, H * 2:H * 3]
    ag = a[:, H * 3:H * 4]

    #the derivative of c^t-1 is special, for it's a combination of two equations
    #let's first compute the gradient of c^t-1
    dc_from_ct = dnext_c*f
    dtanhct = dnext_h*o
    dtanh = np.ones_like(next_c) - np.square(np.tanh(next_c))#the derivative of tanh is 1-tanh^2(x)
    # dprev_c = f*dtanh*dtanhct + dc_from_ct
    dc_total = dnext_c + dtanh*dnext_h*o

    #then compute the gradients for ifog
    di = dc_total*g
    df = dc_total*prev_c
    do = dnext_h*np.tanh(next_c)
    dg = dc_total*i

    #then compute the gradients for ai,af,ao,ag
    #the drivative for sigmoid function is e.g. dai = dupstream*(i*(1-i))
    dai = di*(i*(np.ones_like(i)-i))
    daf = df*(f*(np.ones_like(f)-f))
    dao = do*(o*(np.ones_like(o)-o))
    dag = dg*(np.ones_like(ag)-np.square(np.tanh(ag)))

    #then concatenate those four to get da then get the output gradients
    da = np.concatenate((dai,daf,dao,dag), axis=1)#(N,4H)
    dx = da.dot(Wx.T)
    dprev_h = da.dot(Wh.T)
    dprev_c = f * dtanh * dtanhct + dc_from_ct
    dWx = x.T.dot(da)
    dWh = prev_h.T.dot(da)
    db = np.sum(da, axis=0).T

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return dx, dprev_h, dprev_c, dWx, dWh, db


def lstm_forward(x, h0, Wx, Wh, b):
    """
    Forward pass for an LSTM over an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the LSTM forward, we return the hidden states for all timesteps.

    Note that the initial cell state is passed as input, but the initial cell
    state is set to zero. Also note that the cell state is not returned; it is
    an internal variable to the LSTM and is not accessed from outside.

    Inputs:
    - x: Input data of shape (N, T, D)
    - h0: Initial hidden state of shape (N, H)
    - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
    - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
    - b: Biases of shape (4H,)

    Returns a tuple of:
    - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
    - cache: Values needed for the backward pass.
    """
    h, cache = None, None
    #############################################################################
    # TODO: Implement the forward pass for an LSTM over an entire timeseries.   #
    # You should use the lstm_step_forward function that you just defined.      #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #we will first initialize the various variable then use an explicit for loop to
    #run all the timesteps
    N, T, D = x.shape
    _, H = h0.shape

    cache_h = np.empty([N, T, H])
    cache_c = np.empty([N, T, H])
    I = np.empty([N, T, H])
    F = np.empty([N, T, H])
    O = np.empty([N, T, H])
    G = np.empty([N, T, H])
    nc = np.empty([N, T, H])
    A = np.empty([N, T, 4 * H])
    h = np.empty([N, T, H])

    prev_h = h0
    prev_c = np.zeros_like(h0)#set the first c and hidden state

    for i in range(T):
        cache_h[:, i, :] = prev_h
        cache_c[:, i, :] = prev_c
        next_h, next_c, cache_temp = lstm_step_forward(x[:, i, :], prev_h, prev_c, Wx, Wh, b)
        prev_h = next_h
        prev_c = next_c  # 更新隐藏层状态
        h[:, i, :] = prev_h  # 用来记录状态
        I[:, i, :] = cache_temp[3]
        F[:, i, :] = cache_temp[4]
        O[:, i, :] = cache_temp[5]
        G[:, i, :] = cache_temp[6]
        nc[:, i, :] = cache_temp[9]
        A[:, i, :] = cache_temp[10]

    cache = (x, cache_h, cache_c, I, F, O, G, Wx, Wh, nc, A)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return h, cache


def lstm_backward(dh, cache):
    """
    Backward pass for an LSTM over an entire sequence of data.]

    Inputs:
    - dh: Upstream gradients of hidden states, of shape (N, T, H)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient of input data of shape (N, T, D)
    - dh0: Gradient of initial hidden state of shape (N, H)
    - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    #############################################################################
    # TODO: Implement the backward pass for an LSTM over an entire timeseries.  #
    # You should use the lstm_step_backward function that you just defined.     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x = cache[0]
    N, T, D = x.shape
    N, T, H = dh.shape
    dWx = np.zeros((D, 4 * H))
    dWh = np.zeros((H, 4 * H))
    db = np.zeros(4 * H)
    dout = dh
    dx = np.empty([N, T, D])
    hnow = np.zeros([N, H])
    cnow = np.zeros([N, H])
    for k in range(T):
        i = T - 1 - k
        hnow = hnow + dout[:, i, :]
        cacheT = (cache[0][:, i, :], cache[1][:, i, :], cache[2][:, i, :], cache[3][:, i, :],
                  cache[4][:, i, :], cache[5][:, i, :], cache[6][:, i, :], cache[7], cache[8], cache[9][:, i, :],
                  cache[10][:, i, :])
        # 对应的从之前的cache中取出当前用的那一个
        dx_temp, dprev_h, dprev_c, dWx_temp, dWh_temp, db_temp = lstm_step_backward(hnow, cnow, cacheT)
        hnow = dprev_h
        cnow = dprev_c
        dx[:, i, :] = dx_temp
        dWx = dWx + dWx_temp
        dWh = dWh + dWh_temp
        db = db + db_temp  # 将当前的梯度加上之前累积的
    dh0 = hnow

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return dx, dh0, dWx, dWh, db


def temporal_affine_forward(x, w, b):
    """
    Forward pass for a temporal affine layer. The input is a set of D-dimensional
    vectors arranged into a minibatch of N timeseries, each of length T. We use
    an affine function to transform each of those vectors into a new vector of
    dimension M.

    Inputs:
    - x: Input data of shape (N, T, D)
    - w: Weights of shape (D, M)
    - b: Biases of shape (M,)

    Returns a tuple of:
    - out: Output data of shape (N, T, M)
    - cache: Values needed for the backward pass
    """
    N, T, D = x.shape
    M = b.shape[0]
    out = x.reshape(N * T, D).dot(w).reshape(N, T, M) + b
    cache = x, w, b, out
    return out, cache


def temporal_affine_backward(dout, cache):
    """
    Backward pass for temporal affine layer.

    Input:
    - dout: Upstream gradients of shape (N, T, M)
    - cache: Values from forward pass

    Returns a tuple of:
    - dx: Gradient of input, of shape (N, T, D)
    - dw: Gradient of weights, of shape (D, M)
    - db: Gradient of biases, of shape (M,)
    """
    x, w, b, out = cache
    N, T, D = x.shape
    M = b.shape[0]

    dx = dout.reshape(N * T, M).dot(w.T).reshape(N, T, D)
    dw = dout.reshape(N * T, M).T.dot(x.reshape(N * T, D)).T
    db = dout.sum(axis=(0, 1))

    return dx, dw, db


def temporal_softmax_loss(x, y, mask, verbose=False):
    """
    A temporal version of softmax loss for use in RNNs. We assume that we are
    making predictions over a vocabulary of size V for each timestep of a
    timeseries of length T, over a minibatch of size N. The input x gives scores
    for all vocabulary elements at all timesteps, and y gives the indices of the
    ground-truth element at each timestep. We use a cross-entropy loss at each
    timestep, summing the loss over all timesteps and averaging across the
    minibatch.

    As an additional complication, we may want to ignore the model output at some
    timesteps, since sequences of different length may have been combined into a
    minibatch and padded with NULL tokens. The optional mask argument tells us
    which elements should contribute to the loss.

    Inputs:
    - x: Input scores, of shape (N, T, V)
    - y: Ground-truth indices, of shape (N, T) where each element is in the range
         0 <= y[i, t] < V
    - mask: Boolean array of shape (N, T) where mask[i, t] tells whether or not
      the scores at x[i, t] should contribute to the loss.

    Returns a tuple of:
    - loss: Scalar giving loss
    - dx: Gradient of loss with respect to scores x.
    """

    N, T, V = x.shape

    x_flat = x.reshape(N * T, V)
    y_flat = y.reshape(N * T)
    mask_flat = mask.reshape(N * T)

    probs = np.exp(x_flat - np.max(x_flat, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    loss = -np.sum(mask_flat * np.log(probs[np.arange(N * T), y_flat])) / N
    dx_flat = probs.copy()
    dx_flat[np.arange(N * T), y_flat] -= 1
    dx_flat /= N
    dx_flat *= mask_flat[:, None]

    if verbose:
        print("dx_flat: ", dx_flat.shape)

    dx = dx_flat.reshape(N, T, V)

    return loss, dx
