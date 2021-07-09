from builtins import range
from builtins import object
import numpy as np

from ..layers import *
from ..layer_utils import *




# def affine_bn_relu_forward(x, w, b, gamma, beta, bn_params):
#     a, fc_cache = affine_forward(x, w, b)
#     bn, bn_cache = batchnorm_forward(x=a, gamma = gamma, beta=beta, bn_param=bn_params)
#     out, relu_cache = relu_forward(bn)
#     #connect the three forward layer
#     cache = (fc_cache, bn_cache, relu_cache)
#     #combine the three caches as one
#     return out, cache
#
#
# def affine_bn_relu_backward(dout, cache):
#     fc_cache, bn_cache, relu_cache = cache
#     dbn = relu_backward(dout=dout, cache=relu_cache)
#     da, dgamma, dbeta = batchnorm_backward_alt(dout=dbn, cache=bn_cache)
#     dx, dw, db = affine_backward(dout=da, cache=fc_cache)
#     return dx, dw, db, dgamma, dbeta
def affine_bn_relu_forward(x,w,b,gamma,beta,bn_params):
    """
       Convenience layer that perorms an affine WITH BACTHNORMALIZATION transform followed by a ReLU
       Inputs:
       - x: Input to the affine layer
       - w, b: Weights for the affine layer
       Returns a tuple of:
       - out: Output from the ReLU
       - cache: Object to give to the backward pass
       - gamma: Scale parameter of shape (D,)
       - beta: Shift paremeter of shape (D,)
       - bn_param: Dictionary with the following keys:
         - mode: 'train' or 'test'; required
         - eps: Constant for numeric stability
         - momentum: Constant for running mean / variance.
         - running_mean: Array of shape (D,) giving running mean of features
         - running_var Array of shape (D,) giving running variance of features

       Returns a tuple of:
       - out: Output from the ReLU
       - cache: Object to give to the backward pass
       """
    a_fc, fc_cache = affine_forward(x,w,b)
    a_bn,bn_cache = batchnorm_forward(a_fc,gamma,beta,bn_params)
    out,relu_cache = relu_forward(a_bn)
    cache = (fc_cache, bn_cache, relu_cache)
    return out, cache
def affine_bn_relu_backward(dout, cache):
    """
    Backward pass for the affine-bn-relu convenience layer
    """
    fc_cache, bn_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    da_bn, dgamma,dbeta = batchnorm_backward(da,bn_cache)
    dx,dw,db = affine_backward(da_bn,fc_cache)
    return dx,dw,db,dgamma,dbeta




class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(
        self,
        input_dim=3 * 32 * 32,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # use randn function to initialize the weight
        W1 = np.random.randn(input_dim, hidden_dim) * weight_scale
        W2 = np.random.randn(hidden_dim, num_classes) * weight_scale
        b1 = np.zeros((1, hidden_dim))
        b2 = np.zeros((1, num_classes))

        # store the initialized weight and bias to the dictionary
        self.params['W1'] = W1
        self.params['W2'] = W2
        self.params['b1'] = b1
        self.params['b2'] = b2

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        W1 = self.params['W1']
        W2 = self.params['W2']
        b1 = self.params['b1']
        b2 = self.params['b2']

        z1, cache_z1 = affine_forward(X, W1, b1)
        a1, cache_a1 = relu_forward(z1)
        scores, cache_scores = affine_forward(a1, W2, b2)


        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        data_loss, dscores = softmax_loss(scores, y)
        reg_loss = 0.5*self.reg*(np.sum(W1**2) + np.sum(W2**2))
        loss = data_loss + reg_loss

        da1, dW2, db2 = affine_backward(dscores, cache_scores)
        dz1 = relu_backward(da1, cache_a1)
        dX, dW1, db1 = affine_backward(dz1, cache_z1)

        grads['W1'] = dW1 + self.reg*W1
        grads['W2'] = dW2 + self.reg*W2
        grads['b1'] = db1
        grads['b2'] = db2


        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(
        self,
        hidden_dims,
        input_dim=3 * 32 * 32,
        num_classes=10,
        dropout=1,
        normalization=None,
        reg=0.0,
        weight_scale=1e-2,
        dtype=np.float32,
        seed=None,
    ):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # the first layer is different
        # self.params['W1'] = np.random.randn(input_dim, hidden_dims[0]) * weight_scale
        # self.params['b1'] = np.zeros(hidden_dims[0])
        # for i in range(len(hidden_dims)):
        #     output_dim = hidden_dims[i]
        #     #the layers that has the input dimension of the former layer
        #     if i >= 0:
        #         self.params['W'+str(i+1)] = np.random.randn(hidden_dims[i-1], output_dim)*weight_scale
        #         self.params['b'+str(i+1)] = np.zeros(output_dim)
        #         if self.normalization :
        #             self.params['gamma'+str(i+1)] = np.ones(output_dim)
        #             self.params['beta'+str(i+1)] = np.zeros(output_dim)
        #             #initialize for the batchnormalization
        #     else:
        #         continue
        # #the last layer is also different, it has the output_dim of num_classes
        # self.params['W'+str(self.num_layers)] = np.random.randn(hidden_dims[-1], num_classes)*weight_scale
        # self.params['b'+str(self.num_layers)] = np.zeros(num_classes)
        input_size = input_dim
        for i in range(len(hidden_dims)):
            output_size = hidden_dims[i]
            self.params['W' + str(i + 1)] = np.random.randn(input_size, output_size) * weight_scale
            self.params['b' + str(i + 1)] = np.zeros(output_size)
            if not self.normalization is None:
                self.params['gamma' + str(i + 1)] = np.ones(output_size)
                self.params['beta' + str(i + 1)] = np.zeros(output_size)
            input_size = output_size  # 下一层的输入
        # 输出层，没有BN操作
        self.params['W' + str(self.num_layers)] = np.random.randn(input_size, num_classes) * weight_scale
        self.params['b' + str(self.num_layers)] = np.zeros(num_classes)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout}
            if seed is not None:
                self.dropout_param["seed"] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization == "batchnorm":
            self.bn_params = [{"mode": "train"} for i in range(self.num_layers - 1)]
        if self.normalization == "layernorm":
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param["mode"] = mode
        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # cache = {}
        # layer_input = X
        # for i in range(self.num_layers-1):
        #     if self.normalization :
        #         layer_output, cache[i+1] = affine_bn_relu_forward(layer_input,
        #                                                           w=self.params['W'+str(i+1)], b=self.params['b'+str(i+1)],
        #                                                           gamma=self.params['gamma'+str(i+1)],
        #                                                           beta=self.params['beta'+str(i+1)],
        #                                                           bn_params=self.bn_params[i])
        #     elif self.use_dropout:
        #         pass
        #     #leave alone the normalization and dropout part and focus on the propagation of networks
        #     else:
        #         layer_output, cache[i+1] = affine_relu_forward(layer_input, self.params['W'+str(i+1)], self.params['b'+str(i+1)])
        #         #do one forward propagation
        #     layer_input = layer_output#renew the input to the next layer
        # scores, cache[self.num_layers] = affine_forward(layer_input, self.params['W'+str(self.num_layers)],
        #                                                 self.params['b'+str(self.num_layers)], )
        #the last layer just output the scores for the classification

        cache = {}  # 需要存储反向传播需要的参数
        cache_dropout = {}
        hidden = X
        for i in range(self.num_layers - 1):
            if self.normalization:
                hidden, cache[i + 1] = affine_bn_relu_forward(hidden,
                                                              self.params['W' + str(i + 1)],
                                                              self.params['b' + str(i + 1)],
                                                              self.params['gamma' + str(i + 1)],
                                                              self.params['beta' + str(i + 1)],
                                                              self.bn_params[i])
            else:
                hidden, cache[i + 1] = affine_relu_forward(hidden, self.params['W' + str(i + 1)],
                                                           self.params['b' + str(i + 1)])
            if self.use_dropout:
                hidden, cache_dropout[i+1] =  dropout_forward(hidden, self.dropout_param)
        # 最后一层不用激活
        scores, cache[self.num_layers] = affine_forward(hidden, self.params['W' + str(self.num_layers)],
                                                        self.params['b' + str(self.num_layers)])


        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == "test":
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # loss, dscores = softmax_loss(scores, y)#compute the loss and gradient for the last layer
        # dhidden, grads['W'+str(self.num_layers)], grads['b'+str(self.num_layers)] = affine_backward(dscores, cache[self.num_layers])
        # #the last layer's gradient is different
        # loss += 0.5*self.reg*np.sum(self.params['W'+str(self.num_layers)]*self.params['W'+str(self.num_layers)])#add the regularization term
        # grads['W'+str(self.num_layers)] += self.reg*self.params['W'+str(self.num_layers)]
        #
        # #then get the loss and gradient for each layer
        # for i in range(self.num_layers-2, 0, -1):
        #     if self.use_dropout:
        #         pass
        #     elif self.normalization:
        #         dhidden, dW, db, dgamma, dbeta = affine_bn_relu_backward(dhidden, cache[i])
        #         grads['gamma'+str(i+1)] = dgamma
        #         grads['beta'+str(i+1)] = dbeta
        #     else:
        #         dhidden, dW, db = affine_relu_backward(dhidden, cache[i])
        #     # count from backward to compute each gradient because the upperstream gradient is from the latter layer
        #     #then store the loss and the gradient
        #     loss += 0.5*self.reg*np.sum(self.params['W'+str(i+1)]*self.params['W'+str(i+1)])
        #     grads['W'+str(i+1)] = dW + self.reg*np.sum(self.params['W'+str(i+1)])
        #     grads['b'+str(i+1)] = db
        loss, dS = softmax_loss(scores, y)
        # 最后一层没有relu激活
        dhidden, grads['W' + str(self.num_layers)], grads['b' + str(self.num_layers)] \
            = affine_backward(dS, cache[self.num_layers])
        loss += 0.5 * self.reg * np.sum(
            self.params['W' + str(self.num_layers)] * self.params['W' + str(self.num_layers)])
        grads['W' + str(self.num_layers)] += self.reg * self.params['W' + str(self.num_layers)]

        for i in range(self.num_layers - 1, 0, -1):
            loss += 0.5 * self.reg * np.sum(self.params["W" + str(i)] * self.params["W" + str(i)])
            # 倒着求梯度
            if self.use_dropout:
                dhidden = dropout_backward(dhidden, cache= cache_dropout[i])
            if self.normalization == 'batchnorm':
                dhidden, dw, db, dgamma, dbeta = affine_bn_relu_backward(dhidden, cache[i])
                grads['gamma' + str(i)] = dgamma
                grads['beta' + str(i)] = dbeta
            elif self.normalization == 'layernorm':
                pass
            else:
                dhidden, dw, db = affine_relu_backward(dhidden, cache[i])
            grads['W' + str(i)] = dw + self.reg * self.params['W' + str(i)]
            grads['b' + str(i)] = db




        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
