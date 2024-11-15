import numpy as np

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree.

    Args:
        x: numpy array of shape (N,), N is the number of samples.
        degree: integer.

    Returns:
        poly: numpy array of shape (N,d+1)

    >>> build_poly(np.array([0.0, 1.5]), 2)
    array([[1.  , 0.  , 0.  ],
           [1.  , 1.5 , 2.25]])
    """

    N = len(x)
    for i in range(degree+1):

        if i == 0 :
            poly = np.ones((N,1))
        else:
            add = np.power(x,i*np.ones(N)) #(N,)
            add = np.expand_dims(add, axis=1)  # (N, 1)
            poly = np.concatenate((poly,add),axis = 1)
        
    
    # ***************************************************
    return poly

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold.

    Args:
        y:      shape=(N,)
        k_fold: K in K-fold, i.e. the fold num
        seed:   the random seed

    Returns:
        A 2D array of shape=(k_fold, N/k_fold) that indicates the data indices for each fold

    >>> build_k_indices(np.array([1., 2., 3., 4.]), 2, 1)
    array([[3, 2],
           [0, 1]])
    """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval : (k + 1) * interval] for k in range(k_fold)]

    return np.array(k_indices)

#insérer fct cross_validation, la mienne n'était correcte... 
#(modification avec fonction/méthode comme argument de la fonction)

def sets_for_cross_validation(y, x, k_indices, k):
    """return the loss of ridge regression for a fold corresponding to k_indices

    Args:
        y:          shape=(N,)
        x:          shape=(N,)
        k_indices:  2D array returned by build_k_indices()
        k:          scalar, the k-th fold (N.B.: not to confused with k_fold which is the fold nums)

    Returns:
        train and test root mean square errors rmse = sqrt(2 mse)

    >>> cross_validation(np.array([1.,2.,3.,4.]), np.array([6.,7.,8.,9.]), np.array([[3,2], [0,1]]), 1, 2, 3)
    (0.019866645527597114, 0.33555914361295175)
    """
    # ***************************************************
    
    train_indices = np.concatenate([k_indices[i] for i in range(len(k_indices)) if i != k])
    x_test = x[k_indices[k]] 
    y_test = y[k_indices[k]]
    x_train = x[train_indices]
    y_train = y[train_indices]


    return x_train,y_train,x_test,y_test