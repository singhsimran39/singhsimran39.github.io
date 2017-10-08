
# Softmax Function

The Softmax function is a generalization of the Logistic Function. Where the Logistic function is mostly used in binary classification problems the Softmax function is used for multi class classification problems. The Softmax Function squashes a vector *Z* $\in{R^{n}}$ to produce values in the range [0, 1]. The output of the softmax is normalized and the values can be interpreted as class probabilities. The form of the softmax is:

\begin{equation}
    \hat{y}_{i} = \frac{\mathrm{e}^{z_{i}}}{\sum\limits_{j=1}^{C}   \mathrm{e}^{z_j}}
\end{equation}

Where $C$ is the number of different classes.

# Derivative of Softmax Function
If the softmax function is used in a neural network then its derivative is used for backpropagating the errors to the hidden layers. The derivative is calculated in 2 different cases when $i=j$ and when $i\ne{j}$. The softmax can be written as:

\begin{equation}
\hat{y}_{i} = \frac{\mathrm{e}^{z_{i}}} {\mathrm{e}^{z_{1}} + \mathrm{e}^{z_{2}} +...+ \mathrm{e}^{z_{i}} + ... +  \mathrm{e}^{z_{C}}}
\end{equation}

When $i=j$ (using the Quotient Rule), differentiating and rearranging the sofxtmax equation gives us:

\begin{equation}
    \frac{\partial{\hat{y}_{i}}}{\partial{z_{j}}} = \frac{\mathrm{e}^{z_{i}}}{\sum_{C}} 
    \frac{\sum_{C} -\mathrm{e}^{z_{j}}}{\sum_{C}}
\end{equation}

\begin{equation}
    \frac{\partial{\hat{y}_{i}}}{\partial{z_{j}}} = \hat{y}_{i} (1 - \hat{y}_{i})
\end{equation}

When $i \ne{j}$ (using the Quotient Rule):

\begin{equation}
    \frac{\partial{\hat{y}_{i}}}{\partial{z_{j}}} = \frac{0 - \mathrm{e}^{z_{i}} \mathrm{e}^{z_{j}}}{{\sum_{C}}^2}
\end{equation}

\begin{equation}
    \frac{\partial{\hat{y}_{i}}}{\partial{z_{j}}} = - \hat{y}_{i} \hat{y}_{j}
\end{equation}

# Cross Entropy Loss Function

The softmax classifier generally uses the Cross Entropy Loss function which is of the form:

\begin{equation}
    E = - \sum\limits_{j=1}^{C} t_{j}\log{\hat{y}_{j}}
\end{equation}

When the output vector for the multi class classification is one hot encoded the $t_{j}$ is 1 for only the actual class that the example belongs to and zero at all other places. This reduces the form of the Cross Entropy Loss function to:

\begin{equation}
    E = - t_{j} \log{\hat{y}_{j}}
\end{equation}

where $t_{j}$ and $\hat{y}_{j}$ are 1 and predicted probability by the classifier.

### Backpropagating the Loss to the last hidden layer of the neural network

The $\hat{y}_{j}$ in the loss function comes from the softmax equation, which is a function of the $z$ therefore to backpropagate the error we must differentiate $E$ w.r.t. the values coming into the putput latyer and multiply that derivative with the derivative of the output which is 1. Using the chain rule to backpropagate the error we have:

\begin{equation}
    \frac{\partial{E}}{\partial{z_{i}}} = - \sum\limits_{j=1}^{C} \frac{\partial{E}}{\partial{\hat{y}_{j}}} \frac{\partial{\hat{y}_{j}}}{\partial{z_{i}}}
\end{equation}

\begin{equation}
    \frac{\partial{E}}{\partial{z_{i}}} = - \sum\limits_{j=1}^{C} \frac{t_{j}}{\hat{y}_{j}} \frac{\partial{\hat{y}_{j}}}{\partial{z_{i}}}
\end{equation}

We already have calculated the second part ($\frac{\partial{\hat{y}_{j}}}{\partial{z}_{i}}$) of the above equation in the previous section, thus substituting the values for when $i = j$ and $i\ne{j}$ we get:

\begin{equation}
    \frac{\partial{E}}{\partial{z_{i}}} = - \frac{t_{j}}{\hat{y}_{j}} \hat{y}_{i}(1 - \hat{y}_{i}) - \sum\limits_{j=1, j\ne{i}}^{C} \frac{t_{j}}{\hat{y}_{j}} (-\hat{y}_{j}\hat{y}_{i})
\end{equation}

\begin{equation}
    \frac{\partial{E}}{\partial{z_{i}}} = - t_{j}(1 - \hat{y}_{i}) + \sum\limits_{j\ne{i}} t_{j}\hat{y}_{i}
\end{equation}

\begin{equation}
    \frac{\partial{E}}{\partial{z_{i}}} = - t_{j} + t_{j}\hat{y}_{i} + \sum\limits_{j\ne{i}} t_{j}\hat{y}_{i}
\end{equation}

\begin{equation}
    \frac{\partial{E}}{\partial{z_{i}}} = - t_{j} + \sum\limits_{j=1}^{C} t_{j}\hat{y}_{i}
\end{equation}

since $\hat{y}_{i}$ is constant

\begin{equation}
    \frac{\partial{E}}{\partial{z_{i}}} = - t_{j} + \hat{y}_{i} \sum\limits_{j=1}^{C} t_{j}
\end{equation}

since for one hot encoding the sum of the vector t is equal to, we have:

\begin{equation}
    \frac{\partial{E}}{\partial{z_{i}}} = \hat{y}_{i} - t_{j}
\end{equation}

The above result can then be multiplied with the derivative of the hidden layer and backpropagated further down in the network.
