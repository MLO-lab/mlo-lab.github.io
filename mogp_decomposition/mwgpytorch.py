import os
import numpy as np
import torch
import gpytorch
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle
float_type = torch.float
import tqdm

class LargeFeatureExtractor(torch.nn.Sequential):
    def __init__(self, input_dim, output_dim):
        super(LargeFeatureExtractor, self).__init__()
        self.add_module('linear1', torch.nn.Linear(input_dim, 1))
        self.add_module('relu1', torch.nn.ReLU())
        self.add_module('linear2', torch.nn.Linear(1, output_dim))
        self.add_module('relu2', torch.nn.ReLU())
        
class ExactDKL_GPModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood, mean_module, covar_module, feature_extractor1, feature_extractor2, feature_extractor3):
            super(ExactDKL_GPModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = mean_module
            self.covar_module = covar_module
            self.feature_extractor1 = feature_extractor1
            self.feature_extractor2 = feature_extractor2
            if feature_extractor3 is not None:
                self.feature_extractor3 = feature_extractor3

            # This module will scale the NN features so that they're nice values
            self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1., 1.)

        def forward(self, x1, x2, x3 = None):
            # We're first putting our data through a deep net (feature extractor)
            projected_x1 = self.feature_extractor1(x1)
            projected_x1 = self.scale_to_bounds(projected_x1)  # Make the NN values "nice"
            
            projected_x2 = self.feature_extractor2(x2)
            projected_x2 = self.scale_to_bounds(projected_x2)  # Make the NN values "nice"
            
            if self.feature_extractor3 is not None:
                projected_x3 = self.feature_extractor3(x3)
                projected_x3 = self.scale_to_bounds(projected_x3)  # Make the NN values "nice"
                self.h = tf.concat([projected_x1, projected_x2, projected_x3], axis=1)
            else:
                self.h = tf.concat([projected_x1, projected_x2], axis=1)

            mean_x = self.mean_module()
            covar_x = self.covar_module(self.h)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
                  
        
class GPDTemplate:
    def __init__(
        self,
        M,
        emb_sizes,
        batch_size=512,
        te_size=None,
        obs_mean=None,
        emb_reg=1e-3,
        lr=1e-3,
        ARD=True,
        ExactDKL=True,
        save_path="./",
    ):
        """
        :param M: integer, number of inducing points.
        :param emb_sizes: a list of embedding sizes as integers.
        :param batch_size: integer, mini batch size for training, necessary to define the placeholders.
        :param te_size: integer, batch size for testing, necessary to define the placeholders.
        :param obs_mean: integer, mean of training targets, optional.
        :param emb_reg: float, regularization term for embeddings.
        :param lr: float, learning rate.
        :param ARD: boolean, ARD parameter in gpflow.kernel.
        :param svgp: boolean, if True then apply svgp, sgpr otherwise.
        :param save_path: string, path to save the trained models.
        """
        self.M = M
        self.emb_sizes = emb_sizes
        self.batch_size = batch_size
        self.te_size = te_size
        self.obs_mean = obs_mean
        self.emb_reg = emb_reg
        self.lr = lr
        self.ARD = ARD
        self.ExactDKL = ExactDKL
        self.save_path = save_path
        self.param_ids = list(locals().keys())
        

    def save(self):
        """
        Save trained models.
        :return: None
        """
        params = {}
        for v in self.param_ids:
            if v not in ["self", "kwargs", "__class__"]:
                params[v] = self.__getattribute__(v)
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        save_name = os.path.join(self.save_path, "class_params.pkl")
        with open(save_name, "wb") as handle:
            pickle.dump(params, handle)


class GPD(GPDTemplate):
    def __init__(self, I, J, K, **kwargs):
        """
        :param I: integer, number of entities in the first dimension.
        :param J: integer, number of entities in the second dimension.
        :param K: integer, number of entities in the third dimension.
        :param kwargs: a dictionary for training hyper parameters, forwarded to GPDTemplate.__init__().
        """
        super(GPD, self).__init__(**kwargs)

        self.I = I
        self.J = J
        self.K = K
        self.param_ids.extend(list(locals().keys()))

        # Placeholders:
        #self.index_1 = None
        #self.index_2 = None
        #self.index_3 = None
        #self.y = None

        #self.index_1_te = None
        #self.index_2_te = None
        #self.index_3_te = None

        #self.index_1_M = None
        #self.index_2_M = None
        #self.index_3_M = None

        #self.emb1 = None
        #self.emb2 = None
        #self.emb3 = None

        #self.e1 = None
        #self.e2 = None
        #self.e3 = None
        #self.e1_te = None
        #self.e2_te = None
        #self.e3_te = None
        #self.e1_M = None
        #self.e2_M = None
        #self.e3_M = None

        #self.h = None
        #self.h_te = None
        #self.h_M = None
        self.feature_extractor1=None
        self.feature_extractor2=None
        self.feature_extractor3=None
        
        self.kernel = None
        self.mean_fn = None
        self.gp_model = None
        self.loss = None

        #self.ym = None
        #self.yv = None
        #self.ym_te = None
        #self.yv_te = None

        self.opt_step = None
        #self.sess = None

    def make_kernels(self, emb_size, kernels=None, active_dims=None):
        """
        :param emb_size: integer, embedding size for a kernel of one specific dimension.
        :param kernels: a list of strings, e.g. ['RBF', 'White], the sum of which shall form the kernel for one
        specific dimension.
        :param active_dims: active dimension of the kernel.
        :return: one kernel being the sum of all kernels required by the parameter 'kernels'.
        """
        kern = None
        if "RBF" in kernels:
            kern = gpytorch.kernels.RBFKernel(ard_num_dims=emb_size, active_dims= active_dims)
            #kern = gpflow.kernels.RBF(
            #    input_dim=emb_size, ARD=self.ARD, active_dims=active_dims
            #)

        if "Linear" in kernels:
            if kern is None:
                kern = gpytorch.kernels.LinearKernel(num_dimensions=emb_size)
                #kern = gpflow.kernels.Linear(input_dim=emb_size, ARD=self.ARD, active_dims=active_dims)
                #    input_dim=emb_size, ARD=self.ARD, active_dims=active_dims
                #)
            else:
                kern = kern + gpytorch.kernels.LinearKernel(num_dimensions=emb_size)
                #kern = kern + gpflow.kernels.Linear(
                #    input_dim=emb_size, ARD=self.ARD, active_dims=active_dims
                #)

        #if "White" in kernels:
            #???????????
            #kern = kern + gpflow.kernels.White(emb_size)

        return kern

    def build(self, kernels=["RBF", "White"], optimiser="adam"):  # , **kwargs
        """
        Building the GP-Decomposition model, partially by calling build_svgp or build_sgpr.
        :param kernels: list of strings, defining the kernel structure for each embedding.
        :param optimiser: string, currently either 'adam' or 'adagrad'.
        :return: None.
        """

        #if self.svgp:
        #    self.build_svgp()
        #else:
        #    self.build_sgpr()
        #with tf.variable_scope("embs"):
        #    self.emb1 = tf.keras.layers.Embedding(
        #        input_dim=self.I,
        #        output_dim=self.emb_sizes[0],
        #        dtype=tf.float64,
        #        embeddings_regularizer=tf.keras.regularizers.l2(self.emb_reg),
        #        name="emb1",
        #    )
        #    self.emb2 = tf.keras.layers.Embedding(
        #        input_dim=self.J,
        #        output_dim=self.emb_sizes[1],
        #        dtype=tf.float64,
        #        embeddings_regularizer=tf.keras.regularizers.l2(self.emb_reg),
        #        name="emb2",
        #    )
        #    if self.K is not None:
        #        self.emb3 = tf.keras.layers.Embedding(
        #            input_dim=self.K,
        #            output_dim=self.emb_sizes[2],
        #            dtype=tf.float64,
        #            embeddings_regularizer=tf.keras.regularizers.l2(self.emb_reg),
        #            name="emb3",
        #        )
#
        #    self.e1 = tf.keras.layers.Flatten()(self.emb1(self.index_1))
        #    self.e2 = tf.keras.layers.Flatten()(self.emb2(self.index_2))
        #    if self.K is not None:
        #        self.e3 = tf.keras.layers.Flatten()(self.emb3(self.index_3))
        #        self.h = tf.concat([self.e1, self.e2, self.e3], axis=1)
        #    else:
        #        self.h = tf.concat([self.e1, self.e2], axis=1)
#
        #    self.e1_te = tf.keras.layers.Flatten()(self.emb1(self.index_1_te))
        #    self.e2_te = tf.keras.layers.Flatten()(self.emb2(self.index_2_te))
        #    if self.K is not None:
        #        self.e3_te = tf.keras.layers.Flatten()(self.emb3(self.index_3_te))
        #        self.h_te = tf.concat([self.e1_te, self.e2_te, self.e3_te], axis=1)
        #    else:
        #        self.h_te = tf.concat([self.e1_te, self.e2_te], axis=1)
#
        #    self.e1_M = tf.keras.layers.Flatten()(self.emb1(self.index_1_M))
        #    self.e2_M = tf.keras.layers.Flatten()(self.emb2(self.index_2_M))
        #    if self.K is not None:
        #        self.e3_M = tf.keras.layers.Flatten()(self.emb3(self.index_3_M))
        #        self.h_M = tf.concat([self.e1_M, self.e2_M, self.e3_M], axis=1)
        #    else:
        #        self.h_M = tf.concat([self.e1_M, self.e2_M], axis=1)
#
        #    self.h = tf.cast(self.h, dtype=float_type)
        #    self.h_te = tf.cast(self.h_te, dtype=float_type)
        #    self.h_M = tf.cast(self.h_M, dtype=float_type)
        
        #Defining the DKL Feature Extractor
        self.feature_extractor1 = LargeFeatureExtractor(self.I, self.emb_sizes[0])
        self.feature_extractor2 = LargeFeatureExtractor(self.J, self.emb_sizes[1])
        if self.K is not None:
            self.feature_extractor3 = LargeFeatureExtractor(self.K, self.emb_sizes[2])
            
        # Coregionalization Kernel
        kernels1 = self.make_kernels(
            emb_size=self.emb_sizes[0],
            kernels=kernels,
            active_dims=np.arange(0, self.emb_sizes[0]),
        )
        kernels2 = self.make_kernels(
            emb_size=self.emb_sizes[1],
            kernels=kernels,
            active_dims=np.arange(
                self.emb_sizes[0], self.emb_sizes[0] + self.emb_sizes[1]
            ),
        )
        if self.K is not None:
            kernels3 = self.make_kernels(
                kernels=kernels,
                emb_size=self.emb_sizes[2],
                active_dims=np.arange(
                    self.emb_sizes[0] + self.emb_sizes[1],
                    self.emb_sizes[0] + self.emb_sizes[1] + self.emb_sizes[2],
                ),
            )
            self.kernel = kernels1 * kernels2 * kernels3
        else:
            self.kernel = kernels1 * kernels2
        if self.obs_mean is not None:
            observations_mean = torch.tensor([self.obs_mean], dtype=torch.float64)
            self.mean_fn = lambda _: observations_mean[:, None]
        else:
            self.mean_fn = self.obs_mean

        #Z_size = self.emb_sizes[0] + self.emb_sizes[1]
        #if self.K is not None:
        #    Z_size = Z_size + self.emb_sizes[2]
            
        
        
        #if self.svgp:
        #    self.gp_model = gpflow.models.SVGP(
        #        X=self.h,
        #        Y=tf.cast(self.y, dtype=float_type),
        #        Z=np.zeros((self.M, Z_size)),
        #        likelihood=gpflow.likelihoods.Gaussian(),
        #        mean_function=mean_fn,
        #        num_latent=1,
        #        kern=self.kernel,
        #    )
        #else:
        #    self.gp_model = gpflow.models.SGPR(
        #        X=self.h,
        #        Y=tf.cast(self.y, dtype=float_type),
        #        Z=np.zeros((self.M, Z_size)),
        #        mean_function=mean_fn,
        #        kern=self.kernel,
        #    )    
        #
        #self.loss = -self.gp_model.likelihood_tensor
        
        #m, v = self.gp_model._build_predict(self.h)
        #self.ym, self.yv = self.gp_model.likelihood.predict_mean_and_var(m, v)

        #m_te, v_te = self.gp_model._build_predict(self.h_te)
        #self.ym_te, self.yv_te = self.gp_model.likelihood.predict_mean_and_var(
        #    m_te, v_te
        #)

        #if optimiser == "adam":
        #    with tf.variable_scope("adam"):
        #        self.opt_step = tf.train.AdamOptimizer(
        #            learning_rate=self.lr, beta1=0.0
        #        ).minimize(self.loss)
#
        #else:
        #    with tf.variable_scope("adam"):
        #        self.opt_step = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss)
        

        #tf_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="adam")
        #tf_vars += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="embs")

        #self.sess = tf.Session()
        #self.sess.run(tf.variables_initializer(var_list=tf_vars))
        #self.gp_model.initialize(session=self.sess)
        self.train_loss = []


    def train(self, X_tr, Y_tr, X_val, Y_val, n_iter):
        """
        Performs batch training of the GP decomposition model.
        :param X_tr: matrix of training input. Indices of type integer and in shape (n_triples, n_entities).
        :param Y_tr: matrix of training target. Real numbers of type float and in shape (n_triples, 1).
        :param X_val: matrix of validation input. Indices of type integer and in shape (n_triples, n_entities).
        :param Y_val: matrix of validation target. Real numbers of type float and in shape (n_triples, 1).
        :param n_iter: number of training epochs.
        :param reset_inducing: boolean, True for resetting inducing points before training.
        :return: None
        """

        if self.ExactDKL:
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            self.gp_model = ExactDKL_GPModel(torch.Tensor(X_tr), torch.Tensor(Y_tr), likelihood, self.mean_fn, self.kernel, self.feature_extractor1, self.feature_extractor2, self.feature_extractor3)
        else:
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            self.gp_model = ExactDKL_GPModel(torch.Tensor(X_tr), torch.Tensor(Y_tr), likelihood, self.mean_fn, self.kernel, self.feature_extractor1, self.feature_extractor2, feature_extractor3)
         
        self.gp_model.train()
        likelihood.train()

        # Use the adam optimizer
        if self.K is not None:
            optimizer = torch.optim.Adam([
                {'params': self.gp_model.feature_extractor1.parameters()},
                {'params': self.gp_model.feature_extractor2.parameters()},
                {'params': self.gp_model.feature_extractor3.parameters()},
                {'params': self.gp_model.covar_module.parameters()},
                {'params': self.gp_model.likelihood.parameters()},
            ], lr=self.lr)
        else:
            optimizer = torch.optim.Adam([
                {'params': self.gp_model.feature_extractor1.parameters()},
                {'params': self.gp_model.feature_extractor2.parameters()},
                {'params': self.gp_model.covar_module.parameters()},
                {'params': self.gp_model.likelihood.parameters()},
            ], lr=self.lr)
        self.X_tr = X_tr
        self.Y_tr = Y_tr
        
        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, self.gp_model) 
        
        mb_splits = np.array_split(
            np.arange(X_tr.shape[0]), X_tr.shape[0] // self.batch_size + 1
        )
        
        #iterator = tqdm.notebook.tqdm(range(n_iter))
        #for l in iterator:
        for l in range(n_iter):
            epoch_loss_tr = 0.0
            epoch_loss_val = 0.0
            shuffle_ids = np.random.choice(
                range(X_tr.shape[0]), X_tr.shape[0], replace=False
            )
            X_tr = np.copy(X_tr[shuffle_ids])
            Y_tr = np.copy(Y_tr[shuffle_ids])

            for ll in range(len(mb_splits)):
                mb_ids = mb_splits[ll]
                if len(mb_ids) != self.batch_size:
                    mb_ids = np.random.choice(mb_ids, self.batch_size, replace=True)

                mb_i = X_tr[mb_ids, 0]
                mb_j = X_tr[mb_ids, 1]
                if self.K is not None:
                    mb_k = X_tr[mb_ids, 2]
                mb_y = Y_tr[mb_ids]
                
                # Zero backprop gradients
                optimizer.zero_grad()
                # Get output from model
                if self.K is not None:
                    output = self.gp_model(torch.Tensor(mb_i), torch.Tensor(mb_j), torch.Tensor(mb_k))
                else:
                    output = self.gp_model(torch.Tensor(mb_i), torch.Tensor(mb_j))
                # Calc loss and backprop derivatives
                loss = -mll(output, mb_y)
                loss.backward()
                #iterator.set_postfix(loss=loss.item())
                optimizer.step()
               #if self.K is not None:
               #    _, mb_loss = self.sess.run(
               #        [self.opt_step, self.loss],
               #        feed_dict={
               #            self.index_1: mb_i[:, None],
               #            self.index_2: mb_j[:, None],
               #            self.index_3: mb_k[:, None],
               #            self.y: mb_y[:, None],
               #        },
               #    )
               #else:
               #    _, mb_loss = self.sess.run(
               #        [self.opt_step, self.loss],
               #        feed_dict={
               #            self.index_1: mb_i[:, None],
               #            self.index_2: mb_j[:, None],
               #            self.y: mb_y[:, None],
               #        },
               #    )
               ## mb_loss = mb_loss / len(mb_ids)
                epoch_loss_tr = epoch_loss_tr + mb_loss
            epoch_loss_tr = epoch_loss_tr / X_tr.shape[0]  # len(mb_splits)
            self.train_loss.append(epoch_loss_tr)

            print("epoch " + str(l) + ": " + str(epoch_loss_tr))
       


    def get_weights_params(self):
        """
        Returns the trained model weights, parameters, as well as the hyper parameters for meta learning.
        :return: List with 4 (3) elements with last element being kernel parameters and previous elements the embeddings
        """
        super(GPD, self).save()
        #embs1 = self.sess.run(self.emb1.embeddings)
        #embs2 = self.sess.run(self.emb2.embeddings)
        #if self.K is not None:
        #    embs3 = self.sess.run(self.emb3.embeddings)
        #trainables = self.gp_model.read_trainables(self.sess)
        embs1 = None
        embs2 = None
        embs3 = None
        trainables = None
        if self.K is not None:
            return [embs1, embs2, embs3, trainables]
        else:
            return [embs1, embs2, trainables]


    def save(self):
        """
        Saves the trained model weights, parameters, as well as the hyper parameters in the specified save_path.
        :return: None
        """
        super(GPD, self).save()
        #embs1 = self.sess.run(self.emb1.embeddings)
        #embs2 = self.sess.run(self.emb2.embeddings)
        #if self.K is not None:
        #    embs3 = self.sess.run(self.emb3.embeddings)
        #trainables = self.gp_model.read_trainables(self.sess)
        embs1 = None
        embs2 = None
        embs3 = None
        trainables = None

        save_name = os.path.join(self.save_path, "model_params.pkl")
        if self.K is not None:
            with open(save_name, "wb") as handle:
                pickle.dump([embs1, embs2, embs3, trainables], handle)
        else:
            with open(save_name, "wb") as handle:
                pickle.dump([embs1, embs2, trainables], handle)


    def load_params(self, load_list=None):
        """
        Loads trained weights, parameters as well as the hyper parameters from either the specified save_path or the passed list.
        :return: None
        """
 
        if self.K is not None:
            if load_list is None:
                load_name = os.path.join(self.save_path, "model_params.pkl")
                with open(load_name, "rb") as handle:
                    [embs1, embs2, embs3, gp_params] = pickle.load(handle)
            else:
                [embs1, embs2, embs3, gp_params] = load_list
        else:
            if load_list is None:
                load_name = os.path.join(self.save_path, "model_params.pkl")
                with open(load_name, "rb") as handle:
                    [embs1, embs2, gp_params] = pickle.load(handle)
            else:
                [embs1, embs2, gp_params] = load_list

        #self.gp_model.assign(gp_params, self.sess)
        #self.sess.run(self.emb1.embeddings.assign(embs1))
        #self.sess.run(self.emb2.embeddings.assign(embs2))
        #if self.K is not None:
        #    self.sess.run(self.emb3.embeddings.assign(embs3))





