import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
from .hvae import HVAE
from .base import Downsample

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RHVAE(HVAE):
    def __init__(
        self, input_shape, out_channels, latent_dim, hidden_channels, n_lf = 3, eps_lf = 0.01, 
        beta_zero = 0.3, metric_dim=384, temperature=0.8, regularization=0.01, 
        skip=False, device = device
    ):
        HVAE.__init__(self, input_shape, out_channels, latent_dim, hidden_channels, n_lf, eps_lf, beta_zero, skip, device)
        
        # defines the Neural net to compute the metric
        # self.metric = Encoder(input_shape, hidden_channels, metric_dim)

        self.metric = nn.ModuleList(
            [Downsample(self.input_shape[0], self.hidden_channels[0], stride=1, norm=False)] +
            [Downsample(hidden_channels[i], hidden_channels[i + 1]) for i in range(hidden_channels.__len__() - 1)]
        )
        self.metric_flatten = nn.Flatten()

        self.metric_fc = nn.Linear(self.hidden_channels[-1] * np.prod(self.encoder.encoding_shapes[-1]), metric_dim)
        # diagonal
        self.metric_diag = nn.Linear(metric_dim, self.latent_dim)

        # remaining coefficients
        k = int(self.latent_dim * (self.latent_dim - 1) / 2)
        self.metric_k = nn.Linear(metric_dim, k)

        self.T = nn.Parameter(torch.Tensor([temperature]), requires_grad=False)
        self.lbd = nn.Parameter(
            torch.Tensor([regularization]), requires_grad=False
        )

        # this is used to store the matrices and centroids throughout training for
        # further use in metric update (L is the cholesky decomposition of M)
        self.M = []
        self.centroids = []

        # define a starting metric (c_i = 0 & L = I_d)
        def G(z):
            return (
                torch.eye(self.latent_dim, device=self.device).unsqueeze(0)
                * torch.exp(-torch.norm(z.unsqueeze(1), dim=-1) ** 2)
                .unsqueeze(-1)
                .unsqueeze(-1)
            ).sum(dim=1) + self.lbd * torch.eye(self.latent_dim).to(self.device)

        self.G = G

    def metric_forward(self, x):
        """
        This function returns the outputs of the metric neural network
        Outputs:
        --------
        L (Tensor): The L matrix as used in the metric definition
        M (Tensor): L L^T
        """

        for layer in self.metric:
            x = layer(x)
        x = self.metric_flatten(x)
        x = torch.relu(self.metric_fc(x))
        h1, h2 = self.metric_diag(x), self.metric_k(x)

        L = torch.zeros((x.shape[0], self.latent_dim, self.latent_dim)).to(self.device)
        indices = torch.tril_indices(
            row=self.latent_dim, col=self.latent_dim, offset=-1
        )

        # get non-diagonal coefficients
        L[:, indices[0], indices[1]] = h2

        # add diagonal coefficients
        L = L + torch.diag_embed(h1.exp())

        return L, L @ torch.transpose(L, 1, 2)

    def update_metric(self):
        """
        As soon as the model has seen all the data points (i.e. at the end of 1 loop)
        we update the final metric function using \mu(x_i) as centroids
        """
        # convert to 1 big tensor
        self.M_tens = torch.cat(self.M)
        self.centroids_tens = torch.cat(self.centroids)

        # define new metric
        def G(z):
            return torch.inverse(
                (
                    self.M_tens.unsqueeze(0)
                    * torch.exp(
                        -torch.norm(
                            self.centroids_tens.unsqueeze(0) - z.unsqueeze(1), dim=-1
                        )
                        ** 2
                        / (self.T ** 2)
                    )
                    .unsqueeze(-1)
                    .unsqueeze(-1)
                ).sum(dim=1)
                + self.lbd * torch.eye(self.latent_dim).to(self.device)
            )

        def G_inv(z):
            return (
                self.M_tens.unsqueeze(0)
                * torch.exp(
                    -torch.norm(
                        self.centroids_tens.unsqueeze(0) - z.unsqueeze(1), dim=-1
                    )
                    ** 2
                    / (self.T ** 2)
                )
                .unsqueeze(-1)
                .unsqueeze(-1)
            ).sum(dim=1) + self.lbd * torch.eye(self.latent_dim).to(self.device)

        self.G = G
        self.G_inv = G_inv
        self.M = []
        self.centroids = []

    def forward(self, x):
        """
        The RHVAE model
        """

        recon_x, z0, eps0, mu, log_var, features = self.vae_forward(x)

        z = z0

        if self.training:

            # update the metric using batch data points
            L, M = self.metric_forward(x)

            # store LL^T and mu(x_i) to update final metric
            self.M.append(M.clone().detach())
            self.centroids.append(mu.clone().detach())

            G_inv = (
                M.unsqueeze(0)
                * torch.exp(
                    -torch.norm(mu.unsqueeze(0) - z.unsqueeze(1), dim=-1) ** 2
                    / (self.T ** 2)
                )
                .unsqueeze(-1)
                .unsqueeze(-1)
            ).sum(dim=1) + self.lbd * torch.eye(self.latent_dim).to(self.device)

        else:
            G = self.G(z)
            G_inv = self.G_inv(z)
            L = torch.linalg.cholesky(G)

        G_log_det = -torch.logdet(G_inv)

        gamma = torch.randn_like(z0, device=self.device)
        rho = gamma / self.beta_zero_sqrt
        beta_sqrt_old = self.beta_zero_sqrt

        # sample \rho from N(0, G)
        rho = (L @ rho.unsqueeze(-1)).squeeze(-1)

        recon_x = self.decode(z, features if self.skip else None)

        for k in range(self.n_lf):

            # perform leapfrog steps

            # step 1
            rho_ = self.leap_step_1(recon_x, x, z, rho, G_inv, G_log_det)

            # step 2
            z = self.leap_step_2(recon_x, x, z, rho_, G_inv, G_log_det)

            recon_x = self.decode(z, features if self.skip else None)

            if self.training:

                G_inv = (
                    M.unsqueeze(0)
                    * torch.exp(
                        -torch.norm(mu.unsqueeze(0) - z.unsqueeze(1), dim=-1) ** 2
                        / (self.T ** 2)
                    )
                    .unsqueeze(-1)
                    .unsqueeze(-1)
                ).sum(dim=1) + self.lbd * torch.eye(self.latent_dim).to(self.device)

            else:
                # compute metric value on new z using final metric
                G = self.G(z)
                G_inv = self.G_inv(z)

            G_log_det = -torch.logdet(G_inv)

            # step 3
            rho__ = self.leap_step_3(recon_x, x, z, rho_, G_inv, G_log_det)

            # tempering
            beta_sqrt = self._tempering(k + 1, self.n_lf)
            rho = (beta_sqrt_old / beta_sqrt) * rho__
            beta_sqrt_old = beta_sqrt

        return recon_x, z, z0, rho, eps0, gamma, mu, log_var, G_inv, G_log_det

    def leap_step_1(self, recon_x, x, z, rho, G_inv, G_log_det, steps=3):
        """
        Resolves first equation of generalized leapfrog integrator
        using fixed point iterations
        """

        def f_(rho_):
            H = self.hamiltonian(recon_x, x, z, rho_, G_inv, G_log_det)
            gz = grad(H, z, retain_graph=True)[0]
            return rho - 0.5 * self.eps_lf * gz

        rho_ = rho.clone()
        for _ in range(steps):
            rho_ = f_(rho_)
        return rho_

    def leap_step_2(self, recon_x, x, z, rho, G_inv, G_log_det, steps=3):
        """
        Resolves second equation of generalized leapfrog integrator
        using fixed point iterations
        """
        H0 = self.hamiltonian(recon_x, x, z, rho, G_inv, G_log_det)
        grho_0 = grad(H0, rho)[0]

        def f_(z_):
            H = self.hamiltonian(recon_x, x, z_, rho, G_inv, G_log_det)
            grho = grad(H, rho, retain_graph=True)[0]
            return z + 0.5 * self.eps_lf * (grho_0 + grho)

        z_ = z.clone()
        for _ in range(steps):
            z_ = f_(z_)
        return z_

    def leap_step_3(self, recon_x, x, z, rho, G_inv, G_log_det, steps=3):
        """
        Resolves third equation of generalized leapfrog integrator
        using fixed point iterations
        """
        H = self.hamiltonian(recon_x, x, z, rho, G_inv, G_log_det)
        gz = grad(H, z, create_graph=True)[0]
        return rho - 0.5 * self.eps_lf * gz

    def loss_function(
        self, recon_x, x, z0, zK, rhoK, eps0, gamma, mu, log_var, G_inv, G_log_det
    ):

        logpxz = self.log_p_xz(recon_x, x, zK)  # log p(x, z_K)
        logrhoK = (
            -0.5
            * (torch.transpose(rhoK.unsqueeze(-1), 1, 2) @ G_inv @ rhoK.unsqueeze(-1))
            .squeeze()
            .squeeze()
            - 0.5 * G_log_det
        ) - torch.log(
            torch.tensor([2 * np.pi]).to(self.device)
        ) * self.latent_dim / 2  # log p(\rho_K)

        logp = logpxz + logrhoK

        logq = self.normal.log_prob(eps0) - 0.5 * log_var.sum(dim=1)  # log(q(z_0|x))

        return -(logp - logq).sum()

    def log_p_x(self, x, sample_size=10):
        """
        Estimate log(p(x)) using importance sampling on q(z|x)
        """
        # print(sample_size)
        mu, log_var, featurs = self.encode(x)
        Eps = torch.randn(sample_size, x.size()[0], self.latent_dim, device=self.device)
        Z = (mu + Eps * torch.exp(0.5 * log_var)).reshape(-1, self.latent_dim)

        Z0 = Z

        recon_X = self.decode(Z, featurs if self.skip else None)

        # get metric value
        G_rep = self.G(Z)
        G_inv_rep = self.G_inv(Z)

        G_log_det_rep = torch.logdet(G_rep)

        L_rep = torch.linalg.cholesky(G_rep)

        G_inv_rep_0 = G_inv_rep
        G_log_det_rep_0 = G_log_det_rep

        # initialization
        gamma = torch.randn_like(Z0, device=self.device)
        rho = gamma / self.beta_zero_sqrt
        beta_sqrt_old = self.beta_zero_sqrt

        rho = (L_rep @ rho.unsqueeze(-1)).squeeze(
            -1
        )  # sample from the multivariate N(0, G)

        rho0 = rho

        X_rep = x.repeat(sample_size, 1, 1, 1)

        for k in range(self.n_lf):

            # step 1
            rho_ = self.leap_step_1(recon_X, X_rep, Z, rho, G_inv_rep, G_log_det_rep)

            # step 2
            Z = self.leap_step_2(recon_X, X_rep, Z, rho_, G_inv_rep, G_log_det_rep)

            recon_X = self.decode(Z, featurs if self.skip else None)

            G_rep_inv = self.G_inv(Z)
            G_log_det_rep = -torch.logdet(G_rep_inv)

            # step 3
            rho__ = self.leap_step_3(recon_X, X_rep, Z, rho_, G_inv_rep, G_log_det_rep)

            # tempering
            beta_sqrt = self._tempering(k + 1, self.n_lf)
            rho = (beta_sqrt_old / beta_sqrt) * rho__
            beta_sqrt_old = beta_sqrt

        bce = F.binary_cross_entropy(recon_X, X_rep, reduction="none")

        # compute densities to recover p(x)
        logpxz = -bce.reshape(sample_size, -1, np.prod(self.input_shape)).sum(dim=2)  # log(p(X|Z))

        logpz = self.log_z(Z).reshape(sample_size, -1)  # log(p(Z))

        logrho0 = (
            (
                -0.5
                * (
                    torch.transpose(rho0.unsqueeze(-1), 1, 2)
                    * self.beta_zero_sqrt
                    @ G_inv_rep_0
                    @ rho0.unsqueeze(-1)
                    * self.beta_zero_sqrt
                )
                .squeeze()
                .squeeze()
                - 0.5 * G_log_det_rep_0
            )
            - torch.log(torch.tensor([2 * np.pi]).to(self.device)) * self.latent_dim / 2
        ).reshape(sample_size, -1)

        # log(p(\rho_0))
        logrho = (
            (
                -0.5
                * (
                    torch.transpose(rho.unsqueeze(-1), 1, 2)
                    @ G_inv_rep
                    @ rho.unsqueeze(-1)
                )
                .squeeze()
                .squeeze()
                - 0.5 * G_log_det_rep
            )
            - torch.log(torch.tensor([2 * np.pi]).to(self.device)) * self.latent_dim / 2
        ).reshape(sample_size, -1)
        # log(p(\rho_K))

        logqzx = self.normal.log_prob(Eps) - 0.5 * log_var.sum(dim=1)  # log(q(Z_0|X))

        logpx = (logpxz + logpz + logrho - logrho0 - logqzx).logsumexp(dim=0).mean(
            dim=0
        ) - torch.log(
            torch.Tensor([sample_size]).to(self.device)
        )  # + self.latent_dim /2 * torch.log(self.beta_zero_sqrt ** 2)

        return logpx