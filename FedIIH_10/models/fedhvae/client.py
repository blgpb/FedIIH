import time
import torch
import torch.nn.functional as F
from torch_scatter import scatter_add

from misc.utils import *
from models.nets import *
from modules.federated import ClientModule
from torch_geometric.utils import to_scipy_sparse_matrix
import scipy.sparse as sp
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils import add_self_loops, degree
from numpy.linalg import eig, eigh
from scipy.sparse.linalg import norm

EPS = 1e-5


class HVAE(nn.Module):
    def __init__(self, hidden_dim, nfeat, args, g_id):
        super(HVAE, self).__init__()
        self.node_feature_dims = nfeat
        self.latent_factor_nums = 10
        self.latent_factor_dims = hidden_dim
        self.routit = args.n_routit
        self.gpu_id = g_id

        self.mu_Alpha = nn.Parameter(torch.empty(self.latent_factor_dims),
                                     requires_grad=True)
        nn.init.normal_(self.mu_Alpha)

        self.mu_Beta = nn.Parameter(torch.empty(self.latent_factor_dims), requires_grad=True)
        nn.init.normal_(self.mu_Beta)

        self.mu_Alpha1 = nn.Parameter(torch.empty(self.latent_factor_dims),
                                      requires_grad=True)
        nn.init.normal_(self.mu_Alpha1)

        self.mu_Beta1 = nn.Parameter(torch.empty(self.latent_factor_dims), requires_grad=True)
        nn.init.normal_(self.mu_Beta1)

        self.mu_Alpha2 = nn.Parameter(torch.empty(self.latent_factor_dims),
                                      requires_grad=True)
        nn.init.normal_(self.mu_Alpha2)

        self.mu_Beta2 = nn.Parameter(torch.empty(self.latent_factor_dims), requires_grad=True)
        nn.init.normal_(self.mu_Beta2)

        self.mu_Alpha3 = nn.Parameter(torch.empty(self.latent_factor_dims),
                                      requires_grad=True)
        nn.init.normal_(self.mu_Alpha3)

        self.mu_Beta3 = nn.Parameter(torch.empty(self.latent_factor_dims), requires_grad=True)
        nn.init.normal_(self.mu_Beta3)

        self.mu_Alpha4 = nn.Parameter(torch.empty(self.latent_factor_dims),
                                      requires_grad=True)
        nn.init.normal_(self.mu_Alpha4)

        self.mu_Beta4 = nn.Parameter(torch.empty(self.latent_factor_dims), requires_grad=True)
        nn.init.normal_(self.mu_Beta4)

        self.dropout = args.dropout
        self.loos_ce = nn.BCEWithLogitsLoss()

    def _dropout(self, x):
        return F.dropout(x, self.dropout, training=self.training)

    def sampling(self, z_mu_n, z_logvar_n, z_mu_e, z_logvar_e, z_mu_n1, z_logvar_n1, z_mu_e1, z_logvar_e1, z_mu_n2,
                 z_logvar_n2, z_mu_e2, z_logvar_e2, z_mu_n3, z_logvar_n3, z_mu_e3, z_logvar_e3, z_mu_n4, z_logvar_n4,
                 z_mu_e4, z_logvar_e4):


        noise_n = torch.randn(z_mu_n.shape).cuda(self.gpu_id)

        z_n = z_mu_n + noise_n * torch.exp(0.5 * z_logvar_n)


        noise_e = torch.randn(z_mu_e.shape).cuda(self.gpu_id)

        z_e = z_mu_e + noise_e * torch.exp(0.5 * z_logvar_e)


        noise_n1 = torch.randn(z_mu_n1.shape).cuda(self.gpu_id)

        z_n1 = z_mu_n1 + noise_n1 * torch.exp(0.5 * z_logvar_n1)


        noise_e1 = torch.randn(z_mu_e1.shape).cuda(self.gpu_id)

        z_e1 = z_mu_e1 + noise_e1 * torch.exp(0.5 * z_logvar_e1)


        noise_n2 = torch.randn(z_mu_n2.shape).cuda(self.gpu_id)

        z_n2 = z_mu_n2 + noise_n2 * torch.exp(0.5 * z_logvar_n2)


        noise_e2 = torch.randn(z_mu_e2.shape).cuda(self.gpu_id)

        z_e2 = z_mu_e2 + noise_e2 * torch.exp(0.5 * z_logvar_e2)

        noise_n3 = torch.randn(z_mu_n3.shape).cuda(self.gpu_id)

        z_n3 = z_mu_n3 + noise_n3 * torch.exp(0.5 * z_logvar_n3)

        noise_e3 = torch.randn(z_mu_e3.shape).cuda(self.gpu_id)

        z_e3 = z_mu_e3 + noise_e3 * torch.exp(0.5 * z_logvar_e3)

        noise_n4 = torch.randn(z_mu_n4.shape).cuda(self.gpu_id)

        z_n4 = z_mu_n4 + noise_n4 * torch.exp(0.5 * z_logvar_n4)

        noise_e4 = torch.randn(z_mu_e4.shape).cuda(self.gpu_id)

        z_e4 = z_mu_e4 + noise_e4 * torch.exp(0.5 * z_logvar_e4)

        return z_n, z_e, z_n1, z_e1, z_n2, z_e2, z_n3, z_e3, z_n4, z_e4

    def decode(self, z_n, z_e, z_n1, z_e1, z_n2, z_e2, z_n3, z_e3, z_n4, z_e4):

        z = torch.cat((z_n, z_e, z_n1, z_e1, z_n2, z_e2, z_n3, z_e3, z_n4, z_e4), dim=1)
        A_pred = torch.matmul(z, z.t())

        return A_pred

    def loss_function(self, batch, z_mu_n, z_logvar_n, z_mu_e, z_logvar_e, z_mu_n1, z_logvar_n1, z_mu_e1, z_logvar_e1,
                      z_mu_n2, z_logvar_n2, z_mu_e2, z_logvar_e2, z_mu_n3, z_logvar_n3, z_mu_e3, z_logvar_e3,
                      z_mu_n4, z_logvar_n4, z_mu_e4, z_logvar_e4,
                      Alpha_mu, Beta_mu, Alpha_mu1, Beta_mu1, Alpha_mu2, Beta_mu2, Alpha_mu3, Beta_mu3, Alpha_mu4,
                      Beta_mu4,
                      edge_logits,
                      **kwargs) -> dict:

        log_pmu_Alpha = torch.mean(self.log_normal(self.mu_Alpha), dim=0)
        Alpha_logvar = torch.ones_like(Alpha_mu) * np.log(np.power(0.5, 2))

        log_pmu_Beta = torch.mean(self.log_normal(self.mu_Beta), dim=0)
        Beta_logvar = torch.ones_like(Beta_mu) * np.log(np.power(0.5, 2))

        log_pmu_Alpha1 = torch.mean(self.log_normal(self.mu_Alpha1), dim=0)
        Alpha_logvar1 = torch.ones_like(Alpha_mu1) * np.log(np.power(0.5, 2))

        log_pmu_Beta1 = torch.mean(self.log_normal(self.mu_Beta1), dim=0)
        Beta_logvar1 = torch.ones_like(Beta_mu1) * np.log(np.power(0.5, 2))

        log_pmu_Alpha2 = torch.mean(self.log_normal(self.mu_Alpha2), dim=0)
        Alpha_logvar2 = torch.ones_like(Alpha_mu2) * np.log(np.power(0.5, 2))

        log_pmu_Beta2 = torch.mean(self.log_normal(self.mu_Beta2), dim=0)
        Beta_logvar2 = torch.ones_like(Beta_mu2) * np.log(np.power(0.5, 2))

        log_pmu_Alpha3 = torch.mean(self.log_normal(self.mu_Alpha3), dim=0)
        Alpha_logvar3 = torch.ones_like(Alpha_mu3) * np.log(np.power(0.5, 2))

        log_pmu_Beta3 = torch.mean(self.log_normal(self.mu_Beta3), dim=0)
        Beta_logvar3 = torch.ones_like(Beta_mu3) * np.log(np.power(0.5, 2))

        log_pmu_Alpha4 = torch.mean(self.log_normal(self.mu_Alpha4), dim=0)
        Alpha_logvar4 = torch.ones_like(Alpha_mu4) * np.log(np.power(0.5, 2))

        log_pmu_Beta4 = torch.mean(self.log_normal(self.mu_Beta4), dim=0)
        Beta_logvar4 = torch.ones_like(Beta_mu4) * np.log(np.power(0.5, 2))


        extra_kl_Alpha = torch.mean(self.kld(self.mu_Alpha, Alpha_logvar.cuda(self.gpu_id), Alpha_mu.cuda(self.gpu_id), Alpha_logvar.cuda(self.gpu_id)))
        extra_kl_Beta = torch.mean(self.kld(self.mu_Beta, Beta_logvar.cuda(self.gpu_id), Beta_mu.cuda(self.gpu_id), Beta_logvar.cuda(self.gpu_id)))

        extra_kl_Alpha1 = torch.mean(
            self.kld(self.mu_Alpha1, Alpha_logvar1.cuda(self.gpu_id), Alpha_mu1.cuda(self.gpu_id), Alpha_logvar1.cuda(self.gpu_id)))
        extra_kl_Beta1 = torch.mean(self.kld(self.mu_Beta1, Beta_logvar1.cuda(self.gpu_id), Beta_mu1.cuda(self.gpu_id), Beta_logvar1.cuda(self.gpu_id)))

        extra_kl_Alpha2 = torch.mean(
            self.kld(self.mu_Alpha2, Alpha_logvar2.cuda(self.gpu_id), Alpha_mu2.cuda(self.gpu_id), Alpha_logvar2.cuda(self.gpu_id)))
        extra_kl_Beta2 = torch.mean(self.kld(self.mu_Beta2, Beta_logvar2.cuda(self.gpu_id), Beta_mu2.cuda(self.gpu_id), Beta_logvar2.cuda(self.gpu_id)))

        extra_kl_Alpha3 = torch.mean(
            self.kld(self.mu_Alpha3, Alpha_logvar3.cuda(self.gpu_id), Alpha_mu3.cuda(self.gpu_id), Alpha_logvar3.cuda(self.gpu_id)))
        extra_kl_Beta3 = torch.mean(self.kld(self.mu_Beta3, Beta_logvar3.cuda(self.gpu_id), Beta_mu3.cuda(self.gpu_id), Beta_logvar3.cuda(self.gpu_id)))

        extra_kl_Alpha4 = torch.mean(
            self.kld(self.mu_Alpha4, Alpha_logvar4.cuda(self.gpu_id), Alpha_mu4.cuda(self.gpu_id), Alpha_logvar4.cuda(self.gpu_id)))
        extra_kl_Beta4 = torch.mean(self.kld(self.mu_Beta4, Beta_logvar4.cuda(self.gpu_id), Beta_mu4.cuda(self.gpu_id), Beta_logvar4.cuda(self.gpu_id)))


        edge_index = batch.edge_index

        num_nodes = batch.num_nodes
        adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float).cuda(self.gpu_id)


        for i, j in edge_index.t().tolist():
            adj_matrix[i, j] = 1.0
            adj_matrix[j, i] = 1.0


        logpx_z = torch.mean(self.loos_ce(edge_logits, adj_matrix.view(num_nodes, num_nodes)))


        kl_structure = torch.mean(self.kld(z_mu_e, z_logvar_e, self.mu_Alpha, Alpha_logvar.cuda(self.gpu_id)))


        kl_semantic = torch.mean(self.kld(z_mu_n, z_logvar_n, self.mu_Beta, Beta_logvar.cuda(self.gpu_id)))


        kl_structure1 = torch.mean(self.kld(z_mu_e1, z_logvar_e1, self.mu_Alpha1, Alpha_logvar1.cuda(self.gpu_id)))


        kl_semantic1 = torch.mean(self.kld(z_mu_n1, z_logvar_n1, self.mu_Beta1, Beta_logvar1.cuda(self.gpu_id)))


        kl_structure2 = torch.mean(self.kld(z_mu_e2, z_logvar_e2, self.mu_Alpha2, Alpha_logvar2.cuda(self.gpu_id)))


        kl_semantic2 = torch.mean(self.kld(z_mu_n2, z_logvar_n2, self.mu_Beta2, Beta_logvar2.cuda(self.gpu_id)))


        kl_structure3 = torch.mean(self.kld(z_mu_e3, z_logvar_e3, self.mu_Alpha3, Alpha_logvar3.cuda(self.gpu_id)))


        kl_semantic3 = torch.mean(self.kld(z_mu_n3, z_logvar_n3, self.mu_Beta3, Beta_logvar3.cuda(self.gpu_id)))


        kl_structure4 = torch.mean(self.kld(z_mu_e4, z_logvar_e4, self.mu_Alpha4, Alpha_logvar4.cuda(self.gpu_id)))

        kl_semantic4 = torch.mean(self.kld(z_mu_n4, z_logvar_n4, self.mu_Beta4, Beta_logvar4.cuda(self.gpu_id)))

        l_elbo = torch.mean(
            log_pmu_Alpha + extra_kl_Alpha + log_pmu_Beta + extra_kl_Beta + log_pmu_Alpha1 + extra_kl_Alpha1 + log_pmu_Beta1 + extra_kl_Beta1 + log_pmu_Alpha2 + extra_kl_Alpha2 + log_pmu_Beta2 + extra_kl_Beta2 + log_pmu_Alpha3 + extra_kl_Alpha3 + log_pmu_Beta3 + extra_kl_Beta3 + log_pmu_Alpha4 + extra_kl_Alpha4 + log_pmu_Beta4 + extra_kl_Beta4 + logpx_z + kl_structure + kl_semantic + kl_structure1 + kl_semantic1 + kl_structure2 + kl_semantic2 + kl_structure3 + kl_semantic3 + kl_structure4 + kl_semantic4)


        return l_elbo


    def log_gauss(self, mu, logvar, x):

        return -0.5 * (np.log(2 * np.pi) + logvar + torch.pow((x - mu), 2) / torch.exp(logvar))

    def kld(self, mu, logvar, q_mu, q_logvar):
        return -0.5 * (1 + logvar - q_logvar - (torch.pow(mu - q_mu, 2) + torch.exp(logvar)) / torch.exp(q_logvar))

    def log_normal(self, x):
        return -0.5 * (np.log(2 * np.pi) + torch.pow(x, 2))


class Client(ClientModule):

    def __init__(self, args, w_id, g_id, sd):
        super(Client, self).__init__(args, w_id, g_id, sd)

        self.model = DisentangledGNN(self.args.n_feat, self.args.n_clss, args).cuda(g_id)
        self.parameters = list(self.model.parameters())

        self.vae = HVAE(self.args.n_latentdims, self.args.n_feat, args, g_id).cuda(g_id)
        self.vae_parameters = list(self.vae.parameters())

        self.Alpha_mu = torch.randn(self.args.n_latentdims).cuda(g_id)
        self.Beta_mu = torch.randn(self.args.n_latentdims).cuda(g_id)

        self.Alpha_mu1 = torch.randn(self.args.n_latentdims).cuda(g_id)
        self.Beta_mu1 = torch.randn(self.args.n_latentdims).cuda(g_id)

        self.Alpha_mu2 = torch.randn(self.args.n_latentdims).cuda(g_id)
        self.Beta_mu2 = torch.randn(self.args.n_latentdims).cuda(g_id)

        self.Alpha_mu3 = torch.randn(self.args.n_latentdims).cuda(g_id)
        self.Beta_mu3 = torch.randn(self.args.n_latentdims).cuda(g_id)

        self.Alpha_mu4 = torch.randn(self.args.n_latentdims).cuda(g_id)
        self.Beta_mu4 = torch.randn(self.args.n_latentdims).cuda(g_id)
        self.init_state()

    def init_state(self):
        self.optimizer = torch.optim.Adam(self.parameters, lr=self.args.base_lr, weight_decay=self.args.weight_decay)

        self.log = {
            'lr': [], 'train_lss': [],
            'ep_local_val_lss': [], 'ep_local_val_acc': [],
            'rnd_local_val_lss': [], 'rnd_local_val_acc': [],
            'ep_local_test_lss': [], 'ep_local_test_acc': [],
            'rnd_local_test_lss': [], 'rnd_local_test_acc': [],
            'have_trained_rounds': [],
        }

        self.optimizer_vae = torch.optim.Adam(self.vae_parameters, lr=self.args.base_lr,
                                              weight_decay=self.args.weight_decay)

    def save_state(self):
        torch_save(self.args.checkpt_path, f'{self.client_id}_state.pt', {
            'optimizer_vae': self.optimizer_vae.state_dict(),
            'vae': get_state_dict(self.vae),

            'optimizer': self.optimizer.state_dict(),
            'model': get_state_dict(self.model),

            'log': self.log,

            'z_mu_n': self.z_mu_n,
            'z_logvar_n': self.z_logvar_n,

            'z_mu_e': self.z_mu_e,
            'z_logvar_e': self.z_logvar_e,

            'Alpha_mu': self.Alpha_mu,
            'Beta_mu': self.Beta_mu,

            'z_mu_n1': self.z_mu_n1,
            'z_logvar_n1': self.z_logvar_n1,

            'z_mu_e1': self.z_mu_e1,
            'z_logvar_e1': self.z_logvar_e1,

            'Alpha_mu1': self.Alpha_mu1,
            'Beta_mu1': self.Beta_mu1,

            'z_mu_n2': self.z_mu_n2,
            'z_logvar_n2': self.z_logvar_n2,

            'z_mu_e2': self.z_mu_e2,
            'z_logvar_e2': self.z_logvar_e2,

            'Alpha_mu2': self.Alpha_mu2,
            'Beta_mu2': self.Beta_mu2,

            'z_mu_n3': self.z_mu_n3,
            'z_logvar_n3': self.z_logvar_n3,

            'z_mu_e3': self.z_mu_e3,
            'z_logvar_e3': self.z_logvar_e3,

            'Alpha_mu3': self.Alpha_mu3,
            'Beta_mu3': self.Beta_mu3,

            'z_mu_n4': self.z_mu_n4,
            'z_logvar_n4': self.z_logvar_n4,

            'z_mu_e4': self.z_mu_e4,
            'z_logvar_e4': self.z_logvar_e4,

            'Alpha_mu4': self.Alpha_mu4,
            'Beta_mu4': self.Beta_mu4,

        })

    def load_state(self):
        loaded = torch_load(self.args.checkpt_path, f'{self.client_id}_state.pt')

        set_state_dict(self.vae, loaded['vae'], self.gpu_id)
        self.optimizer_vae.load_state_dict(loaded['optimizer_vae'])

        set_state_dict(self.model, loaded['model'], self.gpu_id)
        self.optimizer.load_state_dict(loaded['optimizer'])

        self.log = loaded['log']

        self.z_mu_n = loaded['z_mu_n']
        self.z_logvar_n = loaded['z_logvar_n']

        self.z_mu_e = loaded['z_mu_e']
        self.z_logvar_e = loaded['z_logvar_e']

        self.Alpha_mu = loaded['Alpha_mu']
        self.Beta_mu = loaded['Beta_mu']

        self.z_mu_n1 = loaded['z_mu_n1']
        self.z_logvar_n1 = loaded['z_logvar_n1']

        self.z_mu_e1 = loaded['z_mu_e1']
        self.z_logvar_e1 = loaded['z_logvar_e1']

        self.Alpha_mu1 = loaded['Alpha_mu1']
        self.Beta_mu1 = loaded['Beta_mu1']

        self.z_mu_n2 = loaded['z_mu_n2']
        self.z_logvar_n2 = loaded['z_logvar_n2']

        self.z_mu_e2 = loaded['z_mu_e2']
        self.z_logvar_e2 = loaded['z_logvar_e2']

        self.Alpha_mu2 = loaded['Alpha_mu2']
        self.Beta_mu2 = loaded['Beta_mu2']

        self.z_mu_n3 = loaded['z_mu_n3']
        self.z_logvar_n3 = loaded['z_logvar_n3']

        self.z_mu_e3 = loaded['z_mu_e3']
        self.z_logvar_e3 = loaded['z_logvar_e3']

        self.Alpha_mu3 = loaded['Alpha_mu3']
        self.Beta_mu3 = loaded['Beta_mu3']

        self.z_mu_n4 = loaded['z_mu_n4']
        self.z_logvar_n4 = loaded['z_logvar_n4']

        self.z_mu_e4 = loaded['z_mu_e4']
        self.z_logvar_e4 = loaded['z_logvar_e4']

        self.Alpha_mu4 = loaded['Alpha_mu4']
        self.Beta_mu4 = loaded['Beta_mu4']

    def on_receive_message(self, curr_rnd):
        self.curr_rnd = curr_rnd


        if self.curr_rnd != 0 and self.curr_rnd != self.args.trained_rounds:
            self.update(self.sd[f'personalized_{self.client_id}' \
                if (f'personalized_{self.client_id}' in self.sd) else 'global'])
        else:
            self.update(self.sd[f'global'])



        if self.curr_rnd != 0 and self.curr_rnd != self.args.trained_rounds:
            self.Alpha_mu = torch.tensor(self.sd[f'Alpha_mu']).cuda(self.gpu_id)
            self.Beta_mu = torch.tensor(self.sd[f'Beta_mu']).cuda(self.gpu_id)
            self.Alpha_mu1 = torch.tensor(self.sd[f'Alpha_mu1']).cuda(self.gpu_id)
            self.Beta_mu1 = torch.tensor(self.sd[f'Beta_mu1']).cuda(self.gpu_id)
            self.Alpha_mu2 = torch.tensor(self.sd[f'Alpha_mu2']).cuda(self.gpu_id)
            self.Beta_mu2 = torch.tensor(self.sd[f'Beta_mu2']).cuda(self.gpu_id)
            self.Alpha_mu3 = torch.tensor(self.sd[f'Alpha_mu3']).cuda(self.gpu_id)
            self.Beta_mu3 = torch.tensor(self.sd[f'Beta_mu3']).cuda(self.gpu_id)
            self.Alpha_mu4 = torch.tensor(self.sd[f'Alpha_mu4']).cuda(self.gpu_id)
            self.Beta_mu4 = torch.tensor(self.sd[f'Beta_mu4']).cuda(self.gpu_id)

    def update(self, update):
        set_state_dict(self.model, update['model'], self.gpu_id, skip_stat=True, skip_mask=True)

    def update_gnn(self, update):
        set_state_dict(self.gnn, update['gnn'], self.gpu_id, skip_stat=True, skip_mask=True)

    def on_round_begin(self):
        self.train()
        self.transfer_to_server()

    def train(self):
        for ep in range(self.args.n_eps):
            self.model.train()

            for _, batch in enumerate(self.loader.pa_loader):
                self.optimizer.zero_grad()

                batch = batch.cuda(self.gpu_id)

                y_hat = self.model(batch)

                if self.args.dataset in ['Minesweeper', 'Tolokers', 'Questions']:
                    train_lss = F.binary_cross_entropy_with_logits(y_hat[batch.train_mask].view(-1), batch.y[batch.train_mask].view(-1))
                else:
                    train_lss = F.cross_entropy(y_hat[batch.train_mask], batch.y[batch.train_mask])

                train_lss.backward()


                self.optimizer.step()

        for ep in range(self.args.n_eps):
            st = time.time()
            self.vae.train()

            for _, batch in enumerate(self.loader.pa_loader):
                self.optimizer_vae.zero_grad()

                batch = batch.cuda(self.gpu_id)

                if self.curr_rnd == 0:
                    self.Alpha_mu = torch.randn(self.args.n_latentdims).cuda(self.gpu_id)
                    self.Beta_mu = torch.randn(self.args.n_latentdims).cuda(self.gpu_id)
                    self.Alpha_mu1 = torch.randn(self.args.n_latentdims).cuda(self.gpu_id)
                    self.Beta_mu1 = torch.randn(self.args.n_latentdims).cuda(self.gpu_id)
                    self.Alpha_mu2 = torch.randn(self.args.n_latentdims).cuda(self.gpu_id)
                    self.Beta_mu2 = torch.randn(self.args.n_latentdims).cuda(self.gpu_id)
                    self.Alpha_mu3 = torch.randn(self.args.n_latentdims).cuda(self.gpu_id)
                    self.Beta_mu3 = torch.randn(self.args.n_latentdims).cuda(self.gpu_id)
                    self.Alpha_mu4 = torch.randn(self.args.n_latentdims).cuda(self.gpu_id)
                    self.Beta_mu4 = torch.randn(self.args.n_latentdims).cuda(self.gpu_id)



                self.z_mu_n, self.z_logvar_n, self.z_mu_e, self.z_logvar_e, self.z_mu_n1, self.z_logvar_n1, self.z_mu_e1, self.z_logvar_e1, self.z_mu_n2, self.z_logvar_n2, self.z_mu_e2, self.z_logvar_e2, self.z_mu_n3, self.z_logvar_n3, self.z_mu_e3, self.z_logvar_e3, self.z_mu_n4, self.z_logvar_n4, self.z_mu_e4, self.z_logvar_e4 = self.model.encode_for_HVAE(
                    batch)

                z_n, z_e, z_n1, z_e1, z_n2, z_e2, z_n3, z_e3, z_n4, z_e4 = self.vae.sampling(self.z_mu_n, self.z_logvar_n,
                                                                                 self.z_mu_e,
                                                                                 self.z_logvar_e,
                                                                                 self.z_mu_n1, self.z_logvar_n1,
                                                                                 self.z_mu_e1,
                                                                                 self.z_logvar_e1,
                                                                                 self.z_mu_n2, self.z_logvar_n2,
                                                                                 self.z_mu_e2,
                                                                                 self.z_logvar_e2,
                                                                                 self.z_mu_n3, self.z_logvar_n3,
                                                                                 self.z_mu_e3,
                                                                                 self.z_logvar_e3,
                                                                                 self.z_mu_n4, self.z_logvar_n4,
                                                                                 self.z_mu_e4,
                                                                                 self.z_logvar_e4)


                edge_logits = self.vae.decode(z_n, z_e, z_n1, z_e1, z_n2, z_e2, z_n3, z_e3, z_n4, z_e4)


                loss = self.vae.loss_function(batch, self.z_mu_n, self.z_logvar_n, self.z_mu_e, self.z_logvar_e,
                                              self.z_mu_n1, self.z_logvar_n1, self.z_mu_e1, self.z_logvar_e1,
                                              self.z_mu_n2, self.z_logvar_n2, self.z_mu_e2, self.z_logvar_e2,
                                              self.z_mu_n3, self.z_logvar_n3, self.z_mu_e3, self.z_logvar_e3,
                                              self.z_mu_n4, self.z_logvar_n4, self.z_mu_e4, self.z_logvar_e4,
                                              self.Alpha_mu, self.Beta_mu, self.Alpha_mu1, self.Beta_mu1,
                                              self.Alpha_mu2, self.Beta_mu2, self.Alpha_mu3, self.Beta_mu3,
                                              self.Alpha_mu4, self.Beta_mu4,
                                              edge_logits)

                loss.backward()

                self.optimizer_vae.step()

            val_local_acc, val_local_lss = self.validate(mode='valid')
            test_local_acc, test_local_lss = self.validate(mode='test')
            self.logger.print(
                f'rnd:{self.curr_rnd + 1}, ep:{ep + 1}, '
                f'val_local_loss: {val_local_lss.item():.4f}, val_local_acc: {val_local_acc:.4f}, lr: {self.get_lr()} ({time.time() - st:.2f}s)'
            )
            self.log['train_lss'].append(train_lss.item())
            self.log['ep_local_val_acc'].append(val_local_acc)
            self.log['ep_local_val_lss'].append(val_local_lss)
            self.log['ep_local_test_acc'].append(test_local_acc)
            self.log['ep_local_test_lss'].append(test_local_lss)
        self.log['rnd_local_val_acc'].append(val_local_acc)
        self.log['rnd_local_val_lss'].append(val_local_lss)
        self.log['rnd_local_test_acc'].append(test_local_acc)
        self.log['rnd_local_test_lss'].append(test_local_lss)
        self.log['have_trained_rounds'].append(self.curr_rnd + 1)
        self.save_log()

    def transfer_to_server(self):
        self.sd[self.client_id] = {
            'model': get_state_dict(self.model),

            'train_size': len(self.loader.partition),

            'z_mu_n': np.mean(self.z_mu_n.detach().cpu().numpy(), axis=0),
            'z_logvar_n': np.mean(self.z_logvar_n.detach().cpu().numpy(), axis=0),

            'z_mu_e': np.mean(self.z_mu_e.detach().cpu().numpy(), axis=0),
            'z_logvar_e': np.mean(self.z_logvar_e.detach().cpu().numpy(), axis=0),

            'z_mu_n1': np.mean(self.z_mu_n1.detach().cpu().numpy(), axis=0),
            'z_logvar_n1': np.mean(self.z_logvar_n1.detach().cpu().numpy(), axis=0),

            'z_mu_e1': np.mean(self.z_mu_e1.detach().cpu().numpy(), axis=0),
            'z_logvar_e1': np.mean(self.z_logvar_e1.detach().cpu().numpy(), axis=0),

            'z_mu_n2': np.mean(self.z_mu_n2.detach().cpu().numpy(), axis=0),
            'z_logvar_n2': np.mean(self.z_logvar_n2.detach().cpu().numpy(), axis=0),

            'z_mu_e2': np.mean(self.z_mu_e2.detach().cpu().numpy(), axis=0),
            'z_logvar_e2': np.mean(self.z_logvar_e2.detach().cpu().numpy(), axis=0),

            'z_mu_n3': np.mean(self.z_mu_n3.detach().cpu().numpy(), axis=0),
            'z_logvar_n3': np.mean(self.z_logvar_n3.detach().cpu().numpy(), axis=0),

            'z_mu_e3': np.mean(self.z_mu_e3.detach().cpu().numpy(), axis=0),
            'z_logvar_e3': np.mean(self.z_logvar_e3.detach().cpu().numpy(), axis=0),

            'z_mu_n4': np.mean(self.z_mu_n4.detach().cpu().numpy(), axis=0),
            'z_logvar_n4': np.mean(self.z_logvar_n4.detach().cpu().numpy(), axis=0),

            'z_mu_e4': np.mean(self.z_mu_e4.detach().cpu().numpy(), axis=0),
            'z_logvar_e4': np.mean(self.z_logvar_e4.detach().cpu().numpy(), axis=0),
        }

    @torch.no_grad()
    def validation_step(self, batch, mask=None):
        self.model.eval()
        y_hat = self.model(batch)
        if torch.sum(mask).item() == 0: return y_hat, 0.0
        if self.args.dataset in ['Minesweeper', 'Tolokers', 'Questions']:
            lss = F.binary_cross_entropy_with_logits(y_hat[mask].view(-1), batch.y[mask].view(-1))
        else:
            lss = F.cross_entropy(y_hat[mask], batch.y[mask])
        return y_hat, lss.item()

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']
