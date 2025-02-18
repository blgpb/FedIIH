import time
import numpy as np
import torch

from misc.utils import *
from models.nets import *
from modules.federated import ServerModule
from torch import nn
from torch_geometric.nn import GCNConv


def kl_divergence(mu_A, logvar_A, mu_B, logvar_B):
    var_A = np.exp(logvar_A)
    var_B = np.exp(logvar_B)

    kl = np.sum(0.5 * (logvar_B - logvar_A + (var_A + (mu_A - mu_B) ** 2) / var_B - 1))

    return kl


def js_divergence(mu_A, logvar_A, mu_B, logvar_B):
    var_A = np.exp(logvar_A)
    var_B = np.exp(logvar_B)

    mu_M = 0.5 * (mu_A + mu_B)
    var_M = 0.5 * (var_A + var_B)

    kl_AM = kl_divergence(mu_A, logvar_A, mu_M, np.log(var_M))
    kl_BM = kl_divergence(mu_B, logvar_B, mu_M, np.log(var_M))

    js = 0.5 * (kl_AM + kl_BM)

    return js


class Server(ServerModule):
    def __init__(self, args, sd, gpu_server):
        super(Server, self).__init__(args, sd, gpu_server)

        self.model = DisentangledGNN(self.args.n_feat, self.args.n_clss, args).cuda(self.gpu_id)

        self.update_lists = []
        self.sim_matrices_n = []
        self.sim_matrices_e = []
        self.sim_matrices_n1 = []
        self.sim_matrices_e1 = []
        self.sim_matrices_n2 = []
        self.sim_matrices_e2 = []
        self.sim_matrices_n3 = []
        self.sim_matrices_e3 = []
        self.sim_matrices_n4 = []
        self.sim_matrices_e4 = []

    def on_round_begin(self, curr_rnd):
        self.round_begin = time.time()
        self.curr_rnd = curr_rnd
        if self.curr_rnd != 0:
            self.load_state()
        self.sd['global'] = self.get_weights()

    def load_state(self):
        loaded = torch_load(self.args.checkpt_path, 'server_state.pt')

        set_state_dict(self.model, loaded['model'], self.gpu_id)

        self.sim_matrices_n = loaded['sim_matrices_n']
        self.sim_matrices_e = loaded['sim_matrices_e']
        self.sim_matrices_n1 = loaded['sim_matrices_n1']
        self.sim_matrices_e1 = loaded['sim_matrices_e1']
        self.sim_matrices_n2 = loaded['sim_matrices_n2']
        self.sim_matrices_e2 = loaded['sim_matrices_e2']
        self.sim_matrices_n3 = loaded['sim_matrices_n3']
        self.sim_matrices_e3 = loaded['sim_matrices_e3']
        self.sim_matrices_n4 = loaded['sim_matrices_n4']
        self.sim_matrices_e4 = loaded['sim_matrices_e4']

    def get_weights(self):
        return {
            'model': get_state_dict(self.model)
        }

    def on_round_complete(self, updated):
        self.update(updated)
        self.save_state()

    def update(self, updated):
        st = time.time()

        local_weights = []

        local_train_sizes = []

        clients_z_mu_n, clients_z_logvar_n = [], []
        clients_z_mu_e, clients_z_logvar_e = [], []
        clients_z_mu_n1, clients_z_logvar_n1 = [], []
        clients_z_mu_e1, clients_z_logvar_e1 = [], []
        clients_z_mu_n2, clients_z_logvar_n2 = [], []
        clients_z_mu_e2, clients_z_logvar_e2 = [], []
        clients_z_mu_n3, clients_z_logvar_n3 = [], []
        clients_z_mu_e3, clients_z_logvar_e3 = [], []
        clients_z_mu_n4, clients_z_logvar_n4 = [], []
        clients_z_mu_e4, clients_z_logvar_e4 = [], []

        for c_id in updated:
            clients_z_mu_n.append(self.sd[c_id]['z_mu_n'])
            clients_z_logvar_n.append(self.sd[c_id]['z_logvar_n'])

            clients_z_mu_e.append(self.sd[c_id]['z_mu_e'])
            clients_z_logvar_e.append(self.sd[c_id]['z_logvar_e'])

            clients_z_mu_n1.append(self.sd[c_id]['z_mu_n1'])
            clients_z_logvar_n1.append(self.sd[c_id]['z_logvar_n1'])

            clients_z_mu_e1.append(self.sd[c_id]['z_mu_e1'])
            clients_z_logvar_e1.append(self.sd[c_id]['z_logvar_e1'])

            clients_z_mu_n2.append(self.sd[c_id]['z_mu_n2'])
            clients_z_logvar_n2.append(self.sd[c_id]['z_logvar_n2'])

            clients_z_mu_e2.append(self.sd[c_id]['z_mu_e2'])
            clients_z_logvar_e2.append(self.sd[c_id]['z_logvar_e2'])

            clients_z_mu_n3.append(self.sd[c_id]['z_mu_n3'])
            clients_z_logvar_n3.append(self.sd[c_id]['z_logvar_n3'])

            clients_z_mu_e3.append(self.sd[c_id]['z_mu_e3'])
            clients_z_logvar_e3.append(self.sd[c_id]['z_logvar_e3'])

            clients_z_mu_n4.append(self.sd[c_id]['z_mu_n4'])
            clients_z_logvar_n4.append(self.sd[c_id]['z_logvar_n4'])

            clients_z_mu_e4.append(self.sd[c_id]['z_mu_e4'])
            clients_z_logvar_e4.append(self.sd[c_id]['z_logvar_e4'])

            local_weights.append(self.sd[c_id]['model'].copy())

            local_train_sizes.append(self.sd[c_id]['train_size'])

            del self.sd[c_id]

        n_connected = round(self.args.n_clients * self.args.frac)
        assert n_connected == len(clients_z_mu_n)

        sim_matrix_n = np.empty(shape=(n_connected, n_connected))
        sim_matrix_e = np.empty(shape=(n_connected, n_connected))
        sim_matrix_n1 = np.empty(shape=(n_connected, n_connected))
        sim_matrix_e1 = np.empty(shape=(n_connected, n_connected))
        sim_matrix_n2 = np.empty(shape=(n_connected, n_connected))
        sim_matrix_e2 = np.empty(shape=(n_connected, n_connected))
        sim_matrix_n3 = np.empty(shape=(n_connected, n_connected))
        sim_matrix_e3 = np.empty(shape=(n_connected, n_connected))
        sim_matrix_n4 = np.empty(shape=(n_connected, n_connected))
        sim_matrix_e4 = np.empty(shape=(n_connected, n_connected))

        self.sd[f'Alpha_mu'] = np.sum(clients_z_mu_e, axis=0) / (n_connected + 0.25)
        self.sd[f'Beta_mu'] = np.sum(clients_z_mu_n, axis=0) / (n_connected + 0.25)
        self.sd[f'Alpha_mu1'] = np.sum(clients_z_mu_e1, axis=0) / (n_connected + 0.25)
        self.sd[f'Beta_mu1'] = np.sum(clients_z_mu_n1, axis=0) / (n_connected + 0.25)
        self.sd[f'Alpha_mu2'] = np.sum(clients_z_mu_e2, axis=0) / (n_connected + 0.25)
        self.sd[f'Beta_mu2'] = np.sum(clients_z_mu_n2, axis=0) / (n_connected + 0.25)
        self.sd[f'Alpha_mu3'] = np.sum(clients_z_mu_e3, axis=0) / (n_connected + 0.25)
        self.sd[f'Beta_mu3'] = np.sum(clients_z_mu_n3, axis=0) / (n_connected + 0.25)
        self.sd[f'Alpha_mu4'] = np.sum(clients_z_mu_e4, axis=0) / (n_connected + 0.25)
        self.sd[f'Beta_mu4'] = np.sum(clients_z_mu_n4, axis=0) / (n_connected + 0.25)

        for i in range(n_connected):
            for j in range(n_connected):

                mu_A = clients_z_mu_n[i]
                logvar_A = clients_z_logvar_n[i]

                mu_B = clients_z_mu_n[j]
                logvar_B = clients_z_logvar_n[j]

                js = js_divergence(mu_A, logvar_A, mu_B, logvar_B)
                sim_matrix_n[i, j] = 1 - js / np.log(2)

                mu_A = clients_z_mu_e[i]
                logvar_A = clients_z_logvar_e[i]

                mu_B = clients_z_mu_e[j]
                logvar_B = clients_z_logvar_e[j]

                js = js_divergence(mu_A, logvar_A, mu_B, logvar_B)
                sim_matrix_e[i, j] = 1 - js / np.log(2)

                mu_A1 = clients_z_mu_n1[i]
                logvar_A1 = clients_z_logvar_n1[i]

                mu_B1 = clients_z_mu_n1[j]
                logvar_B1 = clients_z_logvar_n1[j]

                js = js_divergence(mu_A1, logvar_A1, mu_B1, logvar_B1)
                sim_matrix_n1[i, j] = 1 - js / np.log(2)

                mu_A1 = clients_z_mu_e1[i]
                logvar_A1 = clients_z_logvar_e1[i]

                mu_B1 = clients_z_mu_e1[j]
                logvar_B1 = clients_z_logvar_e1[j]

                js = js_divergence(mu_A1, logvar_A1, mu_B1, logvar_B1)
                sim_matrix_e1[i, j] = 1 - js / np.log(2)

                mu_A2 = clients_z_mu_n2[i]
                logvar_A2 = clients_z_logvar_n2[i]

                mu_B2 = clients_z_mu_n2[j]
                logvar_B2 = clients_z_logvar_n2[j]

                js = js_divergence(mu_A2, logvar_A2, mu_B2, logvar_B2)
                sim_matrix_n2[i, j] = 1 - js / np.log(2)


                mu_A2 = clients_z_mu_e2[i]
                logvar_A2 = clients_z_logvar_e2[i]

                mu_B2 = clients_z_mu_e2[j]
                logvar_B2 = clients_z_logvar_e2[j]

                js = js_divergence(mu_A2, logvar_A2, mu_B2, logvar_B2)
                sim_matrix_e2[i, j] = 1 - js / np.log(2)


                mu_A3 = clients_z_mu_n3[i]
                logvar_A3 = clients_z_logvar_n3[i]

                mu_B3 = clients_z_mu_n3[j]
                logvar_B3 = clients_z_logvar_n3[j]

                js = js_divergence(mu_A3, logvar_A3, mu_B3, logvar_B3)
                sim_matrix_n3[i, j] = 1 - js / np.log(2)

                # 8.
                mu_A3 = clients_z_mu_e3[i]
                logvar_A3 = clients_z_logvar_e3[i]

                mu_B3 = clients_z_mu_e3[j]
                logvar_B3 = clients_z_logvar_e3[j]

                js = js_divergence(mu_A3, logvar_A3, mu_B3, logvar_B3)
                sim_matrix_e3[i, j] = 1 - js / np.log(2)

                # 9.
                mu_A4 = clients_z_mu_n4[i]
                logvar_A4 = clients_z_logvar_n4[i]

                mu_B4 = clients_z_mu_n4[j]
                logvar_B4 = clients_z_logvar_n4[j]

                js = js_divergence(mu_A4, logvar_A4, mu_B4, logvar_B4)
                sim_matrix_n4[i, j] = 1 - js / np.log(2)

                # 10.
                mu_A4 = clients_z_mu_e4[i]
                logvar_A4 = clients_z_logvar_e4[i]

                mu_B4 = clients_z_mu_e4[j]
                logvar_B4 = clients_z_logvar_e4[j]

                js = js_divergence(mu_A4, logvar_A4, mu_B4, logvar_B4)
                sim_matrix_e4[i, j] = 1 - js / np.log(2)

        if self.args.agg_norm == 'exp':
            sim_matrix_n = np.exp(self.args.norm_scale * sim_matrix_n)
            sim_matrix_e = np.exp(self.args.norm_scale * sim_matrix_e)
            sim_matrix_n1 = np.exp(self.args.norm_scale * sim_matrix_n1)
            sim_matrix_e1 = np.exp(self.args.norm_scale * sim_matrix_e1)
            sim_matrix_n2 = np.exp(self.args.norm_scale * sim_matrix_n2)
            sim_matrix_e2 = np.exp(self.args.norm_scale * sim_matrix_e2)
            sim_matrix_n3 = np.exp(self.args.norm_scale * sim_matrix_n3)
            sim_matrix_e3 = np.exp(self.args.norm_scale * sim_matrix_e3)
            sim_matrix_n4 = np.exp(self.args.norm_scale * sim_matrix_n4)
            sim_matrix_e4 = np.exp(self.args.norm_scale * sim_matrix_e4)

        row_sums = sim_matrix_n.sum(axis=1)
        sim_matrix_n = sim_matrix_n / row_sums[:, np.newaxis]

        row_sums = sim_matrix_e.sum(axis=1)
        sim_matrix_e = sim_matrix_e / row_sums[:, np.newaxis]

        row_sums = sim_matrix_n1.sum(axis=1)
        sim_matrix_n1 = sim_matrix_n1 / row_sums[:, np.newaxis]

        row_sums = sim_matrix_e1.sum(axis=1)
        sim_matrix_e1 = sim_matrix_e1 / row_sums[:, np.newaxis]

        row_sums = sim_matrix_n2.sum(axis=1)
        sim_matrix_n2 = sim_matrix_n2 / row_sums[:, np.newaxis]

        row_sums = sim_matrix_e2.sum(axis=1)
        sim_matrix_e2 = sim_matrix_e2 / row_sums[:, np.newaxis]

        row_sums = sim_matrix_n3.sum(axis=1)
        sim_matrix_n3 = sim_matrix_n3 / row_sums[:, np.newaxis]

        row_sums = sim_matrix_e3.sum(axis=1)
        sim_matrix_e3 = sim_matrix_e3 / row_sums[:, np.newaxis]

        row_sums = sim_matrix_n4.sum(axis=1)
        sim_matrix_n4 = sim_matrix_n4 / row_sums[:, np.newaxis]

        row_sums = sim_matrix_e4.sum(axis=1)
        sim_matrix_e4 = sim_matrix_e4 / row_sums[:, np.newaxis]

        st = time.time()
        ratio = (np.array(local_train_sizes) / np.sum(local_train_sizes)).tolist()

        self.set_weights(self.model, self.aggregate(local_weights, ratio))
        self.logger.print(f'global model has been updated ({time.time() - st:.2f}s)')

        st = time.time()
        for i, c_id in enumerate(updated):
            aggr_local_model_weights = self.aggregate_sp(local_weights, sim_matrix_n[i, :], sim_matrix_e[i, :],
                                                         sim_matrix_n1[i, :], sim_matrix_e1[i, :],
                                                         sim_matrix_n2[i, :], sim_matrix_e2[i, :],
                                                         sim_matrix_n3[i, :], sim_matrix_e3[i, :],
                                                         sim_matrix_n4[i, :], sim_matrix_e4[i, :])

            if f'personalized_{c_id}' in self.sd: del self.sd[f'personalized_{c_id}']
            self.sd[f'personalized_{c_id}'] = {'model': aggr_local_model_weights}

        self.update_lists.append(updated)
        self.sim_matrices_n.append(sim_matrix_n)
        self.sim_matrices_e.append(sim_matrix_e)
        self.sim_matrices_n1.append(sim_matrix_n1)
        self.sim_matrices_e1.append(sim_matrix_e1)
        self.sim_matrices_n2.append(sim_matrix_n2)
        self.sim_matrices_e2.append(sim_matrix_e2)
        self.sim_matrices_n3.append(sim_matrix_n3)
        self.sim_matrices_e3.append(sim_matrix_e3)
        self.sim_matrices_n4.append(sim_matrix_n4)
        self.sim_matrices_e4.append(sim_matrix_e4)

        self.logger.print(f'local model has been updated ({time.time() - st:.2f}s)')

    def aggregate_sp(self, local_weights, ratio_semantic, ratio_structure, ratio_semantic1, ratio_structure1, ratio_semantic2, ratio_structure2, ratio_semantic3, ratio_structure3, ratio_semantic4, ratio_structure4):
        aggr_theta = OrderedDict([(k, None) for k in local_weights[0].keys()])

        for name, params in aggr_theta.items():
            if name == 'clf.bias':
                ratio = 1 / len(local_weights)
                aggr_theta[name] = np.sum([theta[name] * ratio for j, theta in enumerate(local_weights)], 0)
            elif name in ['pca.weight', 'clf.weight']:
                aggregated_structure_model = np.sum(
                    [theta[name] * ratio_structure[j] for j, theta in enumerate(local_weights)], 0)
                aggregated_structure = np.split(aggregated_structure_model, indices_or_sections=10, axis=1)[1]

                aggregated_structure_model1 = np.sum(
                    [theta[name] * ratio_structure1[j] for j, theta in enumerate(local_weights)], 0)
                aggregated_structure1 = np.split(aggregated_structure_model1, indices_or_sections=10, axis=1)[3]

                aggregated_structure_model2 = np.sum(
                    [theta[name] * ratio_structure2[j] for j, theta in enumerate(local_weights)], 0)
                aggregated_structure2 = np.split(aggregated_structure_model2, indices_or_sections=10, axis=1)[5]

                aggregated_structure_model3 = np.sum(
                    [theta[name] * ratio_structure3[j] for j, theta in enumerate(local_weights)], 0)
                aggregated_structure3 = np.split(aggregated_structure_model3, indices_or_sections=10, axis=1)[7]

                aggregated_structure_model4 = np.sum(
                    [theta[name] * ratio_structure4[j] for j, theta in enumerate(local_weights)], 0)
                aggregated_structure4 = np.split(aggregated_structure_model4, indices_or_sections=10, axis=1)[9]

                aggregated_semantic_model = np.sum(
                    [theta[name] * ratio_semantic[j] for j, theta in enumerate(local_weights)], 0)
                aggregated_semantic = np.split(aggregated_semantic_model, indices_or_sections=10, axis=1)[0]

                aggregated_semantic_model1 = np.sum(
                    [theta[name] * ratio_semantic1[j] for j, theta in enumerate(local_weights)], 0)
                aggregated_semantic1 = np.split(aggregated_semantic_model1, indices_or_sections=10, axis=1)[2]

                aggregated_semantic_model2 = np.sum(
                    [theta[name] * ratio_semantic2[j] for j, theta in enumerate(local_weights)], 0)
                aggregated_semantic2 = np.split(aggregated_semantic_model2, indices_or_sections=10, axis=1)[4]

                aggregated_semantic_model3 = np.sum(
                    [theta[name] * ratio_semantic3[j] for j, theta in enumerate(local_weights)], 0)
                aggregated_semantic3 = np.split(aggregated_semantic_model3, indices_or_sections=10, axis=1)[6]

                aggregated_semantic_model4 = np.sum(
                    [theta[name] * ratio_semantic4[j] for j, theta in enumerate(local_weights)], 0)
                aggregated_semantic4 = np.split(aggregated_semantic_model4, indices_or_sections=10, axis=1)[8]

                aggregated_model_cat = np.concatenate(
                    (aggregated_semantic, aggregated_structure, aggregated_semantic1, aggregated_structure1, aggregated_semantic2, aggregated_structure2, aggregated_semantic3, aggregated_structure3, aggregated_semantic4, aggregated_structure4), axis=1)

                aggr_theta[name] = aggregated_model_cat
            elif name == 'pca.bias':
                aggregated_structure_model = np.sum(
                    [theta[name] * ratio_structure[j] for j, theta in enumerate(local_weights)], 0)
                aggregated_structure = np.split(aggregated_structure_model, indices_or_sections=10, axis=0)[1]

                aggregated_structure_model1 = np.sum(
                    [theta[name] * ratio_structure1[j] for j, theta in enumerate(local_weights)], 0)
                aggregated_structure1 = np.split(aggregated_structure_model1, indices_or_sections=10, axis=0)[3]

                aggregated_structure_model2 = np.sum(
                    [theta[name] * ratio_structure2[j] for j, theta in enumerate(local_weights)], 0)
                aggregated_structure2 = np.split(aggregated_structure_model2, indices_or_sections=10, axis=0)[5]

                aggregated_structure_model3 = np.sum(
                    [theta[name] * ratio_structure3[j] for j, theta in enumerate(local_weights)], 0)
                aggregated_structure3 = np.split(aggregated_structure_model3, indices_or_sections=10, axis=0)[7]

                aggregated_structure_model4 = np.sum(
                    [theta[name] * ratio_structure4[j] for j, theta in enumerate(local_weights)], 0)
                aggregated_structure4 = np.split(aggregated_structure_model4, indices_or_sections=10, axis=0)[9]

                aggregated_semantic_model = np.sum(
                    [theta[name] * ratio_semantic[j] for j, theta in enumerate(local_weights)], 0)
                aggregated_semantic = np.split(aggregated_semantic_model, indices_or_sections=10, axis=0)[0]

                aggregated_semantic_model1 = np.sum(
                    [theta[name] * ratio_semantic1[j] for j, theta in enumerate(local_weights)], 0)
                aggregated_semantic1 = np.split(aggregated_semantic_model1, indices_or_sections=10, axis=0)[2]

                aggregated_semantic_model2 = np.sum(
                    [theta[name] * ratio_semantic2[j] for j, theta in enumerate(local_weights)], 0)
                aggregated_semantic2 = np.split(aggregated_semantic_model2, indices_or_sections=10, axis=0)[4]

                aggregated_semantic_model3 = np.sum(
                    [theta[name] * ratio_semantic3[j] for j, theta in enumerate(local_weights)], 0)
                aggregated_semantic3 = np.split(aggregated_semantic_model3, indices_or_sections=10, axis=0)[6]

                aggregated_semantic_model4 = np.sum(
                    [theta[name] * ratio_semantic4[j] for j, theta in enumerate(local_weights)], 0)
                aggregated_semantic4 = np.split(aggregated_semantic_model4, indices_or_sections=10, axis=0)[8]

                aggregated_model_cat = np.concatenate((aggregated_semantic, aggregated_structure, aggregated_semantic1, aggregated_structure1, aggregated_semantic2, aggregated_structure2, aggregated_semantic3, aggregated_structure3, aggregated_semantic4, aggregated_structure4), axis=0)

                aggr_theta[name] = aggregated_model_cat

        return aggr_theta

    def set_weights(self, model, state_dict):
        set_state_dict(model, state_dict, self.gpu_id)

    def save_state(self):
        torch_save(self.args.checkpt_path, 'server_state.pt', {
            'model': get_state_dict(self.model),

            'sim_matrices_n': self.sim_matrices_n,
            'sim_matrices_e': self.sim_matrices_e,
            'sim_matrices_n1': self.sim_matrices_n1,
            'sim_matrices_e1': self.sim_matrices_e1,
            'sim_matrices_n2': self.sim_matrices_n2,
            'sim_matrices_e2': self.sim_matrices_e2,
            'sim_matrices_n3': self.sim_matrices_n3,
            'sim_matrices_e3': self.sim_matrices_e3,
            'sim_matrices_n4': self.sim_matrices_n4,
            'sim_matrices_e4': self.sim_matrices_e4,

            'update_lists': self.update_lists
        })
