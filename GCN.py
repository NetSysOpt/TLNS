import torch.nn.functional as F
import torch
import torch_geometric
#import torch_scatter
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from typing import Optional, Tuple, Union
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import (
    Adj,
    OptTensor,
    PairTensor,
    SparseTensor,
)
from torch_geometric.utils import (
    add_self_loops,
    remove_self_loops,
    softmax,
)
import pickle
import numpy as np
import time
class PreNormLayer(torch.nn.Module):
    def __init__(self, n_units, shift=True, scale=True, name=None):
        super().__init__()
        assert shift or scale
        self.register_buffer('shift', torch.zeros(n_units) if shift else None)
        self.register_buffer('scale', torch.ones(n_units) if scale else None)
        self.n_units = n_units
        self.waiting_updates = False
        self.received_updates = False

    def forward(self, input_):
        if self.waiting_updates:
            self.update_stats(input_)
            self.received_updates = True
            raise PreNormException

        if self.shift is not None:
            input_ = input_ + self.shift

        if self.scale is not None:
            input_ = input_ * self.scale

        return input_

    def start_updates(self):
        self.avg = 0
        self.var = 0
        self.m2 = 0
        self.count = 0
        self.waiting_updates = True
        self.received_updates = False

    def update_stats(self, input_):
        """
        Online mean and variance estimation. See: Chan et al. (1979) Updating
        Formulae and a Pairwise Algorithm for Computing Sample Variances.
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm
        """
        assert self.n_units == 1 or input_.shape[
            -1] == self.n_units, f"Expected input dimension of size {self.n_units}, got {input_.shape[-1]}."

        input_ = input_.reshape(-1, self.n_units)
        sample_avg = input_.mean(dim=0)
        sample_var = (input_ - sample_avg).pow(2).mean(dim=0)
        sample_count = np.prod(input_.size())/self.n_units

        delta = sample_avg - self.avg

        self.m2 = self.var * self.count + sample_var * sample_count + delta ** 2 * self.count * sample_count / (
            self.count + sample_count)

        self.count += sample_count
        self.avg += delta * sample_count / self.count
        self.var = self.m2 / self.count if self.count > 0 else 1

    def stop_updates(self):
        """
        Ends pre-training for that layer, and fixes the layers's parameters.
        """
        assert self.count > 0
        if self.shift is not None:
            self.shift = -self.avg

        if self.scale is not None:
            self.var[self.var < 1e-8] = 1
            self.scale = 1 / torch.sqrt(self.var)

        del self.avg, self.var, self.m2, self.count
        self.waiting_updates = False
        self.trainable = False


class Half_Conv(torch.nn.Module):
    def __init__(self, emb_size=64, cons_nfeats=6, edge_nfeats=1, var_nfeats=7):
        super().__init__()
        # emb_size = 64
        # cons_nfeats = 6
        # edge_nfeats = 1
        # var_nfeats = 7

        # CONSTRAINT EMBEDDING
        self.cons_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(cons_nfeats),
            torch.nn.Linear(cons_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )

        # EDGE EMBEDDING
        self.edge_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(edge_nfeats),
        )

        # VARIABLE EMBEDDING
        self.var_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(var_nfeats),
            torch.nn.Linear(var_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )

        self.conv_v_to_c = BipartiteGraphConvolution(emb_size=emb_size)
        self.conv_c_to_v = BipartiteGraphConvolution(emb_size=emb_size)

    def forward(
        self, constraint_features, edge_indices, edge_features, variable_features, v_embedded=False, c_embedded=False
    ):
        reversed_edge_indices = torch.stack([edge_indices[1], edge_indices[0]], dim=0)

        # First step: linear embedding layers to a common dimension (64)
        if not c_embedded:
            constraint_features = self.cons_embedding(constraint_features)
        edge_features = self.edge_embedding(edge_features)
        if not v_embedded:
            variable_features = self.var_embedding(variable_features)

        # Two half convolutions
        constraint_features = self.conv_v_to_c(
            variable_features, reversed_edge_indices, edge_features, constraint_features
        )
        variable_features = self.conv_c_to_v(
            constraint_features, edge_indices, edge_features, variable_features
        )

        return variable_features, constraint_features


class GNNPolicy_raw(torch.nn.Module):
    def __init__(self, emb_size=64, cons_nfeats=6, edge_nfeats=1, var_nfeats=7):
        super().__init__()

        # CONSTRAINT EMBEDDING
        self.cons_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(cons_nfeats),
            torch.nn.Linear(cons_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )

        # EDGE EMBEDDING
        self.edge_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(edge_nfeats),
        )

        # VARIABLE EMBEDDING
        self.var_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(var_nfeats),
            torch.nn.Linear(var_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )

        self.conv_v_to_c = BipartiteGraphConvolution(emb_size=emb_size)
        self.conv_c_to_v = BipartiteGraphConvolution(emb_size=emb_size)

        self.conv_v_to_c2 = BipartiteGraphConvolution(emb_size=emb_size)
        self.conv_c_to_v2 = BipartiteGraphConvolution(emb_size=emb_size)


    def forward(
        self, constraint_features, edge_indices, edge_features, variable_features, v_embedded=False, c_embedded=False
    ):
        reversed_edge_indices = torch.stack([edge_indices[1], edge_indices[0]], dim=0)

        # First step: linear embedding layers to a common dimension (64)
        if not c_embedded:
            constraint_features = self.cons_embedding(constraint_features)
        edge_features = self.edge_embedding(edge_features)
        if not v_embedded:
            variable_features = self.var_embedding(variable_features)

        # Two half convolutions
        constraint_features = self.conv_v_to_c(
            variable_features, reversed_edge_indices, edge_features, constraint_features
        )
        variable_features = self.conv_c_to_v(
            constraint_features, edge_indices, edge_features, variable_features
        )

        constraint_features = self.conv_v_to_c2(
            variable_features, reversed_edge_indices, edge_features, constraint_features
        )
        variable_features = self.conv_c_to_v2(
            constraint_features, edge_indices, edge_features, variable_features
        )
        return variable_features, constraint_features



class GNNPolicy(torch.nn.Module):
    def __init__(self, emb_size=32, cons_nfeats=6, edge_nfeats=1, var_nfeats=7):
        super().__init__()


        # CONSTRAINT EMBEDDING
        self.cons_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(cons_nfeats),
            torch.nn.Linear(cons_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )

        # EDGE EMBEDDING
        self.edge_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(edge_nfeats),
        )

        # VARIABLE EMBEDDING
        self.var_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(var_nfeats),
            torch.nn.Linear(var_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )

        self.conv_v_to_c = BipartiteGraphConvolution(emb_size=emb_size)
        self.conv_c_to_v = BipartiteGraphConvolution(emb_size=emb_size)

        self.conv_v_to_c2 = BipartiteGraphConvolution(emb_size=emb_size)
        self.conv_c_to_v2 = BipartiteGraphConvolution(emb_size=emb_size)



        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, 1, bias=False),
        )

    def forward(
        self, constraint_features, edge_indices, edge_features, variable_features, num_cons=None
    ):
        reversed_edge_indices = torch.stack([edge_indices[1], edge_indices[0]], dim=0)

        # First step: linear embedding layers to a common dimension (64)
        constraint_features = self.cons_embedding(constraint_features)
        edge_features = self.edge_embedding(edge_features)
        variable_features = self.var_embedding(variable_features)

        # Two half convolutions
        constraint_features = self.conv_v_to_c(
            variable_features, reversed_edge_indices, edge_features, constraint_features
        )
        variable_features = self.conv_c_to_v(
            constraint_features, edge_indices, edge_features, variable_features
        )

        constraint_features = self.conv_v_to_c2(
            variable_features, reversed_edge_indices, edge_features, constraint_features
        )
        variable_features = self.conv_c_to_v2(
            constraint_features, edge_indices, edge_features, variable_features
        )

        # A final MLP on the variable features
        output = self.output_module(variable_features).squeeze(-1)

        return output



class BaseModel(torch.nn.Module):
    """
    Our base model class, which implements pre-training methods.
    """

    def pre_train_init(self):
        for module in self.modules():
            if isinstance(module, PreNormLayer):
                module.start_updates()

    def pre_train_next(self):
        for module in self.modules():
            if isinstance(module, PreNormLayer) and module.waiting_updates and module.received_updates:
                module.stop_updates()
                return module
        return None

    def pre_train(self, *args, **kwargs):
        try:
            with torch.no_grad():
                self.forward(*args, **kwargs)
            return False
        except PreNormException:
            return True
class GATPolicy(BaseModel):
    def __init__(self, emb_size=32, num_heads_per_layer=2, bias=True, dropout=0.1, var_nfeats = 7,cons_nfeats = 6,edge_nfeats = 1):
        super().__init__()
        self.emb_size = emb_size

        # CONSTRAINT EMBEDDING GOOD CANDIDATE TO REDUCE SIZE
        self.cons_embedding = torch.nn.Sequential(
            PreNormLayer(cons_nfeats),
            torch.nn.Linear(cons_nfeats, self.emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.emb_size, self.emb_size),
            torch.nn.ReLU(),
        )
        # EDGE EMBEDDING
        self.edge_embedding = torch.nn.Sequential(
            PreNormLayer(edge_nfeats),
            # embedding to hidden dims.
            torch.nn.Linear(edge_nfeats, self.emb_size),
            torch.nn.ReLU(),
        )

        # VARIABLE EMBEDDING   GOOD CANDIDATE TO REDUCE SIZE
        self.var_embedding = torch.nn.Sequential(
            PreNormLayer(var_nfeats),
            torch.nn.Linear(var_nfeats, self.emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.emb_size, self.emb_size),
            torch.nn.ReLU(),
        )
        self.gat_v_to_c = GATv2Conv(in_channels=(self.emb_size, self.emb_size), out_channels=self.emb_size,
                                    heads=num_heads_per_layer, edge_dim=self.emb_size,
                                    add_self_loops=False,
                                    bias=bias, dropout=dropout)

        self.gat_c_to_v = GATv2Conv(in_channels=(self.emb_size * num_heads_per_layer, self.emb_size),
                                    out_channels=self.emb_size,
                                    heads=num_heads_per_layer, edge_dim=self.emb_size,
                                    add_self_loops=False,
                                    bias=bias, dropout=dropout)

        self.gat_v_to_c2 = GATv2Conv(in_channels=(self.emb_size, self.emb_size), out_channels=self.emb_size,
                                    heads=num_heads_per_layer, edge_dim=self.emb_size,
                                    add_self_loops=False,
                                    bias=bias, dropout=dropout)

        self.gat_c_to_v2 = GATv2Conv(in_channels=(self.emb_size * num_heads_per_layer, self.emb_size),
                                    out_channels=self.emb_size,
                                    heads=num_heads_per_layer, edge_dim=self.emb_size,
                                    add_self_loops=False,
                                    bias=bias, dropout=dropout)
        self.c1 = torch.nn.Sequential(torch.nn.Linear(self.emb_size*num_heads_per_layer,self.emb_size))
        self.v1 = torch.nn.Sequential(torch.nn.Linear(self.emb_size * num_heads_per_layer, self.emb_size))
        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(
                self.emb_size * num_heads_per_layer, self.emb_size * num_heads_per_layer),
            torch.nn.ReLU(),
            torch.nn.Linear(
                self.emb_size * num_heads_per_layer, 1, bias=False),
        )

    def forward(self, constraint_features, edge_indices, edge_features, variable_features, num_cons=None):
        variable_features = torch.nan_to_num(variable_features, 0.0)
        constraint_features = self.cons_embedding(constraint_features)
        edge_features = self.edge_embedding(edge_features)
        variable_features = self.var_embedding(variable_features)
        reversed_edge_indices = torch.stack(
            [edge_indices[1], edge_indices[0]], dim=0)

        constraint_features = self.gat_v_to_c((variable_features, constraint_features), reversed_edge_indices,
                                              edge_features)  # , size=(variable_features.shape[0],constraint_features.shape[0]))

        variable_features = self.gat_c_to_v((constraint_features, variable_features), edge_indices,
                                            edge_features)  # , size=(constraint_features.shape[0], variable_features.shape[0]))

        constraint_features = self.c1(constraint_features)
        variable_features = self.v1(variable_features)

        constraint_features = self.gat_v_to_c2((variable_features, constraint_features), reversed_edge_indices,
                                              edge_features)  # , size=(variable_features.shape[0],constraint_features.shape[0]))

        variable_features = self.gat_c_to_v2((constraint_features, variable_features), edge_indices,
                                            edge_features)  # , size=(constraint_features.shape[0], variable_features.shape[0]))
        output = self.output_module(variable_features).squeeze(-1)


        return output

    pass
class BipartiteGraphConvolution(torch_geometric.nn.MessagePassing):
    """
    The bipartite graph convolution is already provided by pytorch geometric and we merely need
    to provide the exact form of the messages being passed.
    """

    def __init__(self, emb_size=64):
        super().__init__("add")
        #emb_size = 64

        self.feature_module_left = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size)
        )
        self.feature_module_edge = torch.nn.Sequential(
            torch.nn.Linear(1, emb_size, bias=False)
        )
        self.feature_module_right = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size, bias=False)
        )
        self.feature_module_final = torch.nn.Sequential(
            torch.nn.LayerNorm(emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
        )

        self.post_conv_module = torch.nn.Sequential(torch.nn.LayerNorm(emb_size))

        # output_layers
        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(2 * emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
        )

    def forward(self, left_features, edge_indices, edge_features, right_features):
        """
        This method sends the messages, computed in the message method.
        """


        output = self.propagate(
            edge_indices,
            size=(left_features.shape[0], right_features.shape[0]),
            node_features=(left_features, right_features),
            edge_features=edge_features,
        )
        b=torch.cat([self.post_conv_module(output), right_features], dim=-1)
        a=self.output_module(
            torch.cat([self.post_conv_module(output), right_features], dim=-1)
        )

        return self.output_module(
            torch.cat([self.post_conv_module(output), right_features], dim=-1)
        )


    def message(self, node_features_i, node_features_j, edge_features):
        #node_features_i,the node to be aggregated
        #node_features_j,the neighbors of the node i

        # print("node_features_i:",node_features_i.shape)
        # print("node_features_j",node_features_j.shape)
        # print("edge_features:",edge_features.shape)

        output = self.feature_module_final(
            self.feature_module_left(node_features_i)
            + self.feature_module_edge(edge_features)
            + self.feature_module_right(node_features_j)
        )

        return output

class GraphDataset(torch_geometric.data.Dataset):
    """
    sol is of the form [[[],],...] [[A,improvement],...] where A is a 0/1 list, improvement is a scalar
    BG is of the form BG = [A, state_vnode_represent(v_nodes, cur_sol_val), c_nodes]
    """
    def __init__(self, sample_files):
        super().__init__(root=None, transform=None, pre_transform=None)
        self.sample_files = sample_files

    def len(self):
        return len(self.sample_files)

    def process_sample(self, filepath):
        BGFilepath, solFilePath = filepath
        with open(BGFilepath, "rb") as f:
            bgData = pickle.load(f)
        with open(solFilePath, "rb") as f:
            solData = pickle.load(f)
        return bgData, solData

    def get(self, index):
        """
        This method loads a node bipartite graph observation as saved on the disk during data collection.
        """

        # nbp, sols, objs, varInds, varNames = self.process_sample(self.sample_files[index])
        BG, sols = self.process_sample(self.sample_files[index])

        A, v_nodes, c_nodes = BG

        constraint_features = c_nodes
        edge_indices = A._indices()

        variable_features = v_nodes
        edge_features = A._values().unsqueeze(1)


        constraint_features[torch.isnan(constraint_features)] = 1

        graph = BipartiteNodeData(
            torch.FloatTensor(constraint_features).to('cuda:0'),
            torch.LongTensor(edge_indices).to('cuda:0'),
            torch.FloatTensor(edge_features).to('cuda:0'),
            torch.FloatTensor(variable_features).to('cuda:0'),
        )

        # We must tell pytorch geometric how many nodes there are, for indexing purposes
        graph.num_nodes = constraint_features.shape[0] + variable_features.shape[0]
        graph.sols = sols  # sols is like [[[0,0,0],1],...], slow to move to GPU
        graph.ntvars = variable_features.shape[0]
        graph.ncons = constraint_features.shape[0]
        return graph



class BipartiteNodeData(torch_geometric.data.Data):
    """
    This class encode a node bipartite graph observation as returned by the `ecole.observation.NodeBipartite`
    observation function in a format understood by the pytorch geometric data handlers.
    """

    def __init__(
            self,
            constraint_features,
            edge_indices,
            edge_features,
            variable_features,

    ):
        super().__init__()
        self.constraint_features = constraint_features
        self.edge_index = edge_indices
        self.edge_attr = edge_features
        self.variable_features = variable_features



    def __inc__(self, key, value, store, *args, **kwargs):
        """
        We overload the pytorch geometric method that tells how to increment indices when concatenating graphs
        for those entries (edge index, candidates) for which this is not obvious.
        """
        if key == "edge_index":
            return torch.tensor(
                [[self.constraint_features.size(0)], [self.variable_features.size(0)]]
            )
        elif key == "candidates":
            return self.variable_features.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)







def full_attention_conv(qs, ks, vs):
    # normalize input
    qs = qs / torch.norm(qs, p=2)  # [N, H, M]
    ks = ks / torch.norm(ks, p=2)  # [L, H, M]
    N = qs.shape[-2]

    # numerator
    #kvs = torch.einsum("lhm,lhd->hmd", ks, vs) # K^T V
    kvs = torch.einsum("bhnm,bhnd->bhmd", ks, vs)  # [b,h,m,d] d=m

    #attention_num = torch.einsum("nhm,hmd->nhd", qs, kvs)  # [N, H, D]
    attention_num = torch.einsum("bhnm,bhmd->bhnd", qs, kvs)
    attention_num += N * vs

    # denominator
    all_ones = torch.ones([ks.shape[-2]]).to(ks.device)
    ks_sum = torch.einsum("bhnm,n->bhm", ks, all_ones)
    attention_normalizer = torch.einsum("bhnm,bhm->bhn", qs, ks_sum)  # [N, H]

    # attentive aggregated results
    attention_normalizer = torch.unsqueeze(
        attention_normalizer, len(attention_normalizer.shape))  # [N, H, 1]
    attention_normalizer += torch.ones_like(attention_normalizer) * N
    attn_output = attention_num / attention_normalizer  # [N, H, D]

    return attn_output




class TransConvLayer(torch.nn.Module):
    '''
    transformer with fast attention
    '''

    def __init__(self, in_channels,
                 out_channels,
                 num_heads,
                 use_weight=True):
        super().__init__()
        self.Wk = torch.nn.Linear(in_channels, out_channels * num_heads)
        self.Wq = torch.nn.Linear(in_channels, out_channels * num_heads)
        if use_weight:
            self.Wv = torch.nn.Linear(in_channels, out_channels * num_heads)

        self.out_channels = out_channels
        self.num_heads = num_heads
        self.use_weight = use_weight

    def reset_parameters(self):
        self.Wk.reset_parameters()
        self.Wq.reset_parameters()
        if self.use_weight:
            self.Wv.reset_parameters()

    def forward(self, query_input, source_input, query_size, mask=None, edge_weights=None):
        # feature transformation
        # print(query_input.shape)
        # [batch_size, num_nodes, embed_dim * num_heads]
        batch_size = query_input.shape[0]


        query = self.Wq(query_input).reshape(batch_size, -1, self.num_heads,
                                             self.out_channels).transpose(1, 2)

        key = self.Wk(source_input).reshape(batch_size, -1,
                                            self.num_heads, self.out_channels).transpose(1, 2)

        value = self.Wv(source_input).reshape(batch_size, -1,
                                              self.num_heads, self.out_channels).transpose(1, 2)

        attention_output = full_attention_conv(
            query, key, value)  # [N, H, D]

        final_output = attention_output
        # print(final_output.shape) [batch_size, num_heads, query_size, embed_dim]
        final_output = final_output.mean(dim=1)

        return final_output


class TransConv(torch.nn.Module):
    def __init__(self, in_channels_v, in_channels_c, hidden_channels, num_layers=1, num_heads=1,
                 alpha=0.8, dropout=0.1, use_bn=True, use_residual=True, use_weight=True, use_act=False):
        super().__init__()
        self.v_embed = torch.nn.Linear(in_channels_v, hidden_channels)
        self.c_embed = torch.nn.Linear(in_channels_c, hidden_channels)
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.LayerNorm(hidden_channels))
        self.fcs = torch.nn.ModuleList()
        for i in range(num_layers):
            self.fcs.append(
                torch.nn.Sequential(torch.nn.Linear(hidden_channels, 2 * hidden_channels, bias=False), torch.nn.ReLU(),
                                    torch.nn.Linear(2 * hidden_channels, hidden_channels, bias=False)))
            self.convs.append(
                TransConvLayer(hidden_channels, hidden_channels, num_heads=num_heads, use_weight=use_weight))
            self.bns.append(torch.nn.LayerNorm(hidden_channels))
            self.bns.append(torch.nn.LayerNorm(hidden_channels))
        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn
        self.residual = use_residual
        self.alpha = alpha
        self.use_act = use_act

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        for fc in self.fcs:
            fc.reset_parameters()

    def forward(self, constraint_features, edge_indices, edge_features, variable_features, mask, embedded=False):
        layer_ = []
        layer2_ = []
        num_vars = variable_features.shape[1]

        # c/v_features: [batch_size, num_nodes, num_feat]
        # input MLP layer
        if not embedded:
            constraint_features = self.c_embed(constraint_features)
            variable_features = self.v_embed(variable_features)
        x = torch.cat((variable_features, constraint_features), dim=1)  # [batch_size, num_all_nodes, embed_dim]
        if not embedded:
            if self.use_bn:
                x = self.bns[0](x)
            x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        layer_.append(x)

        for i, conv in enumerate(self.convs):
            # graph convolution with full attention aggregation
            x = conv(x, x, num_vars)

            if self.residual:
                x = self.alpha * x + (1 - self.alpha) * layer_[i][:, :x.shape[1], :]
            if self.use_bn:
                x = self.bns[2 * i + 1](x)
            if self.use_act:
                x = self.activation(x)

            layer2_.append(x)
            x = self.fcs[i](x)
            if self.residual:
                x = self.alpha * x + (1 - self.alpha) * layer2_[i]
            x = self.bns[2 * i + 2](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            layer_.append(x)

        return x

    def get_attentions(self, x):
        layer_, attentions = [], []
        x = self.fcs[0](x)
        if self.use_bn:
            x = self.bns[0](x)
        x = self.activation(x)
        layer_.append(x)
        for i, conv in enumerate(self.convs):
            x, attn = conv(x, x, output_attn=True)
            attentions.append(attn)
            if self.residual:
                x = self.alpha * x + (1 - self.alpha) * layer_[i]
            if self.use_bn:
                x = self.bns[i + 1](x)
            layer_.append(x)
        return torch.stack(attentions, dim=0)  # [layer num, N, N]


class SGT(torch.nn.Module):
    def __init__(self, in_channels_v=7, in_channels_c=6, hidden_channels=32, out_channels=1, num_layers=1, num_heads=1,
                 alpha=0.8, dropout=0.5, use_bn=True, use_residual=True, use_weight=True, use_graph=True, use_act=True,
                 graph_weight=0, gnn=None, batch_size=1):
        super().__init__()
        self.trans_conv = TransConv(in_channels_v, in_channels_c, hidden_channels, num_layers, num_heads, alpha, dropout, use_bn,
                                    use_residual, use_weight)
        self.gnn = GNNPolicy_raw(emb_size=hidden_channels)
        self.use_graph = use_graph
        self.graph_weight = graph_weight
        self.use_act = use_act
        self.batch_size = batch_size
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, out_channels, bias=False),
        )

        self.params1 = list(self.trans_conv.parameters())
        self.params2 = list(self.gnn.parameters())
        self.params2.extend(list(self.fc.parameters()))

    def forward(self, constraint_features, edge_indices, edge_features, variable_features, num_cons=None):
        # num_cons: [batch_size] , number of constraints for each MILP within the batch
        # constraint_features: [sum(num_cons), cons_nfeats]
        # variable_features: [batch_size * num_vars, vars_nfeats]

        if num_cons is None:
            num_cons = torch.tensor([constraint_features.shape[0]])
        b = num_cons.shape[0]
        v0 = variable_features.reshape(b, -1, variable_features.shape[-1])
        #c0 = constraint_features.reshape(b, -1, constraint_features.shape[-1])
        max_num_cons = torch.max(num_cons).item()
        c_splits = torch.split(constraint_features, num_cons.tolist(), dim=0)
        c = []
        for split in c_splits:
            split = torch.nn.functional.pad(split, (0, 0, 0, max_num_cons-split.shape[0]))
            c.append(split)
        c0 = torch.stack(c)   # [batch_size, max_num_cons, cons_nfeats]
        mask = torch.zeros([b, max_num_cons+v0.shape[1]], dtype=torch.bool)
        indices = torch.arange(mask.shape[1]).unsqueeze(0)
        a = indices >= (num_cons + v0.shape[1]).unsqueeze(1)
        mask[a] = True
        mask = mask.to('cuda:0')
        embedding = self.trans_conv(c0, edge_indices, edge_features, v0, mask)

        v, c = embedding[:, :v0.shape[1], :], embedding[:, v0.shape[1]:v0.shape[1]+c0.shape[1], :]
        v1 = v.reshape(-1, v.shape[-1])
        c1 = c.reshape(-1, c.shape[-1])

        v,_ = self.gnn(c1, edge_indices, edge_features, v1, v_embedded=True, c_embedded=True)

        output = self.fc(v).squeeze(-1)

        return output

    def get_attentions(self, x):
        attns = self.trans_conv.get_attentions(x)  # [layer num, N, N]

        return attns

    def reset_parameters(self):
        self.trans_conv.reset_parameters()
        if self.use_graph:
            self.gnn.reset_parameters()