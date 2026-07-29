from torch import nn
from torch_geometric.nn import Linear, TransformerConv
import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import numpy as np
import random
import torch_scatter
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.utils import unbatch_edge_index
from torch_geometric.utils import unbatch

class Attention(nn.Module):
    """
    Multi-head attention
    Input:
      q: [batch_size, n_node, hidden_dim]
      k, v: q if None
    Output:
      att: [n_node, hidden_dim]
    """
    def __init__(self,
                 q_hidden_dim,
                 k_dim,
                 v_dim,
                 n_head,
                 k_hidden_dim=None,
                 v_hidden_dim=None):
        super().__init__()
        self.q_hidden_dim = q_hidden_dim
        self.k_hidden_dim = k_hidden_dim if k_hidden_dim else q_hidden_dim
        self.v_hidden_dim = v_hidden_dim if v_hidden_dim else q_hidden_dim
        self.k_dim = k_dim
        self.v_dim = v_dim
        self.n_head = n_head

        self.proj_q = nn.Linear(q_hidden_dim, k_dim * n_head, bias=False)
        self.proj_k = nn.Linear(self.k_hidden_dim, k_dim * n_head, bias=False)
        self.proj_v = nn.Linear(self.v_hidden_dim, v_dim * n_head, bias=False)
        self.proj_output = nn.Linear(v_dim * n_head,
                                     self.v_hidden_dim,
                                     bias=False)

    def forward(self, q, k=None, v=None, mask=None):
        if k is None:
            k = q
        if v is None:
            v = k
        if v is None:
            v = q

        bsz, n_node, hidden_dim = q.size()

        qs = torch.stack(torch.chunk(self.proj_q(q), self.n_head, dim=-1),
                         dim=1)  # [batch_size, n_head, n_node, k_dim]
        ks = torch.stack(torch.chunk(self.proj_k(k), self.n_head, dim=-1),
                         dim=1)  # [batch_size, n_head, n_node, k_dim]
        vs = torch.stack(torch.chunk(self.proj_v(v), self.n_head, dim=-1),
                         dim=1)  # [batch_size, n_head, n_node, v_dim]

        normalizer = self.k_dim**0.5
        u = torch.matmul(qs, ks.transpose(2, 3)) / normalizer
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)
            u = u.masked_fill(mask, float('-inf'))
        att = torch.matmul(torch.softmax(u, dim=-1), vs)
        att = att.transpose(1, 2).reshape(bsz, n_node,
                                          self.v_dim * self.n_head)
        att = self.proj_output(att)
        return att


class Encoder(torch.nn.Module):
    def __init__(self, input_dim, edge_input_dim, hidden_dim, num_layers, heads=4, beta=True, dropout=0):
        """
        Params:
        - input_dim: The dimension of input features for each node.
        - hidden_dim: The size of the hidden layers.
        - output_dim: The dimension of the output features (often equal to the
            number of classes in a classification task).
        - num_layers: The number of layer blocks in the model.
        - dropout: The dropout rate for regularization. It is used to prevent
            overfitting, helping the learning process remains generalized.
        - beta: A boolean parameter indicating whether to use a gated residual
            connection (based on equations 5 and 6 from the UniMP paper). The
            gated residual connection (controlled by the beta parameter) helps
            preventing overfitting by allowing the model to balance between new
            and existing node features across layers.
        - heads: The number of heads in the multi-head attention mechanism.
        """
        super().__init__()
        # Initial linear projection of the features
        self.project_node_features = Linear(input_dim, hidden_dim, bias=False)
        self.project_edge_features = Linear(edge_input_dim, hidden_dim, bias=False)

        # The list of transformer conv layers.
        self.num_layers = num_layers
        conv_layers = [TransformerConv(hidden_dim, hidden_dim, heads=heads, beta=beta, concat=False, edge_dim=hidden_dim) for _ in
                        range(num_layers)]

        self.convs = torch.nn.ModuleList(conv_layers)

        # The list of layerNorm for each layer block.
        norm_layers = [torch.nn.BatchNorm1d(hidden_dim) for _ in range(num_layers - 1)]
        self.norms = torch.nn.ModuleList(norm_layers)

        # Probability of an element getting zeroed.
        self.dropout = dropout

    def reset_parameters(self):
        """
        Resets the parameters of the convolutional and normalization layers,
        ensuring they are re-initialized when needed.
        """
        for conv in self.convs:
            conv.reset_parameters()
        for norm in self.norms:
            norm.reset_parameters()

    def forward(self, x, edge_index, edge_att):
        """
        The input features are passed sequentially through the transformer
        convolutional layers. After each convolutional layer (except the last),
        the following operations are applied:
        - Layer normalization (`LayerNorm`).
        - ReLU activation function.
        - Dropout for regularization.
        The final layer is processed without layer normalization and ReLU
        to average the multi-head results for the expected output.

        Params:
        - x: node features x
        - edge_index: edge indices.

        """
        x = self.project_node_features(x)
        edge_att = self.project_edge_features(edge_att)

        for i in range(self.num_layers):
            # Construct the network as shown in the model architecture.
            x = self.convs[i](x, edge_index=edge_index, edge_attr=edge_att)

            if i == self.num_layers - 1:
                # No normalization or relu or dropout at the final layer.
                break

            x = self.norms[i](x)
            x = F.relu(x)
            # By setting training to self.training, we will only apply dropout
            # during model training.
            x = F.dropout(x, p=self.dropout, training=self.training)

        return x, edge_index, edge_att


class Decoder_LSTM(nn.Module):
    """Calculates the next state given the previous state and input embeddings."""

    def __init__(self, hidden_size, n_head, num_layers=1, dropout=0.1):
        super(Decoder_LSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, bias=False,
                            batch_first=True, bidirectional=False, dropout=dropout if num_layers > 1 else 0)

        self.encoder_attn = Attention(hidden_size*2,
                                      int(hidden_size/n_head),
                                      int(hidden_size/n_head),
                                      n_head,
                                      k_hidden_dim=hidden_size,
                                      v_hidden_dim=hidden_size)

        self.proj_k = nn.Linear(hidden_size, hidden_size, bias=False)
        if torch.cuda.is_available():
            self.lstm = self.lstm.cuda()
            self.encoder_attn = self.encoder_attn.cuda()
            self.proj_k = self.proj_k.cuda()

    def forward(self,x, edge_index, data, mu=0.5, parent_mask=None, rollout=False):
        batch_size = data.batch.max().item() + 1
        normalizer = self.hidden_size ** 0.5

        # Initialize variables
        handle_sets = [[] for _ in range(batch_size)]
        log_probs = torch.zeros(batch_size, device=x.device)
        mask = torch.zeros(x.size(0), dtype=torch.bool, device=x.device)

        # Initial selection: node with maximum self-attention
        for i in range(batch_size):
            graph_mask = (data.batch == i)
            edge_mask = graph_mask[edge_index[0]] & graph_mask[edge_index[1]]

            graph_edges = edge_index[:, edge_mask]
            # Identify the depot node (smallest index in the graph)
            graph_nodes = graph_mask.nonzero(as_tuple=False).view(-1)
            depot = graph_nodes[0]  # Assumes the depot is the smallest index
            if parent_mask is not None:
                parent_mask[depot] = True
                graph_mask = (graph_mask&parent_mask)
                parent_set = graph_mask.nonzero(as_tuple=False).view(-1)
            graph_x = x[graph_mask]
            last_hh = graph_x.mean(dim=0, keepdim=True)
            dynamic_input = graph_x.mean(dim=0, keepdim=True).view(1, 1, -1).clone()
            graph_c = last_hh.view(1, 1, -1)
            last_hh = (graph_c, graph_c)

            search = True
            hs = 0
            while search:
                # Find edges crossing the handle
                p_mutation = random.random()
                if hs==0:
                    if parent_mask is not None and p_mutation > mu:
                        if len(parent_set)>=2:
                            neighbor_nodes = parent_set
                        else:
                            neighbor_nodes = graph_nodes
                    else:
                        neighbor_nodes = graph_nodes
                else:
                    if parent_mask is not None and p_mutation > mu:
                        handle_nodes = torch.tensor(handle_sets[i], device=x.device)
                        edge_cross_handle_mask = (
                                    (graph_edges[0].unsqueeze(1) == handle_nodes).any(dim=1) & ~mask[graph_edges[1]])
                        crossing_edges = graph_edges[:, edge_cross_handle_mask]
                        neighbor_nodes = torch.unique(crossing_edges[1])
                        neighbor_nodes = neighbor_nodes[torch.isin(neighbor_nodes, parent_set)]
                    else:
                        handle_nodes = torch.tensor(handle_sets[i], device=x.device)
                        edge_cross_handle_mask = (
                                    (graph_edges[0].unsqueeze(1) == handle_nodes).any(dim=1) & ~mask[graph_edges[1]])
                        crossing_edges = graph_edges[:, edge_cross_handle_mask]
                        neighbor_nodes = torch.unique(crossing_edges[1])

                rnn_out, last_hh = self.lstm(dynamic_input.view(1, 1, -1), last_hh)
                rnn_out = rnn_out.squeeze(1)

                # compute attention
                h_c = torch.cat([rnn_out.view(1,1,-1), last_hh[0]], -1)
                neighbor_hidden = x[neighbor_nodes].view(1, -1, self.hidden_size)
                q = self.encoder_attn(h_c, neighbor_hidden, neighbor_hidden)

                u = torch.tanh(q.bmm(neighbor_hidden.transpose(-2, -1)) / normalizer) * 5

                # Calculate attention scores for neighboring nodes
                node_scores = torch.zeros(neighbor_nodes.size(0), device=x.device) + u.view(neighbor_nodes.size(0))

                if neighbor_nodes.size(0) == 0:
                    search = False
                    # Translate indices by subtracting the depot node index
                    handle_sets_np = np.array(handle_sets[i]) - depot.item()
                    handle_sets[i] = handle_sets_np.tolist()
                    continue
                if rollout:
                    next_node = neighbor_nodes[node_scores.argmax()]
                else:
                    dist = Categorical(logits=node_scores)
                    next_node_idx = dist.sample()
                    next_node = neighbor_nodes[next_node_idx]
                    log_probs[i] += dist.log_prob(next_node_idx)

                if next_node == depot:
                    if hs==0:
                        search=True
                        next_node = graph_nodes[1]
                        dist = Categorical(logits=node_scores)
                        next_node_idx = torch.tensor(1, device=x.device)
                        log_probs[i] += dist.log_prob(next_node_idx)
                    else:
                        search = False
                        # Translate indices by subtracting the depot node index
                        handle_sets_np = np.array(handle_sets[i]) - depot.item()
                        handle_sets[i] = handle_sets_np.tolist()
                        continue
                handle_sets[i].append(next_node.item())
                mask[next_node] = True
                hs += 1
                dynamic_input = ((hs - 1)/hs)*dynamic_input + (1/hs)*x[next_node]
        return handle_sets, log_probs, mask


    def forward_batched(self,x, data, rollout=False):

        batch_size = data.batch.max().item() + 1
        hidden_dim = x.size(1)
        normalizer = hidden_dim ** 0.5

        # 1. Setup
        log_probs = torch.zeros(batch_size, device=x.device)
        active = torch.ones(batch_size, dtype=torch.bool, device=x.device)
        handle_size = torch.zeros(batch_size, dtype=torch.float, device=x.device)

        # 4. Get depot nodes (smallest index per graph)
        _, node_counts = torch.unique_consecutive(data.batch, return_counts=True)
        ids = torch.cumsum(node_counts, dim=0)
        num_graphs = len(ids)
        depots = torch.zeros(num_graphs, dtype=torch.long, device=x.device)
        depots[1:] = ids[0:num_graphs-1]
        # Get list of node indices per graph
        graph_nodes_list = [torch.arange(node_counts[i], device=x.device) for i in range(batch_size)]
        # Pad to create [batch_size, max_nodes]
        batched_graph_nodes = pad_sequence(graph_nodes_list, batch_first=True, padding_value=-1)

        graph_mask = (batched_graph_nodes != -1)
        handle_mask = torch.zeros_like(graph_mask, device=x.device)  # same shape: [batch_size, max_nodes]
        # Create a mask: True where valid node
        ## swapped edges didn't work---edges batched are DIRECTIONAL
        graph_edge_list = unbatch_edge_index(data.edge_index, data.batch)
        graph_edge_list = [graph_edge_list[i].T for i in range(batch_size)]
        # Pad edge lists: [batch_size, 2, max_edges]
        batched_graph_edges = pad_sequence(graph_edge_list, batch_first=True, padding_value=-1).transpose(1, 2)
        batched_edge_mask = (batched_graph_edges[:, 0, :] != -1)  # shape [batch_size, max_edges]
        batch_indices = torch.arange(batch_size, device=x.device, dtype=torch.long).unsqueeze(1).expand(-1, batched_edge_mask.shape[-1])

        in_edges = torch.zeros_like(batched_edge_mask, dtype=torch.bool, device=batched_edge_mask.device)
        out_edges = torch.zeros_like(batched_edge_mask, dtype=torch.bool, device=batched_edge_mask.device)

        # original indices of the large batched graph
        graph_nodes_list_o = unbatch(torch.arange(data.num_nodes, device=x.device), data.batch)
        batched_graph_nodes_o = pad_sequence(graph_nodes_list_o, batch_first=True, padding_value=-1)


        # Gather x using batched_graph_nodes (which contains node indices)
        # pad x with dummy row for -1 lookups
        x_padded = torch.cat([x, torch.zeros(1, hidden_dim, device=x.device)], dim=0)

        # Shift -1 to index the padded dummy row (last row)
        batched_graph_nodes_o[batched_graph_nodes_o == -1] = x_padded.size(0) - 1  # replace -1 with dummy row index

        # Gather node embeddings → shape: [batch_size, max_nodes, hidden_dim]
        batched_x = x_padded[batched_graph_nodes_o]

        # 2. Graph-level mean features
        # -> [batch_size, hidden_size]
        graph_x_mean = torch_scatter.scatter_mean(x, data.batch, dim=0)

        # 3. Get last_hh, dynamic_input, graph_c
        graph_c = graph_x_mean.unsqueeze(0).clone()  # ✅ [1, B, H]
        dynamic_input = graph_x_mean.view(batch_size, 1, -1).clone()
        last_hh = (graph_c.clone(), graph_c.clone())

        while active.any():
            active_idx = active.nonzero(as_tuple=False).view(-1)
            A = active_idx.size(0)
            if A == 0:
                break
            be_a = batched_graph_edges[active_idx]  # [A, 2, E]
            in_edges_a = in_edges[active_idx]
            out_edges_a = out_edges[active_idx]
            batch_indices_a = batch_indices[active_idx]

            if handle_size.sum() > 0:
                # --- Neighborhood mask ---
                n_mask_a = torch.zeros_like(handle_mask, device=handle_mask.device)

                src = be_a[:, 0, :]
                dst = be_a[:, 1, :]
                src_cand = ~in_edges_a & out_edges_a
                dst_cand = in_edges_a & ~out_edges_a
                # Select batch indices and tail indices where selected == 1
                selected_batches = batch_indices_a[src_cand]  # 1D tensor
                selected_src = src[src_cand]  # 1D tensor

                # Update the mask
                n_mask_a[selected_batches, selected_src] = 1

                selected_batches = batch_indices_a[dst_cand]  # 1D tensor
                selected_dst = dst[dst_cand]  # 1D tensor
                # Update the mask
                n_mask_a[selected_batches, selected_dst] = 1
                n_mask_a = n_mask_a[active_idx]

            else:
                n_mask_a = graph_mask.clone()

            # Identify graphs with no valid neighbors
            no_neighbors = (~n_mask_a).all(dim=-1)  # [A] True where no node is valid

            # Deactivate those
            active[active_idx[no_neighbors]] = False

            # Zero out handles for graphs that terminate due to no neighbors, if no neighbors, no candidate teeth....
            for i in active_idx[no_neighbors]:
                handle_mask[i] = False  # clear all selected nodes
                handle_size[i] = 0

            active_idx = active.nonzero(as_tuple=False).view(-1)

            n_mask_a = n_mask_a[~no_neighbors]

            A = active_idx.size(0)
            if A == 0:
                break

            # --- Gather active graph slices ---
            dyn_input_a = dynamic_input[active_idx]  # [A, 1, H]
            h_a = last_hh[0][:, active_idx]  # [1, A, H]
            c_a = last_hh[1][:, active_idx]  # [1, A, H]
            x_a = batched_x[active_idx]  # [A, N, H]

            bgno_a = batched_graph_nodes[active_idx]  # [A, N]
            be_a = batched_graph_edges[active_idx]  # [A, 2, E]
            bem_a = batched_edge_mask[active_idx]  # [A, E]


            # --- LSTM step ---
            rnn_out, (h_new, c_new) = self.lstm(dyn_input_a, (h_a, c_a))  # [A, 1, H], ([1, A, H], [1, A, H])
            last_hh[0][:, active_idx] = h_new[:]
            last_hh[1][:, active_idx] = c_new[:]
            rnn_out = rnn_out.squeeze(1)  # [A, H]

            # --- Attention query vector ---
            h_c = torch.cat([rnn_out.unsqueeze(1), h_new.transpose(0, 1)], dim=-1)  # [A, 1, 2H]

            # --- Attention ---
            att = self.encoder_attn(h_c, x_a, x_a, mask=~n_mask_a)  # [A, 1, H]
            scores = torch.tanh(att.bmm(x_a.transpose(-2, -1)) / normalizer) * 5 #(thiswas10)

            # Calculate attention scores for neighboring nodes
            scores = scores.squeeze(1).masked_fill(~n_mask_a, float('-inf'))  # [A, N]
            # --- Sampling or rollout ---
            if rollout:
                next_node_idx = scores.argmax(dim=1)
            else:
                dist = Categorical(logits=scores)
                next_node_idx = dist.sample()
                log_probs[active_idx] += dist.log_prob(next_node_idx)
            # --- Get global selected node ids ---
            selected_ids = bgno_a.gather(1, next_node_idx.unsqueeze(1)).squeeze(1)  # [A]
            # --- Depot logic ---
            is_depot = (selected_ids == 0)
            override = is_depot & (handle_size[active_idx] == 0)
            stop = is_depot & (handle_size[active_idx] > 0)

            first_non_depot = bgno_a[:, 1]
            selected_ids = torch.where(override, first_non_depot, selected_ids)

            # --- Update active ---
            active[active_idx[stop]] = False

            #--update edges in or out handle:
            # Expand selected_ids for comparison: [A, 1] -> [A, E]
            src = be_a[:, 0, :]
            selected_ids_exp = selected_ids.unsqueeze(1).expand_as(src)  # [A, E]

            # Match: where src == selected node (only valid edges)
            matches = (src == selected_ids_exp) & bem_a  # [A, E]
            # Update in_edges only at active indices
            in_edges[active_idx] |= matches

            dst = be_a[:, 1, :]

            # Match: where src == selected node (only valid edges)
            matches2 = (dst == selected_ids_exp) & bem_a  # [A, E]

            # Update in_edges only at active indices
            out_edges[active_idx] |= matches2

            # --- Update handle mask ---
            sel_mask = (bgno_a == selected_ids.unsqueeze(1))  # [A, N]
            handle_mask[active_idx] |= sel_mask
            handle_size[active_idx] += (~stop).float()

            # --- Update dynamic input with new node embedding ---
            sel_idx = sel_mask.float().argmax(dim=1)
            sel_feats = x_a[torch.arange(A, device=x.device), sel_idx].unsqueeze(1)  # [A, 1, H]
            alpha = (1.0 / handle_size[active_idx]).view(-1, 1, 1)
            dynamic_input[active_idx] = (1 - alpha) * dyn_input_a + alpha * sel_feats

        # Reconstruct handle sets
        handle_sets = []
        for i in range(batch_size):
            mask_row = handle_mask[i]
            node_ids = batched_graph_nodes[i][mask_row]
            # 🚫 remove node 0 (the depot)
            node_ids = node_ids[node_ids != 0]
            handle_sets.append(node_ids.tolist())
        return handle_sets, log_probs, handle_mask


class GATRL(nn.Module):
    def __init__(self, input_dim, edge_input_dim, hidden_dim, num_layers, dropout, beta, heads):
        super(GATRL, self).__init__()
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.beta = beta
        self.heads = heads
        self.edge_input_dim = edge_input_dim
        self.encoder = Encoder(self.input_dim, self.edge_input_dim, self.hidden_dim, self.num_layers, heads=self.heads, beta=self.beta, dropout=self.dropout)
        self.decoder = Decoder_LSTM(self.hidden_dim, self.heads, dropout=self.dropout)

    def forward(self, data, parent_mask=None, rollout=False):
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        # Swap the rows and concatenate horizontally
        swapped_edge_index = torch.stack([edge_index[1, :], edge_index[0, :]], dim=0)
        edge_index = torch.cat((edge_index, swapped_edge_index), dim=1)
        edge_attr = torch.cat((edge_attr, edge_attr), dim=0).view(-1, 1)
        x, edge_index, edge_attr = self.encoder(data.x, edge_index, edge_attr)
        handles, log_probs, mask = self.decoder(x, edge_index, data, parent_mask=parent_mask, rollout=rollout)
        return handles, log_probs, mask

    def forward_b(self, data, parent_mask=None, rollout=False):
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        # Swap the rows and concatenate horizontally
        swapped_edge_index = torch.stack([edge_index[1, :], edge_index[0, :]], dim=0)
        edge_index = torch.cat((edge_index, swapped_edge_index), dim=1)
        edge_attr = torch.cat((edge_attr, edge_attr), dim=0).view(-1, 1)
        x, edge_index, edge_attr = self.encoder(data.x, edge_index, edge_attr)
        handles, log_probs, mask = self.decoder.forward_batched(x, data, rollout=rollout)
        return handles, log_probs, mask

    def encode_init(self, data):
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        # Swap the rows and concatenate horizontally
        swapped_edge_index = torch.stack([edge_index[1, :], edge_index[0, :]], dim=0)
        edge_index = torch.cat((edge_index, swapped_edge_index), dim=1)
        edge_attr = torch.cat((edge_attr, edge_attr), dim=0).view(-1, 1)

        self.x, self.edge_index, self.edge_attr = self.encoder(data.x, edge_index, edge_attr)

    def generation_zero(self, data, data_large, mu=0.5):
        handles_dict = {}  # Dictionary to store unique handles and masks
        # First individual (with rollout=True)
        h, _, m = self.decoder(self.x, self.edge_index, data, rollout=True, mu=mu)
        handle = frozenset(h[0])  # Convert to frozenset for uniqueness
        handles_dict[handle] = {"mask": m, "reward": 0, "original": [], "teeth": [], "rhs":0}

        # Remaining individuals (with rollout=False)
        handles, _, mask = self.forward_b(data_large, rollout=False)
        for id in range(len(handles)):
            h = handles[id]
            m = mask[id]
            #h, _, m = self.decoder(self.x, self.edge_index, data, rollout=False, mu=mu)
            #handle = frozenset(h[0])
            handle = frozenset(h)

            if handle not in handles_dict:  # Store only if unique
                handles_dict[handle] = {"mask": m, "reward": 0, "original": [], "teeth": [], "rhs":0}

        # Convert to ordered lists for indexing if needed
        unique_handles = list(handles_dict.keys())

        return unique_handles, handles_dict

    def generation_next_off(self, data, parent_masks, mu=0.5):
        handles, masks = [[], []], [[], []]
        h, _, m = self.decoder(self.x, self.edge_index, data, parent_mask=parent_masks, rollout=False, mu=mu)
        handles[0] = frozenset(h[0])
        masks[0] = m
        h, _, m = self.decoder(self.x, self.edge_index, data, parent_mask=~parent_masks, rollout=False, mu=mu)
        handles[1] = frozenset(h[0])
        masks[1] = m
        return handles, masks
