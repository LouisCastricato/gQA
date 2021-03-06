import torch
import torch.nn as nn
from torch.autograd import Variable

def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)

class AttrProxy(object):
    """
    Translates index lookups into attribute lookups.
    To implement some trick which able to use list of nn.Module in a nn.Module
    see https://discuss.pytorch.org/t/list-of-nn-module-in-a-nn-module/219/2
    """
    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))


class Propagator(nn.Module):
    """
    Gated Propagator for GGNN
    Using GRU gating mechanism
    """
    def __init__(self, state_dim, n_edge_types):
        super(Propagator, self).__init__()

        self.n_edge_types = n_edge_types

        self.reset_gate = nn.Sequential(
            nn.Linear(state_dim*3, state_dim),
            nn.Sigmoid()
        )
        self.update_gate = nn.Sequential(
            nn.Linear(state_dim*3, state_dim),
            nn.Sigmoid()
        )
        self.transform = nn.Sequential(
            nn.Linear(state_dim*3, state_dim),
            nn.Tanh()
        )
        self.state_dim = state_dim

    def forward(self, state_in, state_out, state_cur, A, n_nodes): #A = [A_in, A_out]
        A_in = A
        A_out = A

        a_in = torch.bmm(A_in, state_in) #batch size x |V| x state dim
        a_out = torch.bmm(A_out, state_out)
        a = torch.cat((a_in, a_out, state_cur), 2) #batch size x |V| x 3*state dim

        r = self.reset_gate(a.view(-1, self.state_dim*3)) #batch size*|V| x state_dim 
        z = self.update_gate(a.view(-1, self.state_dim*3))
        r = r.view(-1, n_nodes, self.state_dim)
        z = z.view(-1, n_nodes, self.state_dim)
        joined_input = torch.cat((a_in, a_out, r * state_cur), 2)
        h_hat = self.transform(joined_input.view(-1, self.state_dim*3))
        h_hat = h_hat.view(-1, n_nodes, self.state_dim)
        output = (1 - z) * state_cur + z * h_hat

        return output


class GGNN(nn.Module):
    """
    Gated Graph Sequence Neural Networks (GGNN)
    Mode: SelectNode
    Implementation based on https://arxiv.org/abs/1511.05493
    """
    def __init__(self, state_dim, annotation_dim, n_edge_types, n_steps):
        super(GGNN, self).__init__()

        assert (state_dim >= annotation_dim, 'state_dim must be no less than annotation_dim')

        self.state_dim = state_dim
        self.annotation_dim = annotation_dim
        self.n_edge_types = n_edge_types
        self.n_steps = n_steps
        for i in range(self.n_edge_types):
            # incoming and outgoing edge embedding
            in_fc = nn.Linear(self.state_dim, self.state_dim)
            out_fc = nn.Linear(self.state_dim, self.state_dim)
            self.add_module("in_{}".format(i), in_fc)
            self.add_module("out_{}".format(i), out_fc)

        self.in_fcs = AttrProxy(self, "in_")
        self.out_fcs = AttrProxy(self, "out_")

        # Propagation Model
        self.propagator = Propagator(self.state_dim, self.n_edge_types)
        self.linear_gcn = nn.Linear(self.state_dim, self.state_dim)

        # Output Model
        self.graph_rep = nn.Sequential(
            nn.Linear(self.state_dim + self.annotation_dim, self.state_dim), #self.state_dim + self.annotation_dim
            nn.Tanh(),
            nn.Linear(self.state_dim, 1)
        )
        self._initialization()


    def _initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, prop_state, A, annotations=None):
        n_embd = prop_state
        batch, n_nodes, d_input = prop_state.size()
        for i_step in range(self.n_steps):
            #print ("PROP STATE SIZE:", prop_state.size()) #batch size x |V| x state dim
            in_states = []
            out_states = []
            for i in range(self.n_edge_types):
                in_states.append(self.in_fcs[i](prop_state.view(-1, self.state_dim)))
                out_states.append(self.out_fcs[i](prop_state.view(-1, self.state_dim)))
            in_states = torch.stack(in_states).transpose(0, 1).contiguous()
            in_states = in_states.view(-1, n_nodes*self.n_edge_types, self.state_dim)
            out_states = torch.stack(out_states).transpose(0, 1).contiguous()
            out_states = out_states.view(-1, n_nodes*self.n_edge_types, self.state_dim) #batch size x |V||E| x state dim

            prop_state = self.propagator(in_states, out_states, prop_state, A, n_nodes)

        #Highway
        n_embd = self.linear_gcn(prop_state)

        if annotations is not None:
            join_state = torch.cat((n_embd, annotations), 2) #batch size x |V| x 2*state dim
            output = self.graph_rep(join_state.view(-1, self.state_dim + self.annotation_dim))
            out = self.score(output.view(-1, n_nodes))
        else:
            out = n_embd
        #output = output.sum(1)
        return out