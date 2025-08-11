import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import FastRGCNConv, RGATConv, GCNConv, GATConv


def info_nce_loss(A, B, temp=0.1, allow_gradient=False):
    labels = torch.cat([torch.arange(A.size(0)) for i in range(1)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.cuda()

    A = F.normalize(A, dim=1)

    if allow_gradient:
        B = F.normalize(B, dim=1)

    else:
        B = F.normalize(B.detach(), dim=1)
    
    S = torch.matmul(A, B.T)

    mask = torch.eye(labels.shape[0], dtype=torch.bool).cuda()
    labels = labels[~mask].view(labels.shape[0], -1)
    S = S[~mask].view(S.shape[0], -1)

    positives = S[labels.bool()].view(labels.shape[0], -1)
    negatives = S[~labels.bool()].view(S.shape[0], -1)
    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
    logits = logits / temp
    loss = torch.nn.CrossEntropyLoss()(logits, labels)

    return loss

def get_gcn(gcn_type, input_dim, output_dim, num_edge_type):
    if gcn_type == "gcn":
        gcn = GCNConv(input_dim, output_dim)

    if gcn_type == "gat":
        gcn = GATConv(input_dim, output_dim)

    if gcn_type == "rgcn":
        gcn = FastRGCNConv(input_dim, output_dim, num_edge_type)

    if gcn_type == "rgat":
        gcn = RGATConv(input_dim, output_dim, num_edge_type)

    return gcn

class RGCN_v1(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers, num_edge_type, gcn_type="rgcn"):
        super(RGCN_v1, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = hidden_size
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.num_edge_type = num_edge_type
        self.rgcn_layers = nn.ModuleList()
        self.relu = nn.ReLU(inplace=True)
        self.input_proj = nn.Linear(self.num_layers * self.hidden_size, self.hidden_size)
        self.gcn_type = gcn_type

        for i in range(self.num_layers):
            if i == 0:
                self.rgcn_layers.append(get_gcn(self.gcn_type, self.input_dim, self.hidden_size, self.num_edge_type))
            else:
                self.rgcn_layers.append(get_gcn(self.gcn_type, self.hidden_size, self.hidden_size, self.num_edge_type))

    def forward(self, graph):
        node_feat = graph.node_feat.cuda()
        edge_index = graph.edge_index.cuda()
        edge_type = graph.edge_type.cuda()
        graph_reps = []
        cur_graph_rep = node_feat

        for i in range(self.num_layers):
            if self.gcn_type == "gcn":
                h = self.rgcn_layers[i](cur_graph_rep, edge_index)
            else:
                h = self.rgcn_layers[i](cur_graph_rep, edge_index, edge_type)

            cur_graph_rep = self.relu(h)
            graph_reps.append(cur_graph_rep)

        final_graph_rep = self.input_proj(torch.cat(graph_reps, dim=-1))

        return final_graph_rep, graph_reps

class BertTemporalOrdering(nn.Module):
    def __init__(self, encoder, config):
        super(BertTemporalOrdering, self).__init__()
        self.encoder = encoder
        self.config = config
        self.attention_proj = nn.ModuleList()
        self.attention_queries = []
        
        for i in range(5):
            self.attention_proj.append(nn.Linear(self.config.hidden_size, self.config.hidden_size))
            self.attention_queries.append(torch.rand(1, self.config.hidden_size, requires_grad=True))

        self.attention_queries = nn.Parameter(torch.vstack(self.attention_queries))

    def word_attention(self, words, index):
        query = self.attention_queries[index, :].unsqueeze(0)
        projected_words = nn.Tanh()(self.attention_proj[index](words))
        alphas = torch.matmul(query, projected_words.permute(1, 0))
        max_value = torch.max(alphas.detach())
        alphas = alphas - max_value
        alphas = nn.Softmax(dim=1)(alphas)
        att_rep = torch.sum(words * alphas.permute(1,0).repeat(1,self.config.hidden_size), dim=0).unsqueeze(0)
        
        return att_rep

    def forward(self, batch):
        tokenized_inputs, indices = batch
        tokenized_inputs = {x:y.cuda() for x,y in tokenized_inputs.items() if x != 'labels'}
        outputs = self.encoder(**tokenized_inputs, output_hidden_states=True)
        full_emb = outputs[1]
        last_hidden_states = outputs[2][-1]
        sub_reps = [[], [], [], [], []]

        for i, example in enumerate(indices):
            for j, seq in enumerate(example):
                start, end = seq
                cur_words = last_hidden_states[i, start:end, :]
                mean_pool_rep = torch.mean(cur_words, dim=0).unsqueeze(0) if start != end else torch.zeros((1, self.config.hidden_size)).cuda()
                cur_att_rep = self.word_attention(cur_words, j) if start != end else torch.zeros((1, self.config.hidden_size)).cuda()
                sub_reps[j].append(torch.cat([mean_pool_rep, cur_att_rep], dim=1))

        sub_reps = [torch.vstack(x) for x in sub_reps]
        sub_reps.append(full_emb)
        final_reps = torch.cat(sub_reps, dim=1)

        return final_reps, full_emb

class MulCo(nn.Module):
    def __init__(self, syntax_gcn, temporal_gcn, context_encoder, classifier, dropout, k_hops, temp=1, bert_classifier=None, gcn_classifier=None):
        super(MulCo, self).__init__()
        self.syntax_gcn = syntax_gcn
        self.temporal_gcn = temporal_gcn
        self.context_encoder = context_encoder
        self.classifier = classifier
        self.dropout = nn.Dropout(dropout)
        self.temp = temp
        self.bert_classifier = bert_classifier
        self.gcn_classifier = gcn_classifier
        self.clf_loss_fn = nn.CrossEntropyLoss()
        self.context_encoder_temp = None
        self.syntax_gcn_temp = None
        self.temporal_gcn_temp = None

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.syntax_gcn.output_size, nhead=1, batch_first=True).cuda()
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1).cuda()
        self.k_hops = k_hops
        gcn_head_size = self.syntax_gcn.hidden_size
        self.gcn_head = nn.Sequential(nn.Linear(gcn_head_size, 2048, bias=False), nn.Linear(2048, 2048, bias=False), nn.ReLU()).cuda()
        self.bert_head = nn.Sequential(nn.Linear(768, 2048, bias=False), nn.Linear(2048, 2048, bias=False), nn.ReLU()).cuda()        

    def forward(self, batch):
        tokenized_inputs, token_index, indices, doc_ids, epairs, syntax_graphs, temporal_graphs = batch
        syntax_graph_emb_dict = {}
        multi_syntax_graph_emb_dict = {}
        temporal_graph_emb_dict = {}
        multi_temporal_graph_emb_dict = {}
        e1_embs = []
        e2_embs = []
        g_embs = []

        concat_reps = []
        bert_outputs = None
        gcn_outputs = None

        if self.syntax_gcn:
            for i, doc_id in enumerate(doc_ids):
                if doc_id not in syntax_graph_emb_dict:
                    syntax_graph = syntax_graphs[i]
                    syntax_graph_emb, syntax_graph_reps = self.syntax_gcn(syntax_graph)
                    syntax_graph_emb_dict[doc_id] = syntax_graph_emb
                    multi_syntax_graph_emb_dict[doc_id] = syntax_graph_reps
                
                if doc_id not in temporal_graph_emb_dict:
                    temporal_graph = temporal_graphs[i]
                    temporal_graph_emb, temporal_graph_reps = self.temporal_gcn(temporal_graph)
                    temporal_graph_emb_dict[doc_id] = temporal_graph_emb
                    multi_temporal_graph_emb_dict[doc_id] = temporal_graph_reps

                e1, e2 = epairs[i]
                e1_syntax_node_idx = temporal_graphs[i].event_to_word_idx[e1]
                e1_temporal_node_idx = temporal_graphs[i].node_idx[e1]
                e2_syntax_node_idx = temporal_graphs[i].event_to_word_idx[e2]
                e2_temporal_node_idx = temporal_graphs[i].node_idx[e2]                
                e1_syntax_subgraphs = syntax_graph_emb_dict[doc_id][torch_geometric.utils.k_hop_subgraph(e1_syntax_node_idx, self.k_hops, syntax_graphs[i].edge_index)[0]]
                e1_temporal_subgraphs = temporal_graph_emb_dict[doc_id][torch_geometric.utils.k_hop_subgraph(e1_temporal_node_idx, self.k_hops, temporal_graphs[i].edge_index)[0]]
                e1_emb = torch.mean(self.transformer_encoder(torch.vstack([e1_syntax_subgraphs, e1_temporal_subgraphs]).unsqueeze(0)), dim=1)
                e2_syntax_subgraphs = syntax_graph_emb_dict[doc_id][torch_geometric.utils.k_hop_subgraph(e2_syntax_node_idx, self.k_hops, syntax_graphs[i].edge_index)[0]]
                e2_temporal_subgraphs = temporal_graph_emb_dict[doc_id][torch_geometric.utils.k_hop_subgraph(e2_temporal_node_idx, self.k_hops, temporal_graphs[i].edge_index)[0]]
                e2_emb = torch.mean(self.transformer_encoder(torch.vstack([e2_syntax_subgraphs, e2_temporal_subgraphs]).unsqueeze(0)), dim=1)
                e1_embs.append(e1_emb)
                e2_embs.append(e2_emb)   
                
                syntax_subgraph_embs = []
                temporal_subgraph_embs = []

                for syntax_graph_emb in multi_syntax_graph_emb_dict[doc_id]:
                    e1_syntax_subgraph = syntax_graph_emb[torch_geometric.utils.k_hop_subgraph(e1_syntax_node_idx, self.k_hops, syntax_graphs[i].edge_index)[0]]
                    e2_syntax_subgraph = syntax_graph_emb[torch_geometric.utils.k_hop_subgraph(e2_syntax_node_idx, self.k_hops, syntax_graphs[i].edge_index)[0]]
                    syntax_subgraph_emb = torch.mean(self.transformer_encoder(torch.vstack([e1_syntax_subgraph, e2_syntax_subgraph]).unsqueeze(0)), dim=1)
                    syntax_subgraph_embs.append(syntax_subgraph_emb)

                for temporal_graph_emb in multi_temporal_graph_emb_dict[doc_id]:
                    e1_temporal_subgraph = temporal_graph_emb[torch_geometric.utils.k_hop_subgraph(e1_temporal_node_idx, self.k_hops, temporal_graphs[i].edge_index)[0]]
                    e2_temporal_subgraph = temporal_graph_emb[torch_geometric.utils.k_hop_subgraph(e2_temporal_node_idx, self.k_hops, temporal_graphs[i].edge_index)[0]]
                    temporal_subgraph_emb = torch.mean(self.transformer_encoder(torch.vstack([e1_temporal_subgraph, e2_temporal_subgraph]).unsqueeze(0)), dim=1)
                    temporal_subgraph_embs.append(temporal_subgraph_emb)

                g_emb = torch.mean(self.transformer_encoder(torch.vstack(syntax_subgraph_embs + temporal_subgraph_embs).unsqueeze(0)), dim=1)
                g_embs.append(g_emb)

            e1_embs = torch.vstack(e1_embs)
            e2_embs = torch.vstack(e2_embs)
            g_embs = torch.vstack(g_embs)
            gcn_reps = torch.cat([e1_embs, e2_embs], dim=-1)
            concat_reps.append(gcn_reps)
            gcn_outputs = self.gcn_classifier(gcn_reps)

        if self.context_encoder:
            bert_reps, bert_embs = self.context_encoder((tokenized_inputs, indices))
            concat_reps.append(bert_reps)
            bert_outputs = self.bert_classifier(bert_reps)

        final_output = self.classifier(self.dropout(torch.cat(concat_reps, dim=-1)))
        z_gcn_reps = self.gcn_head(g_embs)
        z_bert_reps = self.bert_head(bert_embs)

        loss = 0.0
        if "labels" in tokenized_inputs:
            labels = tokenized_inputs['labels'].cuda()
            cl_loss = (info_nce_loss(z_gcn_reps, z_bert_reps, self.temp) + info_nce_loss(z_bert_reps, z_gcn_reps, self.temp)) / 2
            clf_loss = self.clf_loss_fn(final_output, labels)
            loss = clf_loss + cl_loss
        
        return loss, final_output

