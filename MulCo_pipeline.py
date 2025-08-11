import argparse
import os
import pickle
import json
import networkx
from datetime import datetime
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--doc_file', type=str, action='store', required=True, help='Path to file containing processed documents')
parser.add_argument('--gold_pairs', type=str, action='store', required=True, help='Path to file containing gold event pairs')
parser.add_argument('--test_pairs', type=str, action='store', required=True, help='Path to file containing test event pairs')
parser.add_argument('--out_dir', type=str, action='store', required=True, help='Path to store model outputs')
parser.add_argument('--event_map', type=str, action='store', default='data/event_map.pkl', help='Path to file containing e-ID to ei-ID mapping')
parser.add_argument('--bert_model', type=str, action='store', default='bert', help='BERT model to use')
parser.add_argument('--bert_encoder_type', type=str, action='store', default='neighbor', help='BERT encoding type to use')
parser.add_argument('--learning_rate', type=float, action='store', default=1e-5, help='Learning rate')
parser.add_argument('--dropout', type=float, action='store', default=0.5, help='Learning rate')
parser.add_argument('--batch_size', type=int, action='store', default=16, help='Batch size')
parser.add_argument('--accumulation_steps', type=int, action='store', default=2, help='Accumulation steps')
parser.add_argument('--epochs', type=int, action='store', default=10, help='Number of epochs')
parser.add_argument('--syntax_file', type=str, action='store', default='syntax_graph.pkl', help='Path to file containing syntax graphs')
parser.add_argument('--temporal_file', type=str, action='store', default='time_graph.pkl', help='Path to file containing temporal graphs')
parser.add_argument('--gcn_hidden_size', type=int, action='store', default=64, help='Hidden size for syntax GCN module')
parser.add_argument('--gcn_num_layers', type=int, action='store', default=1, help='Number of layers for syntax GCN module')
parser.add_argument('--dist_cutoff', type=int, action='store', default=2, help='Distance at which events should be considered short-distance')
parser.add_argument('--dist_type', type=str, action='store', default="mix", help='mix/long/short training only')
parser.add_argument('--bert_or_gcn', type=str, action='store', default="both", help='bert/gcn/both for temporal model')
parser.add_argument('--gcn_type', type=str, action='store', default="rgcn", help='type of gcn used in the model')
parser.add_argument('--seed', type=int, action='store', default=2513, help='Random seed for reproducibility')
parser.add_argument('--k_hops', type=int, action='store', default=1, help='Number of hop for k-hop subgraph for each node')
parser.add_argument('--temp', type=float, action='store', default=0.1, help='Temperature for contrastive objective')
args = parser.parse_args()
seed = args.seed   

import numpy
import random
import torch

if seed > 0:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    numpy.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print("Set seed", seed)
    
import torch_geometric
from torch_scatter import scatter_mean
from graph_classes import *
from syntactic_graph import *
from semantic_graph import *
from MulCo_model import *
from MulCo_trainers import BaseTrainer
from transformers import AutoConfig, AutoModel, AutoTokenizer

syn_edge_type_dict = {'Doc-Sent': 0, 'Sent-Sent': 1, 'Sent-Word': 2, 'Word-Word': 3, 'Dependency': 4}
temporal_edge_type_dict = {'DCT-Timex': 0, 'Pred-Timex': 1, 'Timex-Timex': 2, 'Self-Loop':3}


def compute_scores(preds, gold, label_vocab, ignore_files=[], distance_flags=None):
    # print(gold)
    # print(preds)
    label_reverse = {'i': 'ii', 'ii': 'i', 'b': 'a', 'a': 'b', 'm': 'im', 'im': 'm', 's': 's', 'v': 'v'}
    gold_count = 0.0
    pred_count = 0.0
    correct_count = 0.0
    short_distance_gold = 0.0
    short_distance_pred = 0.0
    short_distance_correct = 0.0
    long_distance_gold = 0.0
    long_distance_pred = 0.0
    long_distance_correct = 0.0
    for file in preds:
        if ignore_files and file in ignore_files:
            print('Ignoring {}'.format(file))
            continue
        for epair in gold[file]:
            gold_count += 1
            if distance_flags is not None:
                if distance_flags[file][epair]:
                    short_distance_gold += 1
                else:
                    long_distance_gold += 1
            if epair not in preds[file] and (epair[1], epair[0]) not in preds[file]:
                continue
            pred_count += 1
            if distance_flags is not None:
                if distance_flags[file][epair]:
                    short_distance_pred += 1
                else:
                    long_distance_pred += 1
            if epair in preds[file]:
                if preds[file][epair] == gold[file][epair]:
                    correct_count += 1
                    if distance_flags is not None:
                        if distance_flags[file][epair]:
                            short_distance_correct += 1
                        else:
                            long_distance_correct += 1
            elif (epair[1], epair[0]) in preds[file]:
                rev_epair = (epair[1], epair[0])
                pred_label = ''
                for key in label_vocab:
                    if label_vocab[key] == preds[file][rev_epair]:
                        pred_label = key
                rev_pred_label = label_vocab[label_reverse[pred_label]]
                if rev_pred_label == gold[file][epair]:
                    correct_count += 1
                    if distance_flags is not None:
                        if distance_flags[file][epair]:
                            short_distance_correct += 1
                        else:
                            long_distance_correct += 1

    precision = correct_count / pred_count if pred_count != 0.0 else 0.0
    recall = correct_count / gold_count if gold_count != 0.0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if precision+recall != 0.0 else 0.0
    short_precision = short_distance_correct / short_distance_pred if short_distance_pred != 0.0 else 0.0
    short_recall = short_distance_correct / short_distance_gold if short_distance_gold != 0.0 else 0.0
    short_f1 = (2 * short_precision * short_recall) / (short_precision + short_recall) if short_precision+short_recall != 0.0 else 0.0
    long_precision = long_distance_correct / long_distance_pred if long_distance_pred != 0.0 else 0.0
    long_recall = long_distance_correct / long_distance_gold if long_distance_gold != 0.0 else 0.0
    long_f1 = (2 * long_precision * long_recall) / (long_precision + long_recall) if long_precision+long_recall != 0.0 else 0.0
    
    results = {
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ShortDistancePrecision": short_precision,
        "ShortDistanceRecall": short_recall,
        "ShortDistanceF1": short_f1,
        "LongDistancePrecision": long_precision,
        "LongDistanceRecall":long_recall,
        "LongDistanceF1": long_f1
    }

    return results

def to_torch_syntax_graph(graph, graph_type="syntax"):
    if graph_type == "syntax":
        type_dict = syn_edge_type_dict

    else:
        type_dict = temporal_edge_type_dict

    nodes = set()
    edges = []
    
    for edge in graph.edge_list:
        if hasattr(edge, 'orig_node'):
            edges.append((edge.orig_node, edge.targ_node, type_dict[edge.relation]))
            nodes.update([edge.orig_node, edge.targ_node])

        else:
            edges.append((edge.node_one, edge.node_two, type_dict[edge.relation]))
            nodes.update([edge.node_one, edge.node_two])

    if graph_type == "syntax":
        nodes = sorted([n for n in nodes if "D" in n]) + sorted([n for n in nodes if "v" in n], key=lambda x:int(x[1:])) + sorted([n for n in nodes if "w" in n], key=lambda x:int(x[1:]))

    else:
        nodes = sorted([n for n in nodes if "D" in n]) + sorted([n for n in nodes if "ei" in n], key=lambda x:int(x[2:])) + sorted([n for n in nodes if "t" in n], key=lambda x:int(x[1:]))

    graph = networkx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_weighted_edges_from(edges)
    graph_torch = torch_geometric.utils.from_networkx(graph)
    graph_torch.nodes = list(graph.nodes)
    graph_torch.node_feat = None
    graph_torch.edge_type = graph_torch.weight
    graph_torch.node_idx = {node:i for i, node in enumerate(graph.nodes)}

    return graph_torch

@torch.no_grad()
def get_node_feat(encoder, graph, tokenized_doc_sents):
    sent_nums = tokenized_doc_sents["input_ids"].shape[0]
    mappings = [tokenized_doc_sents.word_ids(i) for i in range(sent_nums)]
    tokenized_inputs = {x:y for x,y in tokenized_doc_sents.items()}
    output = encoder(**tokenized_inputs, output_hidden_states=True)    
    last_four_layer_embeds = output.hidden_states[-2:]
    token_embeds = torch.mean(torch.vstack([x.unsqueeze(0) for x in last_four_layer_embeds]), dim=0)
    
    sent_embeds = []
    node_feats = []

    for i in range(sent_nums):
        sent_tokens = token_embeds[i]
        mapping = [m if m != None else -1 for m in mappings[i]]
        mapping = [m if m >= 0 else max(mapping)+1 for m in mapping]
        index = torch.Tensor(mapping).long()
        node_feat = scatter_mean(sent_tokens, index, dim=0)[:-1]
        sent_embeds.append(torch.mean(node_feat, dim=0))
        node_feats.append(node_feat)

    sent_embeds = torch.vstack(sent_embeds)
    doc_embed = torch.mean(sent_embeds, dim=0).unsqueeze(0)

    node_feats = torch.vstack(node_feats)
    final_node_feat = torch.vstack([doc_embed, sent_embeds, node_feats])
    torch.cuda.empty_cache()
    
    return final_node_feat

def event_re_mapping(sents, timex):
    offset = 0
    sent_offset = []

    for sent in sents:
        sent_offset.append(offset)
        offset += len(sent)

    timex_word_mapping = {}

    for t_id, (sent_num, pos, tokens, _) in timex.items():
        tokens_len = len(tokens.split())
        timex_word_mapping[t_id] = [sent_offset[int(sent_num)] + pos, sent_offset[int(sent_num)] + pos + tokens_len]

    return timex_word_mapping


if __name__ == '__main__':
    if args.bert_model == "bert":
        bert_model_name = "bert-base-uncased"

    elif args.bert_model == "roberta":
        bert_model_name = "roberta-base"

    else:
        bert_model_name = "bert-base-uncased"

    model_path = os.path.join("./models", str(abs(hash(str(vars(args)))) % 10000))
    cache_dir = os.path.join("cache-%s"%(args.bert_model), args.gold_pairs)

    if not os.path.isdir(model_path):
        os.makedirs(model_path)

    if not os.path.isdir(cache_dir):
        os.makedirs(cache_dir)

    test_files = ['APW19980418.0210', 'CNN19980213.2130.0155', 'APW19980227.0494', \
            'CNN19980126.1600.1104', 'NYT19980402.0453', 'APW19980227.0489', \
            'PRI19980306.2000.1675', 'PRI19980115.2000.0186', 'APW19980308.0201']
    dev_files = ['APW19980227.0487', 'PRI19980216.2000.0170', 'ed980111.1130.0089', \
            'CNN19980223.1130.0960', 'NYT19980212.0019']

    if "tbd" in args.gold_pairs or "tdd" in args.gold_pairs:
        rel_dict = {'a': 0, 'b': 1, 's': 2, 'i': 3, 'ii': 4, 'v': 5}

    else: 
        rel_dict = {'a': 0, 'b': 1, 'e': 2, 'v': 3}

    if args.bert_model == "roberta":
        tokenizer = AutoTokenizer.from_pretrained(bert_model_name, add_prefix_space=True)

    else:
        tokenizer = AutoTokenizer.from_pretrained(bert_model_name)

    bert_config = AutoConfig.from_pretrained(bert_model_name)
    bert_encoder = AutoModel.from_pretrained(bert_model_name, config=bert_config)

    full_dict = pickle.load(open(args.doc_file, 'rb'))
    annotated_event_pairs = pickle.load(open(args.gold_pairs, 'rb'))
    test_event_pairs = pickle.load(open(args.test_pairs, 'rb'))
    event_map = pickle.load(open(args.event_map, 'rb'))
    syntax_graphs = pickle.load(open(args.syntax_file, 'rb'))
    temporal_graphs = pickle.load(open(args.temporal_file, 'rb'))
    timex_dict = pickle.load(open('timex_dict.pkl', 'rb'))

    train_data = []
    dev_data = []
    test_data = []
    gold_test_labels = {}
    distance_flags = {}
    
    for document in tqdm(full_dict, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):
        doc_id = document.split('.tml')[0]
        distance_flags[doc_id] = {}
        # Since some documents are not present in TDD-Man, those files will be skipped
        if doc_id not in annotated_event_pairs:
            continue

        doc_dict = full_dict[document]
        doc_sents = doc_dict["sents"]
        doc_sents = {int(x):y for x,y in doc_sents.items()}
        doc_events = doc_dict["events"]
        doc_events = {x:[int(y[0]), int(y[1])-1, y[2]] for x,y in doc_events.items()}
        
        if "hand" not in args.temporal_file:
            doc_timexes = timex_dict[doc_id]
            doc_timexes = {x:[int(y[0]), int(y[1]), y[2], y[3]] for x,y in doc_timexes.items()}

        else:
            doc_timexes = doc_dict["timexes"]
            doc_timexes = {x:[int(y[0]), int(y[1])-1, y[2], y[3]] for x,y in doc_timexes.items()}
        
        max_sent_num = max(list(doc_sents.keys()))
        word_idx_to_loc = []
        loc_to_word_idx = {}
        event_to_word_idx = {}
        sents = []

        for i, sent in sorted(doc_sents.items(), key=lambda x:x[0]):
            tokens = sent.split()
            sents.append(tokens)

            for j, token in enumerate(tokens):
                word_idx_to_loc.append((i, j))
        
        for i, loc in enumerate(word_idx_to_loc):
            loc_to_word_idx[loc] = i

        for e, loc in doc_events.items():
            sent, offset, _ = loc
            event_to_word_idx[e] = loc_to_word_idx[(sent, offset)] + len(doc_sents) + 1

        for e, loc in doc_timexes.items():
            sent, offset, _, _ = loc
            event_to_word_idx[e] = loc_to_word_idx[(sent, offset)] + len(doc_sents) + 1

        syntax_graph = to_torch_syntax_graph(syntax_graphs[doc_id], graph_type="syntax")
        tokenized_doc_sents = tokenizer(
            sents,
            padding='max_length',
            truncation=True,
            is_split_into_words=True,
            max_length=512,
            return_tensors='pt'
        )

        try:
            if args.re_init:
                1 / 0
            node_feat = torch.load(os.path.join(cache_dir, "%s-graph.pkl"%(doc_id)))

        except: 
            node_feat = get_node_feat(bert_encoder, syntax_graph, tokenized_doc_sents)
            torch.save(node_feat, os.path.join(cache_dir, "%s-graph.pkl"%(doc_id)))

        syntax_graph.node_feat = node_feat
        syntax_graph.tokenized_doc_sents = tokenized_doc_sents
        temporal_graph = to_torch_syntax_graph(temporal_graphs[doc_id], graph_type="temporal")
        temporal_graph.tokenized_doc_sents = tokenized_doc_sents
        temporal_graph.event_to_word_idx = {e:i for e, i in event_to_word_idx.items() if e in temporal_graph.nodes}
        temporal_graph.node_feat = torch.vstack([syntax_graph.node_feat[0]] + [syntax_graph.node_feat[i] for e, i in temporal_graph.event_to_word_idx.items()])
        syntax_graph.event_to_word_idx = temporal_graph.event_to_word_idx
        current_event_pairs = annotated_event_pairs[doc_id]

        if doc_id in test_files:
            gold_test_labels[doc_id] = {}
            for epair in current_event_pairs:
                e1, e2, label = epair
                if e1.startswith('t') or e2.startswith('t'):
                    continue
                if label == 'v':
                    continue
                gold_test_labels[doc_id][(event_map[doc_id][e1], event_map[doc_id][e2])] = rel_dict[label]
                e1_sent_index = doc_events[event_map[doc_id][e1]][0]
                e2_sent_index = doc_events[event_map[doc_id][e2]][0]
                distance_flags[doc_id][(event_map[doc_id][e1], event_map[doc_id][e2])] = True \
                    if abs(e1_sent_index - e2_sent_index) < args.dist_cutoff else False
            current_event_pairs = test_event_pairs[doc_id]
        
        for epair in current_event_pairs:
            e1, e2, label = epair
            if label == 'v':
                continue
            if e1 not in event_map[doc_id] or e2 not in event_map[doc_id]:
                continue
            if event_map[doc_id][e1] not in doc_events or event_map[doc_id][e2] not in doc_events:
                continue
            e1_data = doc_events[event_map[doc_id][e1]]
            e2_data = doc_events[event_map[doc_id][e2]]
            e1_sent_index = e1_data[0]
            e2_sent_index = e2_data[0]
            distance_flags[doc_id][(event_map[doc_id][e1], event_map[doc_id][e2])] = True \
                if abs(e1_sent_index - e2_sent_index) < args.dist_cutoff else False
            if abs(e1_sent_index - e2_sent_index) > 15 and doc_id in test_files:
                continue
            
            if args.dist_type != "mix":
                if args.dist_type  == "short" and not distance_flags[doc_id][(event_map[doc_id][e1], event_map[doc_id][e2])]:
                    continue
            
                if args.dist_type == "long" and distance_flags[doc_id][(event_map[doc_id][e1], event_map[doc_id][e2])]:
                    continue
            
            min_sent = max(0, min([e1_sent_index, e2_sent_index])-1)
            min_token_index = 0

            for i in range(0, min_sent):
                min_token_index += len(doc_sents[i].split())

            max_sent = min(max_sent_num, max([e1_sent_index, e2_sent_index])+1)
            min_event_sent, min_event_word = -1, -1
            max_event_sent, max_event_word = -1, -1
            if e1_sent_index < e2_sent_index:
                min_event_sent = e1_sent_index
                max_event_sent = e2_sent_index
                min_event_word = e1_data[1]
                max_event_word = e2_data[1]
            elif e1_sent_index > e2_sent_index:
                min_event_sent = e2_sent_index
                max_event_sent = e1_sent_index
                min_event_word = e2_data[1]
                max_event_word = e1_data[1]
            else:
                min_event_sent = e1_sent_index
                max_event_sent = e2_sent_index
                if e1_data[1] < e2_data[1]:
                    min_event_word = e1_data[1]
                    max_event_word = e2_data[1]
                else:
                    min_event_word = e2_data[1]
                    max_event_word = e1_data[1]
            total_text = []
            total_text_index = []
            for i in range(min_sent, max_sent+1):
                tokens = doc_sents[i].split()
                total_text += tokens
                total_text_index += [idx+min_token_index+1+len(doc_sents) for idx, t in enumerate(tokens)]
                min_token_index += len(tokens)

            min_event_word_index = 0
            for i in range(min_sent, min_event_sent):
                min_event_word_index += len(doc_sents[i].split())
            min_event_word_index += min_event_word
            max_event_word_index = 0
            for i in range(min_sent, max_event_sent):
                max_event_word_index += len(doc_sents[i].split())
            max_event_word_index += max_event_word
            indices = [
                    [0, min_event_word_index], 
                    [min_event_word_index, min_event_word_index+1], 
                    [min_event_word_index+1, max_event_word_index], 
                    [max_event_word_index, max_event_word_index+1], 
                    [max_event_word_index+1,len(total_text)]
            ]
            example = [total_text, total_text_index, indices, rel_dict[label] if label != '' else label, doc_id, event_map[doc_id][e1], event_map[doc_id][e2], syntax_graph, temporal_graph]
            if doc_id in test_files:
                test_data.append(example)
            elif doc_id in dev_files:
                dev_data.append(example)
            else:
                train_data.append(example)
    
    print('Loaded {} training examples'.format(len(train_data)))
    print('Loaded {} development examples'.format(len(dev_data)))
    print('Loaded {} test examples'.format(len(test_data)))

    def batch_examples(examples, split, batch_size):
        batched_examples = []
        for start in range(0, len(examples), batch_size):
            end = min(len(examples), start+batch_size)
            cur_examples = examples[start:end]
            example_texts = [x[0] for x in cur_examples]
            example_token_index = [x[1] for x in cur_examples]
            tokenized_inputs = tokenizer(
                example_texts,
                padding='max_length',
                truncation=True,
                is_split_into_words=True,
                max_length=512,
                return_tensors='pt'
            )
            renewed_indices = []
            doc_ids = []
            epairs = []
            syntax_graphs = []
            temporal_graphs = []
            for i, example in enumerate(cur_examples):
                doc_ids.append(example[4])
                epairs.append((example[5], example[6]))
                syntax_graphs.append(example[7])
                temporal_graphs.append(example[8])
                indices = example[2]
                subtoken_indices = []
                start_index = 1
                mapping = tokenized_inputs.word_ids(batch_index=i)
                for cur_index_pair in indices:
                    subtoken_indices.append([start_index])
                    while mapping[start_index] != None and mapping[start_index] < cur_index_pair[1]:
                        start_index += 1
                    subtoken_indices[-1].append(start_index)
                renewed_indices.append(subtoken_indices)
            if split != 'test':
                tokenized_inputs['labels'] = torch.LongTensor([x[3] for x in cur_examples])
            batched_examples.append([tokenized_inputs, example_token_index, renewed_indices, doc_ids, epairs, syntax_graphs, temporal_graphs])
        return batched_examples

    train_batches = batch_examples(train_data, 'train', args.batch_size)
    dev_batches = batch_examples(dev_data, 'dev', args.batch_size)
    test_batches = batch_examples(test_data, 'test', args.batch_size)

    context_encoder = BertTemporalOrdering(bert_encoder.cuda(), bert_config).cuda()
    bert_output_units = 11 * bert_config.hidden_size
    syntax_gcn = RGCN_v1(bert_config.hidden_size, args.gcn_hidden_size, args.gcn_num_layers, len(syn_edge_type_dict), args.gcn_type).cuda()
    temporal_gcn = RGCN_v1(bert_config.hidden_size, args.gcn_hidden_size, args.gcn_num_layers, len(temporal_edge_type_dict), args.gcn_type).cuda() 
    gcn_output_units = args.gcn_hidden_size * 2
    final_units = bert_output_units + gcn_output_units
    
    classifier = nn.Sequential(nn.Linear(final_units, len(rel_dict)), nn.ReLU(), nn.Softmax()).cuda()
    gcn_classifier = nn.Sequential(nn.Linear(gcn_output_units, len(rel_dict)), nn.ReLU(), nn.Softmax()).cuda()
    bert_classifier = nn.Sequential(nn.Linear(bert_output_units, len(rel_dict)), nn.ReLU(), nn.Softmax()).cuda()
    
    model = MulCo(syntax_gcn, temporal_gcn, context_encoder, classifier, args.dropout, args.k_hops, args.temp, bert_classifier, gcn_classifier)
    
    epochs = args.epochs
    trainer = BaseTrainer(args.learning_rate)
    trainer.train(model, train_batches, dev_batches, args.epochs, model_path, rel_dict, distance_flags, args.accumulation_steps)
    mulco_predictions = trainer.test(model, test_batches, model_path)
    results = compute_scores(mulco_predictions, gold_test_labels, rel_dict, distance_flags=distance_flags)
    print(results)

    json.dump(results, open(os.path.join(model_path, "eval_scores.json"), "w"), indent=4)
    json.dump(vars(args), open(os.path.join(model_path, "hyperparameters.json"), "w"), indent=4)

