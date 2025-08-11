from graph_classes import *
import sys
import pickle
from datetime import datetime, timedelta
import spacy

class SemanticGraph:
    def __init__(self):
        self.spacy_model = spacy.load("en_core_web_sm")

    def forward(self,doc_sents,doc_events,doc_timexes,doc_timex_vals,doc_id):
        self.get_spacy(doc_sents)
        self.get_graph(doc_sents,doc_events,doc_timexes,doc_timex_vals,doc_id)   
    def get_spacy(self,doc_sents):
        full_doc = []
    
        for sent_num,sent in list(sorted(doc_sents.items(), key = lambda x:x[0])):
            full_doc.extend(sent.split())

        tokenized_doc = spacy.tokens.doc.Doc(self.spacy_model.vocab,words=full_doc)

        for name, proc in self.spacy_model.pipeline:
            tokenized_doc = proc(tokenized_doc)
        self.spacy = tokenized_doc

    def get_graph(self,doc_sents,doc_events,doc_timexes,doc_timex_vals,doc_id):
        self.semantic_graph = Graph()
       
        print(doc_id)
        try:
            doc_date_start = datetime.strptime(doc_id[3:11],"%Y%m%d")
        except:
            try:
                doc_date_start = datetime.strptime(doc_id[2:8],"%y%m%d")
            except:
                print("BAD DATE")
        doc_date_end = doc_date_start + timedelta(days=1)
        doc_date = (doc_date_start,doc_date_end)

        self.get_spacy(doc_sents)
        
        for tid in doc_timex_vals:
            if tid in doc_timexes.keys():
                _,timex_val = doc_timex_vals[tid]

                cmpr = self.date_compare(doc_date,timex_val)
                self.semantic_graph.add_edge(True,"D",tid,"DCT-Timex",cmpr)

                for tid2 in doc_timex_vals:
                    if tid2 in doc_timexes.keys():
                        _,timex_val_2 = doc_timex_vals[tid2]
                        if tid != tid2:
                            cmpr = self.date_compare(timex_val,timex_val_2)
                            self.semantic_graph.add_edge(True,tid,tid2,"Timex-Timex",cmpr)

        event_preds = {}
        for eid in doc_events:
            _,e_word,_ = doc_events[eid]

            event_spacy = self.spacy[e_word]
            event_preds[e_word] = (eid,False)
            
        for tid in doc_timex_vals:
            if tid in doc_timexes.keys():
                _,t_word,_ = doc_timexes[tid]
                timex_spacy = self.spacy[t_word]
                pred = timex_spacy
                while pred.i not in event_preds.keys() and pred.dep_ != "ROOT":
                    pred = pred.head

                if pred.i in event_preds.keys():
                    eid,_ = event_preds[pred.i]
                    self.semantic_graph.add_edge(False,eid,tid,"Pred-Timex")
                    event_preds[pred.i] = (eid,True)

        for e_key in event_preds:
            eid,ebool = event_preds[e_key]
            if ebool == False:
                self.semantic_graph.add_edge(False,eid,eid,"Self-Loop")

    def date_compare(self,t1,t2):
        t1_start,t1_end = t1
        t2_start,t2_end = t2

        if t1_start is None or t2_start is None:
            return "None"
        
        t1_start = t1_start.date()
        t1_end = t1_end.date()
        t2_start = t2_start.date()
        t2_end = t2_end.date()
        
        if t1_start > t2_end:
            return "After"
        if t1_end < t2_start:
            return "Before"
        if t1_start == t2_start and t1_end == t2_end:
            return "Simultaneous"
        
        return "None"

if __name__ == "__main__":
    doc_file = sys.argv[1]
    full_dict = pickle.load(open(doc_file,"rb"))

    timex_file = sys.argv[3]
    all_timex_vals = pickle.load(open(timex_file,"rb"))

    graphs_dict = {}
    for document in full_dict:
        doc_id = document.split('.tml')[0]
        doc_dict = full_dict[document]
        doc_sents = doc_dict["sents"]
        doc_events = doc_dict["events"]
        doc_timexes = doc_dict["timexes"]
        doc_timex_vals = all_timex_vals[doc_id]

        doc_sents = {int(x):y for x,y in doc_sents.items()}
        doc_events = {x:[int(y[0]), int(y[1])-1, y[2]] for x,y in doc_events.items()}
        doc_timexes = {x:[int(y[0]), int(y[1])-1, y[2]] for x,y in doc_timexes.items()}

        doc_len = 0
        for sent_num, sent in list(sorted(doc_sents.items(), key = lambda x:x[0])):
            if sent_num == 0:
                doc_len += len(sent.split())
                continue
            for event in doc_events:
                if doc_events[event][0] == sent_num:
                    doc_events[event][1] = doc_events[event][1] + doc_len
            for timex in doc_timexes:
                if doc_timexes[timex][0] == sent_num:
                    doc_timexes[timex][1] = doc_timexes[timex][1] + doc_len
            doc_len += len(sent.split())
        
        semantic_model = SemanticGraph()
        semantic_model.forward(doc_sents,doc_events,doc_timexes,doc_timex_vals,doc_id)
        semantic_graph = semantic_model.semantic_graph
        graphs_dict[doc_id] = semantic_graph

    out_file = sys.argv[2]
    pickle.dump(graphs_dict,open(out_file,"wb+"))
