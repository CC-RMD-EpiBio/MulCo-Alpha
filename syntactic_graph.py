from graph_classes import *
import sys
import pickle
import spacy

class SyntaxGraph:
    def __init__(self):
        self.spacy_model = spacy.load("en_core_web_sm")

    def forward(self,doc_sents):     
        self.get_spacy(doc_sents)
        self.get_graph(doc_sents)

    def get_spacy(self,doc_sents):
        full_doc = []
    
        for sent_num,sent in list(sorted(doc_sents.items(), key = lambda x:x[0])):
            full_doc.extend(sent.split())

        tokenized_doc = spacy.tokens.doc.Doc(self.spacy_model.vocab,words=full_doc)

        for name, proc in self.spacy_model.pipeline:
            tokenized_doc = proc(tokenized_doc)
        self.spacy = tokenized_doc

    def get_graph(self,doc_sents):
        self.syntax_graph = Graph()

        word_count = 0
        for i in doc_sents:
        
            self.syntax_graph.add_edge(True,"D","v" + str(i),"Doc-Sent")
            if i != len(doc_sents) - 1:
                self.syntax_graph.add_edge(True,"v" + str(i),"v" + str(i+1),"Sent-Sent")

        doc_len = 0
        for sent_num,sent in list(sorted(doc_sents.items(), key = lambda x:x[0])):
            for word_num in range(len(sent.split())):
                self.syntax_graph.add_edge(True,"v" + str(sent_num),"w" + str(word_num+doc_len),"Sent-Word")

                if word_num < len(sent.split()) - 1:
                    self.syntax_graph.add_edge(True,"w" + str(word_num+doc_len),"w" + str(word_num+doc_len+1),"Word-Word")

            doc_len += len(sent.split())

        for token in self.spacy:
            self.syntax_graph.add_edge(False,"w" + str(token.i),"w" + str(token.head.i),"Dependency")

if __name__ == "__main__":
    doc_file = sys.argv[1]
    full_dict = pickle.load(open(doc_file,"rb"))

    graphs_dict = {}
    for document in full_dict:
        doc_id = document.split('.tml')[0]
        doc_dict = full_dict[document]
        doc_sents = doc_dict["sents"]

        doc_sents = {int(x):y for x,y in doc_sents.items()}
        
        syntax_model = SyntaxGraph()
        syntax_model.forward(doc_sents)
        syntax_graph = syntax_model.syntax_graph
        graphs_dict[doc_id] = syntax_graph

    out_file = sys.argv[2]
    pickle.dump(graphs_dict,open(out_file,"wb+"))
