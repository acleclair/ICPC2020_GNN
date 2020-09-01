from bs4 import BeautifulSoup
from html.parser import HTMLParser
from myutils import prep, drop, print_ast
import multiprocessing
import pickle
import networkx as nx
import re
import statistics
import numpy as np

def load_good_fid():
    filename = './output/dataset.coms'
    good_fid = []
    for line in open(filename):
        tmp = [x.strip() for x in line.split(',')]
        fid = int(tmp[0])
        good_fid.append(fid)

    return good_fid

prep('loading srcmlunits... ')
srcmlunits = pickle.load(open('srcml-standard.pkl', 'rb'))
sml2 = pickle.load(open('fundatsparsed-srcml-final-allcoms.pkl', 'rb'))

for key, val in sml2.items():
    srcmlunits[key] = val

drop()

def re_0002(i):
    # split camel case and remove special characters
    tmp = i.group(0)
    if len(tmp) > 1:
        if tmp.startswith(' '):
            return tmp
        else:
            return '{} {}'.format(tmp[0], tmp[1])
    else:
        return ' '.format(tmp)

re_0001_ = re.compile(r'([^a-zA-Z0-9 ])|([a-z0-9_][A-Z])')

class MyHTMLParser(HTMLParser):
    def __init__(self):
        super(MyHTMLParser, self).__init__()
        self.parentstack = list()
        self.curtag = -1
        self.tagidx = -1
        self.graph = nx.Graph()
        self.seq = list()
        
    def handle_starttag(self, tag, attrs):
        self.parentstack.append(self.curtag)
        self.tagidx += 1
        self.seq.append(tag)
        self.graph.add_node(self.tagidx, text=tag)
        if self.parentstack[-1] >= 0:
            self.graph.add_edge(self.parentstack[-1], self.tagidx)
        self.curtag = self.tagidx
        
    def handle_endtag(self, tag):
        self.curtag = self.parentstack.pop()
        
    def handle_data(self, data):
        
        # first, do dats text preprocessing
        data = re_0001_.sub(re_0002, data).lower().rstrip()
        
        # second, create a node if there is text
        if(data != ''):
            for d in data.split(' '): # each word gets its own node
                if d != '':
                    self.parentstack.append(self.curtag)
                    self.tagidx += 1
                    self.seq.append(d)
                    self.graph.add_node(self.tagidx, text=d)
                    self.graph.add_edge(self.parentstack[-1], self.tagidx)
                    self.curtag = self.tagidx
                    self.curtag = self.parentstack.pop()
        
    def get_graph(self):
        return(self.graph)

    def get_seq(self):
        return(self.seq)

c = 0

def xmldecode(unit):
    parser = MyHTMLParser()
    parser.feed(unit)
    return(parser.get_graph(), parser.get_seq())

prep('loading tokenizer... ')
smlstok = pickle.load(open('smls.tok', 'rb'), encoding='UTF-8')
drop()

lens = list()
good_fid = load_good_fid()
print('num good fids:', len(good_fid))
srcml_nodes = dict()
srcml_edges = dict()
fopn = open('./output/dataset.srcml_nodes.pkl', 'wb')
fope = open('./output/dataset.srcml_edges.pkl', 'wb')
blanks = 0

def w2i(word):
    try:
        i = smlstok.w2i[word]
    except KeyError:
        i = smlstok.oov_index
    return i

prep('parsing xml... ')
for fid in good_fid:
    try:
        unit = srcmlunits[fid]
    except:
        unit = ''

    (graph, seq) = xmldecode(unit)
    seq = ' '.join(seq)
    c += 1
    
    lens.append(len(graph.nodes.data()))
    
    nodes = list(graph.nodes.data())
    try:
        nodes = np.asarray([w2i(x[1]['text']) for x in list(graph.nodes.data())])
        edges = nx.adjacency_matrix(graph)
    except:
        eg = nx.Graph()
        eg.add_node(0)
        nodes = np.asarray([0])
        edges = nx.adjacency_matrix(eg)
        blanks += 1
    
    srcml_nodes[int(fid)] = nodes
    srcml_edges[int(fid)] = edges

    if(c % 10000 == 0):
        print(c)
drop()

print('blanks:', blanks)
print('avg:', sum(lens) / len(lens))
print('max:', max(lens))
print('median:', statistics.median(lens))
print('% abv 200:', sum(i > 200 for i in lens) / len(lens))

prep('writing pkl... ')
pickle.dump(srcml_nodes, fopn)
pickle.dump(srcml_edges, fope)
drop()

prep('cleaning up... ')
fopn.close()
fope.close()
drop()
