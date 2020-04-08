import random
import sys
from timeit import default_timer as timer

import keras
import numpy as np

# do NOT import keras in this header area, it will break predict.py
# instead, import keras as needed in each function

# refactor this so it imports in the necessary functions
dataprep = '/nfs/projects/funcom/data/standard'
sys.path.append(dataprep)

start = 0
end = 0

def init_tf(gpu):
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu


def index2word(tok):
	i2w = {}
	for word, index in tok.w2i.items():
		i2w[index] = word

	return i2w

def seq2sent(seq, tokenizer):
    sent = []
    check = index2word(tokenizer)
    for i in seq:
        sent.append(check[i])

    return(' '.join(sent))
            
class batch_gen(keras.utils.Sequence):
    def __init__(self, seqdata, tt, config, nodedata=None, edgedata=None):
        self.comvocabsize = config['comvocabsize']
        self.tt = tt
        self.batch_size = config['batch_size']
        self.seqdata = seqdata
        self.allfids = list(seqdata['dt%s' % (tt)].keys())
        self.config = config
        self.edgedata = edgedata
        self.nodedata = nodedata
        random.shuffle(self.allfids) # actually, might need to sort allfids to ensure same order

    def __getitem__(self, idx):
        start = (idx*self.batch_size)
        end = self.batch_size*(idx+1)
        batchfids = self.allfids[start:end]
        return self.make_batch(batchfids)

    def make_batch(self, batchfids):
        return self.divideseqs(batchfids, self.seqdata, self.nodedata, self.edgedata, self.comvocabsize, self.tt)
        

    def __len__(self):
        return int(np.ceil(len(list(self.seqdata['dt%s' % (self.tt)]))/self.batch_size))

    def on_epoch_end(self):
        random.shuffle(self.allfids)

    def divideseqs(self, batchfids, seqdata, nodedata, edge1, comvocabsize, tt):
        import keras.utils

        tdatseqs = list()
        comseqs = list()
        smlnodes = list()

        wedge_1 = list()

        comouts = list()

        fiddat = dict()

        for fid in batchfids:

            wtdatseq = seqdata['dt%s' % (tt)][fid]
            wcomseq = seqdata['c%s' % (tt)][fid]
            # wsmlnodes = seqdata['s%s_nodes' % (tt)][fid]
            try:
                wsmlnodes = nodedata[fid]
            except:
                continue

            try:
                edge_1 = edge1[fid]

            except:
                continue

            # crop/expand ast sequence
            wsmlnodes = wsmlnodes[:self.config['maxastnodes']]
            tmp = np.zeros(self.config['maxastnodes'], dtype='int32')
            tmp[:wsmlnodes.shape[0]] = wsmlnodes
            wsmlnodes = np.int32(tmp)

            # crop/expand ast adjacency matrix to dense
            edge_1 = np.asarray(edge_1.todense())
            edge_1 = edge_1[:self.config['maxastnodes'], :self.config['maxastnodes']]
            tmp_1 = np.zeros((self.config['maxastnodes'], self.config['maxastnodes']), dtype='int32')
            tmp_1[:edge_1.shape[0], :edge_1.shape[1]] = edge_1
            edge_1 = np.int32(tmp_1)

            # crop tdat to max tdat len specified by model config
            wtdatseq = wtdatseq[:self.config['tdatlen']]

            if tt == 'test':
                fiddat[fid] = [wtdatseq, wcomseq, wsmlnodes, edge_1]
            else:
                for i in range(0, len(wcomseq)):
                    tdatseqs.append(wtdatseq)
                    smlnodes.append(wsmlnodes)
                    wedge_1.append(edge_1)

                    comseq = wcomseq[0:i]
                    comout = wcomseq[i]
                    comout = keras.utils.to_categorical(comout, num_classes=comvocabsize)

                    for j in range(0, len(wcomseq)):
                        try:
                            comseq[j]
                        except IndexError as ex:
                            comseq = np.append(comseq, 0)

                    comseqs.append(comseq)
                    comouts.append(np.asarray(comout))

 
        tdatseqs = np.asarray(tdatseqs)
        smlnodes = np.asarray(smlnodes)

        wedge_1 = np.asarray(wedge_1)

        comseqs = np.asarray(comseqs)
        comouts = np.asarray(comouts)

        if tt == 'test':
            return fiddat
        else:
            # if self.config['num_output'] == 2:
            # return [[tdatseqs, comseqs, smlnodes, wedge_1],
            #         [comouts, comouts]]
            # else:
            #     if (self.config['use_tdats']):
            return [[tdatseqs, comseqs, smlnodes, wedge_1],
                    comouts]
            #     else:
            # return [[comseqs, smlnodes, wedge_1], comouts]

