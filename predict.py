import argparse
import os
import pickle
import random
import sys

import numpy as np
import tensorflow as tf

from utils.myutils import batch_gen, init_tf, seq2sent
import keras
import keras.backend as K
from utils.model import create_model
from timeit import default_timer as timer
from models.custom.graphlayer import GCNLayer


def gen_pred(model, data, comstok, comlen, batchsize, config, strat='greedy'):
    # right now, only greedy search is supported...
    tdats, coms, wsmlnodes, wedge_1 = zip(*data.values())
    tdats = np.array(tdats)
    coms = np.array(coms)
    wsmlnodes = np.array(wsmlnodes)
    wedge_1 = np.array(wedge_1)

    for i in range(1, comlen):
        if config['modeltype'] == 'attendgru':
            results = model.predict([tdats,coms], batch_size=batchsize)
        elif config['modeltype'] == 'ast-attendgru':
            results = model.predict([tdats, coms], batch_size=batchsize)
        else:
            results = model.predict([tdats, coms, wsmlnodes, wedge_1],
                                    batch_size=batchsize)
        for c, s in enumerate(results):
            coms[c][i] = np.argmax(s)

    final_data = {}
    for fid, com in zip(data.keys(), coms):
        final_data[fid] = seq2sent(com, comstok)

    return final_data

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('model', type=str, default=None)
    parser.add_argument('--modeltype', dest='modeltype', type=str, default=None)
    parser.add_argument('--gpu', dest='gpu', type=str, default='')
    parser.add_argument('--data', dest='dataprep', type=str, default='./data')
    parser.add_argument('--outdir', dest='outdir', type=str, default='modelout/')
    parser.add_argument('--batch-size', dest='batchsize', type=int, default=300) 
    parser.add_argument('--outfile', dest='outfile', type=str, default=None)

    args = parser.parse_args()

    modelfile = args.model
    outdir = args.outdir
    dataprep = args.dataprep
    gpu = args.gpu
    batchsize = args.batchsize
    modeltype = args.modeltype
    outfile = args.outfile

    config = dict()

    # User set parameters#
    config['maxastnodes'] = 100
    config['asthops'] = 10

    if modeltype == None:
        modeltype = modelfile.split('_')[0].split('/')[-1]


    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    tdatstok = pickle.load(open('%s/tdats.tok' % (dataprep), 'rb'), encoding='UTF-8')
    comstok = pickle.load(open('%s/coms.tok' % (dataprep), 'rb'), encoding='UTF-8')
    smltok = pickle.load(open('%s/smls.tok' % (dataprep), 'rb'), encoding='UTF-8')
   
    seqdata = pickle.load(open('%s/dataset.pkl' % (dataprep), 'rb'))

    allfids = list(seqdata['ctest'].keys())
    datvocabsize = tdatstok.vocab_size
    comvocabsize = comstok.vocab_size
    smlvocabsize = smltok.vocab_size

    config['tdatvocabsize'] = datvocabsize
    config['comvocabsize'] = comvocabsize
    config['smlvocabsize'] = smlvocabsize

    # set sequence lengths
    config['tdatlen'] = 50
    config['comlen'] = len(list(seqdata['ctrain'].values())[0])
    config['smllen'] = len(list(seqdata['strain_nodes'].values())[0])
    
    config['batch_size'] = batchsize

    comlen = len(seqdata['ctest'][list(seqdata['ctest'].keys())[0]])

    config, _ = create_model(modeltype, config)
    print("MODEL LOADED")
    model = keras.models.load_model(modelfile, custom_objects={"tf":tf, "keras":keras,'AlexGraphLayer':GCNLayer})

    node_data = seqdata['stest_nodes']
    edgedata = seqdata['stest_edges']
    config['batch_maker'] = 'graph_multi_1'

    print(model.summary())

    # set up prediction string and output file
    comstart = np.zeros(comlen)
    stk = comstok.w2i['<s>']
    comstart[0] = stk
    outfn = outdir+"/predictions/predict-{}.txt".format(modeltype)
    outf = open(outfn, 'w')
    print("writing to file: " + outfn)
    batch_sets = [allfids[i:i+batchsize] for i in range(0, len(allfids), batchsize)]
    bg = batch_gen(seqdata, 'test', config, nodedata=node_data, edgedata=edgedata)

    for c, fid_set in enumerate(batch_sets):
        st = timer()
        for fid in fid_set:
            seqdata['ctest'][fid] = comstart

        batch = bg.make_batch(fid_set)
   
        batch_results = gen_pred(model, batch, comstok, comlen, batchsize, config, strat='greedy')
        
        for key, val in batch_results.items():
            outf.write("{}\t{}\n".format(key, val))

        end = timer ()
        print("{} processed, {} per second this batch".format((c+1)*batchsize, int(batchsize/(end-st))), end='\r')

    outf.close()        

