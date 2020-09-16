import argparse
import re
import sys

from utils.myutils import prep, drop
from nltk.translate.bleu_score import corpus_bleu


# import MySQLdb

def fil(com):
    ret = list()
    for w in com:
        if not '<' in w:
            ret.append(w)
    return ret

def bleu_so_far(refs, preds):
    Ba = corpus_bleu(refs, preds)
    B1 = corpus_bleu(refs, preds, weights=(1,0,0,0))
    B2 = corpus_bleu(refs, preds, weights=(0,1,0,0))
    B3 = corpus_bleu(refs, preds, weights=(0,0,1,0))
    B4 = corpus_bleu(refs, preds, weights=(0,0,0,1))

    Ba = round(Ba * 100, 2)
    B1 = round(B1 * 100, 2)
    B2 = round(B2 * 100, 2)
    B3 = round(B3 * 100, 2)
    B4 = round(B4 * 100, 2)

    ret = ''
    ret += ('for %s functions\n' % (len(preds)))
    ret += ('Ba %s\n' % (Ba))
    ret += ('B1 %s\n' % (B1))
    ret += ('B2 %s\n' % (B2))
    ret += ('B3 %s\n' % (B3))
    ret += ('B4 %s\n' % (B4))
    
    return ret

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

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('input', type=str, default=None)
    parser.add_argument('--data', dest='dataprep', type=str, default='/nfs/projects/funcom/data/standard_3dfiles_graphast/output')  
    parser.add_argument('--outdir', dest='outdir', type=str, default='/nfs/projects/funcom/data/outdir')
    parser.add_argument('--challenge', action='store_true', default=False)
    parser.add_argument('--obfuscate', action='store_true', default=False)
    parser.add_argument('--sbt', action='store_true', default=False)
    parser.add_argument('--de', type=str, default='\t')
    args = parser.parse_args()
    outdir = args.outdir
    dataprep = args.dataprep
    input_file = args.input
    challenge = args.challenge
    obfuscate = args.obfuscate
    sbt = args.sbt
    delim = args.de

    if challenge:
        dataprep = '../data/challengeset/output'

    if obfuscate:
        dataprep = '../data/obfuscation/output'

    if sbt:
        dataprep = '../data/sbt/output'

    if input_file is None:
        print('Please provide an input file to test with --input')
        exit()


    sys.path.append(dataprep)

    prep('preparing predictions list... ')
    preds = dict()
    predicts = open(input_file, 'r')
    for c, line in enumerate(predicts):
        (fid, pred) = line.split(delim)
        fid = int(fid)
        pred = pred.split()
        pred = fil(pred)
        preds[fid] = pred
    predicts.close()
    drop()

    #db = MySQLdb.connect(host='localhost', user='ports_20k', passwd='s3m3rU', db='sourcerer')
    #cur = db.cursor()
    re_0001_ = re.compile(r'([^a-zA-Z0-9 ])|([a-z0-9_][A-Z])')

    refs = list()
    newpreds = list()
    d = 0
    targets = open('%s/coms.test' % (dataprep), 'r')
    for line in targets:
        (fid, com) = line.split(',')
        fid = int(fid)
        com = com.split()
        com = fil(com)

        try:
            newpreds.append(preds[fid])
        except KeyError as ex:
            continue
        
        refs.append([com])

    print('final status')
    for i in range(10):
        print(' '.join(refs[i][0]),'----', ' '.join(newpreds[i]))
    print(bleu_so_far(refs, newpreds))

