from rouge import Rouge
import argparse
import re
import glob
import sys
from utils.myutils import prep, drop
import mlflow

def fil(com):
    ret = list()
    for w in com:
        if not '<' in w:
            ret.append(w)
    return ret

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--data', dest='dataprep', type=str, default='/nfs/projects/funcom/data/standard_3dfiles_graphast/output')  
    parser.add_argument('--outdir', dest='outdir', type=str, default='/nfs/projects/funcom/data/outdir')
    parser.add_argument('--de', type=str, default='\t')


    args = parser.parse_args()


    outdir = args.outdir
    dataprep = args.dataprep
    delim = args.de


    expid = input("Enter experiment_id: ")
    runid = input("Enter run_id: ")
    epoch = input("Enter epoch #: ")


input_file = "/nfs/home/alex/funcom_graph/mlruns/{}/{}/artifacts/predict_E{}/".format(expid, runid, epoch)
print(input_file)
input_file = glob.glob(input_file+'predict*.txt')[0]

sys.path.append(dataprep)

prep('preparing predictions list... ')
preds = dict()
predicts = open(input_file, 'r')
for c, line in enumerate(predicts):
    (fid, pred) = line.split(delim)
    fid = int(fid)
    pred = pred.split()
    pred = fil(pred)
    preds[fid] = ' '.join(pred)
    #print(preds[fid])
predicts.close()
drop()

#db = MySQLdb.connect(host='localhost', user='ports_20k', passwd='s3m3rU', db='sourcerer')
#cur = db.cursor()


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
    com = ' '.join(com)
    refs.append([com])


rouge = Rouge(metrics=['rouge-l'],
                       max_n=4,
                       limit_length=True,
                       length_limit=100,
                       length_limit_type='words',
                       alpha=1.0, # Default F1_score
                       weight_factor=1.2)

# print(refs[0])
# print(newpreds[0])

r = rouge.get_scores(newpreds, refs)
print("ROUGE-L SCORES")
print("F1 -- ",r['rouge-l']['f'])
print("PR -- ",r['rouge-l']['p'])
print("RE -- ",r['rouge-l']['r'])


with mlflow.start_run(run_id=runid, experiment_id=expid) as run:
    mlflow.log_metric('ROUGE_L_F_E{}'.format(epoch), r['rouge-l']['f'])
    mlflow.log_metric('ROUGE_L_P_E{}'.format(epoch), r['rouge-l']['p'])
    mlflow.log_metric('ROUGE_L_R_E{}'.format(epoch), r['rouge-l']['r'])
#pickle.dump(new_data, open('rouge_scores.pkl', 'wb'))