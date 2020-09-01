import pickle

bad_fid = pickle.load(open('autogenfid.pkl', 'rb'))

comdata = 'com_pp.txt'
good_fid = []
outfile = './output/dataset.coms' 

fo = open(outfile, 'w')
for line in open(comdata):
    tmp = line.split(',')
    fid = int(tmp[0].strip())
    if bad_fid[fid]:
        continue  
    com = tmp[1].strip()
    com = com.split()
    if len(com) > 13 or len(com) < 3:
    	continue
    com = ' '.join(com)
    fo.write('{}, <s> {} </s>\n'.format(fid, com))
            

fo.close()
