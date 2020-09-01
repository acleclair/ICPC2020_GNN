import pickle
import collections
import random
import sys

random.seed = 1337

def load(fname):
    return pickle.load(open(fname, 'rb'))

def write(data, fname):
    fo = open(fname, 'w')
    for fid, string in data.items():
        fo.write("{}, {}\n".format(fid, string))
    fo.close()
    
valsize = 0.05
testsize = 0.05

tdatfile = './output/dataset.tdats'
sdatfile = './output/dataset.sdats'
comfile = './output/dataset.coms'
smlfile = './output/dataset.srcml'
# Make dictionary of unique source code entries
tdats = collections.defaultdict(str)
sdats = collections.defaultdict(str)
seen_dats = set()
dup_count = 0
for line in open(tdatfile):
    tmp = [x.strip() for x in line.split(',')]
    fid = int(tmp[0])
    data = tmp[1]

    if data == '':
        continue

    if data in seen_dats:
        dup_count += 1
        continue
    else:
        tdats[fid] = data
        seen_dats.add(data)

for line in open(sdatfile):
    tmp = [x.strip() for x in line.split(',')]
    fid = int(tmp[0])
    data = tmp[1]
    if fid in tdats.keys():
        sdats[fid] = data

# Load comments for above source entries
coms = collections.defaultdict(str)
for line in open(comfile):
    tmp = [x.strip() for x in line.split(',')]
    fid = int(tmp[0])
    data = tmp[1]
    if fid in tdats.keys():
        coms[fid] = data

# Load src ml tokens for above source entries
srcml = collections.defaultdict(str)
for line in open(smlfile):
    tmp = [x.strip() for x in line.split(',')]
    fid = int(tmp[0])
    data = tmp[1]
    if fid in tdats.keys():
        srcml[fid] = data

assert len(tdats) == len(coms), "tdats and coms not same length"
assert len(sdats) == len(coms), "sdats and coms not same length"

#bad fidproj-j1-fids.pkl are projects that did not have a valid PID, so we generated some for them
projfile = 'fidproj-j1-t.pkl'
badprojfile = 'fidproj-j1-badfids.pkl'

#these projects had full obfuscation done, so we force them into the test set
reserved_pids = [10846, 35863, 52302, 3474]

proj = load(projfile)
badproj = load(badprojfile)

l1 = len(proj)
l2 = len(badproj)
for k, v in badproj.items():
    proj[k] = v

assert len(proj) == (l1+l2), "project list not combined correctly"



# make proj->fid dict
proj2fid = collections.defaultdict(list)
for fid, pid in proj.items():
    if pid not in reserved_pids:
        proj2fid[pid].append(fid)

# split them into 3 by project
test_num = int(testsize*len(proj2fid))
val_num = int(valsize*len(proj2fid))

plist = list(proj2fid.keys())
# probably pointless to shuffle 10 times, but it makes me feel good
for i in range(10):
    random.shuffle(plist)

val_pids = plist[:val_num]
test_pids = plist[val_num:val_num+test_num]
train_pids = plist[val_num+test_num:]

test_pids += reserved_pids

assert len(val_pids)+len(test_pids)+len(train_pids) == len(plist)+len(reserved_pids), "val/test/train not same length as pid list"
print(len(plist))
train_fids = []
val_fids = []
test_fids = []

count = 0

train_coms = collections.defaultdict(str)
train_tdats = collections.defaultdict(str)
train_sdats = collections.defaultdict(str)
train_smls = collections.defaultdict(str)
for pid in train_pids:
    for fid in proj2fid[pid]:
        #if sdats[fid] != '' and tdats[fid] != '' and coms[fid] != '':
        if tdats[fid] != '' and coms[fid] != '':
            train_fids.append(fid)
            train_coms[fid] = coms[fid]
            train_tdats[fid] = tdats[fid]
            train_sdats[fid] = sdats[fid]
            train_smls[fid] = srcml[fid]
        else:
            count += 1
            
assert len(train_coms) == len(train_tdats), "Train coms/dats different lengths"

val_coms = collections.defaultdict(str)
val_tdats = collections.defaultdict(str)
val_sdats = collections.defaultdict(str)
val_smls = collections.defaultdict(str)
for pid in val_pids:
    for fid in proj2fid[pid]:
        #if sdats[fid] != '' and tdats[fid] != '' and coms[fid] != '':
        if tdats[fid] != '' and coms[fid] != '':
            val_fids.append(fid)
            val_coms[fid] = coms[fid]
            val_tdats[fid] = tdats[fid]
            val_sdats[fid] = sdats[fid]
            val_smls[fid] = srcml[fid]
        else:
            count += 1
assert len(val_coms) == len(val_tdats), "Train coms/dats different lengths"

test_coms = collections.defaultdict(str)
test_tdats = collections.defaultdict(str)
test_sdats = collections.defaultdict(str)
test_smls = collections.defaultdict(str)
for pid in test_pids:
    for fid in proj2fid[pid]:
        #if sdats[fid] != '' and tdats[fid] != '' and coms[fid] != '':
        if tdats[fid] != '' and coms[fid] != '':
            test_fids.append(fid)
            test_coms[fid] = coms[fid]
            test_tdats[fid] = tdats[fid]
            test_sdats[fid] = sdats[fid]
            test_smls[fid] = srcml[fid]
        else:
            count += 1
assert len(test_coms) == len(test_tdats), "Train coms/dats different lengths"

print(len(train_tdats)+len(test_tdats)+len(val_tdats))
print(len(train_pids)+len(test_pids)+len(val_pids))
print("{} blank dats or coms".format(count))
print(len(proj))
print(dup_count)

write(train_coms, "./output/coms.train")
write(train_tdats, "./output/tdats.train")
write(train_sdats, "./output/sdats.train")
write(train_smls, "./output/smls.train")
write(val_coms, "./output/coms.val")
write(val_tdats, "./output/tdats.val")
write(val_sdats, "./output/sdats.val")
write(val_smls, "./output/smls.val")
write(test_coms, "./output/coms.test")
write(test_tdats, "./output/tdats.test")
write(test_sdats, "./output/sdats.test")
write(test_smls, "./output/smls.test")
