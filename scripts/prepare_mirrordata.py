
def add_mirror_files(filename):
    mirrorFname = filename[:-4] + '_mirror' + filename[-4:]
    towrite = []
    with open(filename, 'r') as f:
        for line in f:
            spls = line.split(' ')
            num = int(spls[-1])
            if num>1000000:
                print 'a'
            newnum = str(1000000+num) + '\n'
            newname = spls[0]
            newname = newname[:-11] + '1' + newname[-10:]
            towrite.append(' '.join([newname, newnum]))
    mf = open(mirrorFname, 'w')
    mf.writelines(towrite)
    mf.close()

def add_mirror_captions(captfile):
    mirrorFname = captfile[:-4] + '_mirror' + captfile[-4:]
    towrite = []
    with open(captfile, 'r') as f:
        for line in f:
            spls = line.split()
            newname = spls[0]
            newname = newname[:-11] + '1' + newname[-10:]
            if '634' in spls[1:] or '655' in spls[1:]:
                where634 = [j for j, nmn in enumerate(spls[1:]) if nmn=='634']
                where655 = [j for j, nmn in enumerate(spls[1:]) if nmn=='655']
                for n34 in where634:
                    spls[1+n34] = '655'
                for n55 in where655:
                    spls[1+n55] = '634'

            tmp = ' '.join(spls[1:]) + '\n'
            towrite.append(' '.join([newname, tmp]))
    mf = open(mirrorFname, 'w')
    mf.writelines(towrite)
    mf.close()

import argparse

parser = argparse.ArgumentParser(description='Generate tsv for vqa_mirror data - from non-mirror data',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--coco_split_dir',
                    help='', required=True)

args = parser.parse_args()

names_files = [args.coco_split_dir+'karpathy_val_images.txt',
               args.coco_split_dir+'karpathy_train_images.txt',
               args.coco_split_dir + 'karpathy_test_images.txt']
captions_file = args.coco_split_dir + 'trainval_captions.txt'

for nf in names_files:
    add_mirror_files(nf)

add_mirror_captions(captions_file)