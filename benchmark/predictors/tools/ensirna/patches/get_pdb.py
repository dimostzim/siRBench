import pandas as pd

import json
from utils.rna_utils import VOCAB
import argparse
import os
import sys

import torch
import RNA
import re
import subprocess
import multiprocessing
import tempfile


#example （5'->3' and 3'->5'）
#seq1 = "CUUACGCUGAGUACUUCGA".lower()
#seq2 = "GAAUGCGACUCAUGAAGCU".lower()[::-1]

#python -m data.get_pdb


def resolve_rosetta_dir(rosetta_dir=None):
    if rosetta_dir:
        base = rosetta_dir
    else:
        base = os.environ.get("ROSETTA_DIR", "/app/ENsiRNA-main/rosetta/rosetta.binary.linux.release-371")
    ff_candidates = [
        os.path.join(base, "main/source/bin/rna_denovo.static.linuxgccrelease"),
        os.path.join(base, "main/source/bin/rna_denovo.linuxgccrelease"),
        os.path.join(base, "bin/rna_denovo.default.linuxgccrelease"),
        os.path.join(base, "usr/local/bin/rna_denovo.default.linuxgccrelease"),
        os.path.join(base, "usr/local/bin/rna_denovo.linuxgccrelease"),
    ]
    ff = None
    for cand in ff_candidates:
        if os.path.exists(cand):
            ff = cand
            break
    if ff is None:
        import glob
        matches = glob.glob(os.path.join(base, "main/source/bin", "rna_denovo*linux*release"))
        if matches:
            ff = matches[0]

    ex = os.path.join(base, "main/tools/rna_tools/silent_util/extract_lowscore_decoys.py")
    if not os.path.exists(ex):
        import glob
        matches = glob.glob(os.path.join(base, "main/tools/rna_tools/**/extract_lowscore_decoys.py"), recursive=True)
        if matches:
            ex = matches[0]
    return base, ff, ex

#MOD_VOCAB.mod2index acsiRNA_mod atom_mod

class Data_Prepare:
    def __init__(self, excel_dir, pdb_dir, rosetta_dir=None, workers=4):
        self.excel_dir=excel_dir
        self.pdb_dir=pdb_dir

        self.json_dir=excel_dir[:-4]+'.json'
        self.secondary_structure = True
        self.chunk_size=None
        self.num_cores = workers
        self.json = True

        if self.secondary_structure == False:
            self.json_dir=excel_dir[:-4]+'no2.json'

        self.rosetta_dir, self.ff, self.ex = resolve_rosetta_dir(rosetta_dir)
        if not self.ff or not os.path.exists(self.ff):
            raise FileNotFoundError(f"Rosetta rna_denovo not found under: {self.rosetta_dir}")
        if not self.ex or not os.path.exists(self.ex):
            raise FileNotFoundError("Rosetta extract_lowscore_decoys.py is required")

        self.database = os.path.join(self.rosetta_dir, "database")
        if not os.path.isdir(self.database):
            self.database = os.environ.get("ROSETTA_DATABASE")
        if not self.database or not os.path.isdir(self.database):
            raise FileNotFoundError("Rosetta database is required")


    def get_path(self,siRNA):
        return f"{self.pdb_dir}/{siRNA}.pdb"


    def raw_pre(self,seq):
        return seq.lower().replace(' + ','').replace('d','').replace('t','u').replace(' ','')

    def get_rpos(self,item):
        pos = [0]
        for i in range(1,1+len(item['sense seq'])):
            pos.append(i)
        for i in range(30,30+len(item['anti seq'])):
            pos.append(i)
        return pos

    def chunk_dataframe(self,df, chunk_size):
        chunks = []
        chunk_count = len(df) // chunk_size
        for i in range(chunk_count):
            chunks.append(df[i * chunk_size : (i + 1) * chunk_size])
        if len(df) % chunk_size != 0:
            chunks.append(df[chunk_count * chunk_size:])
        return chunks

    def get_anti_start(self,data):
        seq1=data['sense seq']
        seq2=data['anti seq']
        target_len = 61
        padlen = max(0, int((target_len - len(seq1)) / 2))

        try:
            raw_se = subprocess.run(["RNAplex"], input=f"{seq1}\n{seq2}\n",
                                    capture_output=True, text=True, check=True).stdout
            secondary_seq1 = re.split(r'\s+', raw_se)[0].split('&')[0]
            secondary_seq2 = re.split(r'\s+', raw_se)[0].split('&')[1]
            seq2_ = re.split(r'\s+', raw_se)[3].split(',')
            seq1_ = re.split(r'\s+', raw_se)[1].split(',')
            anti1=0
            anti2=0
            s1=0
            s2=0
            flag=0
            for i in  secondary_seq2:
                if i=='.':
                    anti1+=1
                    flag=1
                else:
                    if flag==1:
                        anti1+=len(seq2[:int(int(seq2_[0])-1)])
                    break    
            for i in  secondary_seq2[::-1]:
                if i=='.':
                    flag=-1
                    anti2-=1
                else:
                    if flag==-1:
                        anti2-=len(seq2[int(seq2_[1]):])
                    break  
            flag=0
            for i in  secondary_seq1:
                if i=='.':
                    flag=1
                    s1+=1
                else:
                    if flag==1:
                        s1+=len(seq1[:int(seq1_[0])-1])
                    break
            for i in  secondary_seq1[::-1]:
                if i=='.':
                    flag=-1
                    s2-=1
                else:
                    if flag==-1:
                        s2-=len(seq1[int(seq1_[1]):])
                    break
            seq2_ = re.split(r'\s+', raw_se)[3].split(',')
            sec_pos = [1000]
            chain = [0]
            for i in range(-padlen,len(seq1)+padlen): #mrna
                sec_pos.append(i)
                chain.append(1)
            sec_pos.append(2000)
            chain.append(2)
            for i in range(len(seq1)): #sense
                sec_pos.append(i)
                chain.append(3)
            for i in range(s2+anti1+len(seq1)-1,s1+anti2-1,-1): #anti
                sec_pos.append(i)
                chain.append(3)
            if len(sec_pos) != target_len+len(seq2)+len(seq1)+1+1:
                raise ValueError("RNAplex position metadata has the wrong length")
            return sec_pos,chain
        except (IndexError, ValueError) as error:
            raise ValueError(f"Invalid RNAplex output for {data['siRNA']}") from error

    def process(self):
        df=pd.read_csv(self.excel_dir) 
        df['sense seq']=df['sense seq'].apply(self.raw_pre)
        df['anti seq']=df['anti seq'].apply(self.raw_pre)


        df[['start', 'chain']] = df.apply(self.get_anti_start, axis=1, result_type='expand')

        total_rows = len(df)
        self.chunk_size = max(1, total_rows // self.num_cores)
        chunks = self.chunk_dataframe(df, self.chunk_size)
        with multiprocessing.Pool(processes=min(self.num_cores, len(chunks))) as pool:
            drops=pool.map(self.get_data, chunks)

        #print(drops)

        #df=df.drop(index=drops)
        if self.json==True:

            dfj=df[['siRNA','mRNA_seq','position','sense seq','anti seq','efficiency','start','chain']] #delect marker

            dfj['pdb_data_path']=dfj['siRNA'].apply(self.get_path)
            dfj['efficiency']=dfj['efficiency'].apply(lambda x: x if x > 0 else 0)
            if dfj.isna().any().any():
                raise ValueError('Incomplete PDB metadata; refusing to drop benchmark rows')
            dfj.to_json(self.json_dir, orient='records', lines=True)

    def get_data(self,df):
        
        drops=[]
 
        for index, row in df.iterrows():
            if os.path.exists(f"{self.pdb_dir}/{row['siRNA']}.pdb")==True:
                continue
            if self.secondary_structure ==True:
                if self.get_secondary_structure(row)==False:
                    print(f"drop{self.pdb_dir}/{row['siRNA']}.pdb")
                    drops.append(index)

            else:
                if self.get_structure(row)==False:
                    print(f"drop{self.pdb_dir}/{row['siRNA']}.pdb")
                    drops.append(index)

        return drops
    



    

    def get_structure(self,data):
    
        seq1=data['sense seq']
        seq2=data['anti seq']

        pose = assembler.build_init_pose(seq1, seq2)
        pose.dump_pdb(f"{self.pdb_dir}/{data['siRNA']}.pdb")
        if os.path.exists(f"{self.pdb_dir}/{data['siRNA']}.pdb"):
            return True
        else:
            return False

    def get_secondary_structure(self,data):
        seq1=data['sense seq']
        seq2=data['anti seq']
        seq=seq1+' '+seq2
        output = subprocess.run(["RNAplex"], input=f"{seq1}\n{seq2}\n",
                                capture_output=True, text=True, check=True).stdout
        fields = output.split()
        paired = fields[0].split('&')
        structures = []
        for sequence, structure, field in zip((seq1, seq2), paired, (fields[1], fields[3])):
            start, end = map(int, field.split(','))
            full = '.' * (start - 1) + structure + '.' * (len(sequence) - end)
            if len(full) != len(sequence):
                raise ValueError("RNAplex returned inconsistent secondary-structure coordinates")
            structures.append(full)
        secondary_seq = ' '.join(structures)

        # A failed generation never becomes a reusable .pdb cache entry.
        with tempfile.TemporaryDirectory(prefix=f"{data['siRNA']}_", dir=self.pdb_dir) as workdir:
            cmd = [self.ff, '-sequence', seq, '-secstruct', secondary_seq,
                   '-minimize_rna', '-out:file:silent', 'default.out',
                   '-database', self.database, '-constant_seed', '-jran', '0']
            subprocess.run(cmd, cwd=workdir, check=True)
            subprocess.run([sys.executable, self.ex, 'default.out', '-rosetta_folder',
                            self.rosetta_dir, '1'], cwd=workdir, check=True)
            pdb_path = os.path.join(workdir, 'default.out.1.pdb')
            with open(pdb_path) as handle:
                if not any(line.startswith('ATOM') for line in handle):
                    raise ValueError(f"Rosetta did not produce a PDB for {data['siRNA']}")
            os.replace(pdb_path, f"{self.pdb_dir}/{data['siRNA']}.pdb")
        return True

   

def parse():
    parser = argparse.ArgumentParser(description='Process data')
    parser.add_argument('-f','--filenames', nargs='+', help='train/valsiRNA/test set')
    parser.add_argument('-p','--pdb_dir', type=str, default=None, help='Path to save processed data')
    parser.add_argument('--rosetta-dir', type=str, default=None, help='Rosetta base directory')
    parser.add_argument('--workers', type=int, default=4)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse()
    for filename in args.filenames:
        Data_Prepare(filename, args.pdb_dir, rosetta_dir=args.rosetta_dir, workers=args.workers).process()
  

