# Retrieving sequences from fasta files

from Bio import SeqIO

siRNA_dict = {record.id: str(record.seq) for record in SeqIO.parse("C:/Users/user/Desktop/iGEM25-ML/Simone Dataset/sirna_Simone.fas", "fasta")}
mRNA_dict = {record.id: str(record.seq) for record in SeqIO.parse("C:/Users/user/Desktop/iGEM25-ML/Simone Dataset/mRNA_Simone.fas", "fasta")}


# Retrieving efficacy values from csv file

import csv

siRNA_list = []
mRNA_list = []
eff_dict = {}

with open("C:/Users/user/Desktop/iGEM25-ML/Simone Dataset/Simone_efficacy.csv", "r") as file:
    reader = csv.DictReader(file)  
    for row in reader:
        si = row["siRNA"]  
        mR = row["mRNA"]
        eff = row["efficacy"]
        siRNA_list.append(si)
        mRNA_list.append(mR)
        eff_dict[si] = eff


# Creating the csv file with the whole mRNA sequence

with open("simone_dataset_whole_mRNA.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["siRNA", "mRNA", "efficacy"])

    for i in range(len(siRNA_list)):
        cur_si = siRNA_list[i]
        cur_mR = mRNA_list[i]
        writer.writerow([cur_si, mRNA_dict[str(cur_mR)], eff_dict[str(cur_si)]])


# Creating the final csv file containing only the target mRNA subsequence + extended sequence on each side

# mRNA has 'T' instead of 'U'

def complement_base(base):
    if base == 'A':
        return 'T'
    if base == 'U':
        return 'A'
    if base == 'G':
        return 'C'
    return 'G'


def find_rev_complement(sequence):
    rev_compl = ""
    for base in sequence:
        rev_compl += complement_base(str(base))
    rev_compl = rev_compl[::-1]
    return rev_compl

target_mRNA = []

with open("C:/Users/user/Desktop/iGEM25-ML/Simone Dataset/simone_dataset_whole_mRNA.csv", "r") as file:
    reader = csv.DictReader(file)  
    for row in reader:
        si = str(row["siRNA"])  
        si = str(siRNA_dict[si])
        mR = str(row["mRNA"])
        eff = str(row["efficacy"])
        len_si = len(si)
        target_seq = find_rev_complement(str(si))
        for i in range(len(mR)-len_si+1):
            cnt = 0
            while cnt < len_si and target_seq[cnt] == mR[i+cnt]:
                cnt += 1
            if cnt == len_si:
                targ = ''
                for j in range(i-len_si,i+2*len_si):
                    if j>=0 and j<len(mR):
                        targ += mR[j]
                target_mRNA.append(targ)
                break

with open("simone_dataset_final.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["siRNA", "mRNA", "efficacy"])

    for i in range(len(siRNA_list)):
        cur_si = siRNA_list[i]
        writer.writerow([siRNA_dict[str(cur_si)], target_mRNA[i], eff_dict[str(cur_si)]])