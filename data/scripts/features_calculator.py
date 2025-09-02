import subprocess

DELTA_G = {
    'AA': -0.93, 'UU': -0.93, 'AU': -1.10, 'UA': -1.33, 
    'CU': -2.08, 'AG': -2.08, 'CA': -2.11, 'UG': -2.11, 
    'GU': -2.24, 'AC': -2.24, 'GA': -2.35, 'UC': -2.35, 
    'CG': -2.36, 'GG': -3.26, 'CC': -3.26, 'GC': -3.42, 
    'init': 4.09, 'endAU': 0.45, 'sym': 0.43
}

DELTA_H = {
    'AA': -6.82, 'UU': -6.82, 'AU': -9.38, 'UA': -7.69, 
    'CU': -10.48, 'AG': -10.48, 'CA': -10.44, 'UG': -10.44, 
    'GU': -11.40, 'AC': -11.40, 'GA': -12.44, 'UC': -12.44, 
    'CG': -10.64, 'GG': -13.39, 'CC': -13.39, 'GC': -14.88, 
    'init': 3.61, 'endAU': 3.72, 'sym': 0
}

def calculate_end_diff(sirna_seq):
    count = 0
    five_prime = sirna_seq[:2]
    three_prime = sirna_seq[-2:]
    if five_prime in ['AC', 'AG', 'UC', 'UG']:
        count += 1
    elif five_prime in ['GA', 'GU', 'CA', 'CU']:
        count -= 1
    
    if three_prime in ['AC', 'AG', 'UC', 'UG']:
        count += 1
    elif three_prime in ['GA', 'GU', 'CA', 'CU']:
        count -= 1
    
    ends_diff = DELTA_G.get(five_prime, 0.0) - DELTA_G.get(three_prime, 0.0) + count * 0.45
    return float('{:.2f}'.format(ends_diff))

def calculate_dgh(sirna_seq):
    total_dg = DELTA_G['init']
    total_dh = DELTA_H['init']
    
    if sirna_seq[0] in ['A', 'U']:
        total_dg += DELTA_G['endAU']
        total_dh += DELTA_H['endAU']
    if sirna_seq[18] in ['A', 'U']:
        total_dg += DELTA_G['endAU']
        total_dh += DELTA_H['endAU']
    
    if sirna_seq == sirna_seq[::-1]:
        total_dg += DELTA_G['sym']
        total_dh += DELTA_H['sym']
    
    for i in range(18):
        dinuc = sirna_seq[i:i+2]
        total_dg += DELTA_G.get(dinuc, 0.0)
        total_dh += DELTA_H.get(dinuc, 0.0)
    
    return total_dg, total_dh

def calculate_oligoformer_features_exact(sirna_seq):
    seq = sirna_seq.upper()
    features = []
    
    features.append(calculate_end_diff(seq))
    features.append(DELTA_G.get(seq[0:2], 0.0))
    features.append(DELTA_H.get(seq[0:2], 0.0))
    features.append(float(seq[0] == 'U'))
    features.append(float(seq[0] == 'G'))
    _, total_dh = calculate_dgh(seq)
    features.append(total_dh)
    features.append(seq.count('U') / 19)
    features.append(float(seq[0:2] == 'UU'))
    features.append(seq.count('G') / 19)
    features.append(float(seq[0:2] == 'GG'))
    features.append(float(seq[0:2] == 'GC'))
    gg_count = sum(1 for i in range(18) if seq[i:i+2] == 'GG')
    features.append(gg_count / 18)
    features.append(DELTA_G.get(seq[1:3], 0.0))
    ua_count = sum(1 for i in range(18) if seq[i:i+2] == 'UA')
    features.append(ua_count / 18)
    features.append(float(seq[1] == 'U'))
    features.append(float(seq[0] == 'C'))
    cc_count = sum(1 for i in range(18) if seq[i:i+2] == 'CC')
    features.append(cc_count / 18)
    features.append(DELTA_G.get(seq[17:19], 0.0))
    features.append(float(seq[0:2] == 'CC'))
    gc_count = sum(1 for i in range(18) if seq[i:i+2] == 'GC')
    features.append(gc_count / 18)
    features.append(float(seq[0:2] == 'CG'))
    features.append(DELTA_G.get(seq[12:14], 0.0))
    uu_count = sum(1 for i in range(18) if seq[i:i+2] == 'UU')
    features.append(uu_count / 18)
    features.append(float(seq[18] == 'A'))
    
    return features

def calculate_duplex_folding_energy(sirna_seq, target_seq):
    combined_seq = f"{sirna_seq}&{target_seq}"
    
    try:
        result = subprocess.run(['RNAcofold', '--noPS'], 
                              input=combined_seq, 
                              text=True, 
                              capture_output=True)
        
        for line in result.stdout.strip().split('\n'):
            if '(' in line and ')' in line and '&' in line:
                start = line.rfind('(')
                end = line.rfind(')')
                if start != -1 and end != -1 and start < end:
                    energy_str = line[start+1:end].strip()
                    return float(energy_str)
    except Exception as e:
        print(f"Error calculating duplex energy: {e}")
        return None
    
    return None

def get_feature_names():
    return [
        'ends', 'DG_1', 'DH_1', 'U_1', 'G_1', 'DH_all', 'U_all', 'UU_1', 
        'G_all', 'GG_1', 'GC_1', 'GG_all', 'DG_2', 'UA_all', 'U_2', 'C_1', 
        'CC_all', 'DG_18', 'CC_1', 'GC_all', 'CG_1', 'DG_13', 'UU_all', 'A_19'
    ]