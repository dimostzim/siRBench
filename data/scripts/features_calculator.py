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

    step_dg = []
    step_dh = []
    for i in range(18):
        dinuc = sirna_seq[i:i+2]
        dg = DELTA_G.get(dinuc, 0.0)
        dh = DELTA_H.get(dinuc, 0.0)
        total_dg += dg
        total_dh += dh
        step_dg.append(dg)
        step_dh.append(dh)

    return total_dg, total_dh, step_dg, step_dh

def calculate_oligoformer_features_exact(sirna_seq):
    seq = sirna_seq.upper()
    features = []
    
    features.append(calculate_end_diff(seq))
    total_dg, total_dh, step_dg, step_dh = calculate_dgh(seq)
    features.append(total_dg)
    features.append(total_dh)
    features.extend(step_dg)
    features.extend(step_dh)
    
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
    names = ['ends', 'DG_total', 'DH_total']
    names += [f'DG_pos{i}' for i in range(1, 19)]
    names += [f'DH_pos{i}' for i in range(1, 19)]
    return names
