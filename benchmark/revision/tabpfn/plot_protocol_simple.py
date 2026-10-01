"""Draw the frozen shared evaluation design; no model outcomes are used."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyBboxPatch, Patch
import numpy as np
import pandas as pd


def box(axis, x, y, width, height, text, color):
    axis.add_patch(FancyBboxPatch((x,y),width,height,boxstyle='round,pad=0.012',
                                 facecolor=color,edgecolor='#66727c',linewidth=.7))
    axis.text(x+width/2,y+height/2,text,ha='center',va='center',fontsize=12)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Choose a new, empty figure directory')
    membership = pd.read_csv(args.protocol/'membership.csv')
    groups = pd.read_csv(args.protocol/'groups.csv').set_index('record_id')
    hela = pd.read_csv(args.protocol/'hela_full.csv',usecols=['record_id'])
    primary = membership[membership.axis.eq('grouped')]
    tests = primary[primary.part.eq('test')].sort_values(['fold','record_id'])
    if not tests.record_id.is_unique or set(tests.record_id)&set(hela.record_id):
        raise ValueError('Test membership must be unique and disjoint from HeLa')
    tests = tests.join(groups,on='record_id').sort_values(['fold','split_group_id','record_id'])
    ids = tests.record_id.tolist()
    colors = ['#b7c8d5','#e4ba63','#426b94']
    codes = {'train':0,'val':1,'test':2}
    matrix, counts = [], []
    for fold in range(5):
        frame = primary[primary.fold.eq(fold)].set_index('record_id').loc[ids]
        if not frame.index.is_unique or len(frame)!=len(ids):
            raise ValueError('Each fold must assign every non-HeLa record once')
        matrix.append(frame.part.map(codes).to_numpy())
        counts.append(frame.part.value_counts().to_dict())
    if not np.isfinite(matrix).all():
        raise ValueError('Unknown partition label')
    plt.rcParams.update({'font.size':11,'font.family':'DejaVu Sans',
                         'pdf.fonttype':42,'svg.fonttype':'none'})
    fig = plt.figure(figsize=(7.8,4.3),layout='constrained')
    grid = fig.add_gridspec(2,1,height_ratios=[1.05,2.1])
    top = fig.add_subplot(grid[0]); top.set(xlim=(0,1),ylim=(0,1)); top.axis('off')
    top.text(0,1,'A  Evaluation cohorts',weight='bold',va='top',fontsize=13)
    box(top,.02,.10,.43,.61,f'Non-HeLa · {len(ids):,} records\nGrouped evaluation','#eaf0f5')
    box(top,.54,.10,.43,.61,f'HeLa · {len(hela):,} records\nTransfer evaluation','#f5ead3')
    middle = fig.add_subplot(grid[1])
    middle.imshow(np.array(matrix),aspect='auto',interpolation='nearest',
                  cmap=ListedColormap(colors),vmin=0,vmax=2)
    middle.set_title('B  Shared grouped folds',loc='left',weight='bold',fontsize=13,pad=16)
    middle.set_yticks(range(5),[f'Fold {i}' for i in range(5)])
    middle.set_xticks([])
    middle.set_xlabel(f'{len(ids):,} non-HeLa records',labelpad=7)
    middle.tick_params(axis='y',length=0)
    for spine in middle.spines.values(): spine.set_visible(False)
    for boundary in tests.groupby('fold').size().cumsum().iloc[:-1]:
        middle.axvline(boundary-.5,color='white',linewidth=.8)
    for fold, sizes in enumerate(counts):
        middle.text(1.012,fold,f"{sizes['train']:,} / {sizes['val']:,} / {sizes['test']:,}",
                    transform=middle.get_yaxis_transform(),va='center',fontsize=10.5)
    middle.text(1.012,-.9,'Train / val / test',transform=middle.get_yaxis_transform(),fontsize=10.5)
    middle.legend(handles=[Patch(facecolor=color,label=label) for color,label in zip(colors,['Training','Validation','Test'])],
                  loc='upper center',bbox_to_anchor=(.5,-.19),ncol=3,frameon=False)
    args.output.mkdir(parents=True,exist_ok=True)
    for suffix in ['pdf','svg','png']:
        fig.savefig(args.output/f'evaluation_protocol.{suffix}',dpi=300,bbox_inches='tight')
    plt.close(fig)
    paths = [args.protocol/name for name in ['membership.csv','groups.csv','hela_full.csv','manifest.json']]
    report = {'nonhela_records':len(ids),'hela_records':len(hela),'fold_counts':counts,
              'description':'Shared main evaluation inputs and actual frozen fold assignments. No performance results shown.',
              'compact':True,'matplotlib':matplotlib.__version__,
              'input_sha256':{str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in [*paths,Path(__file__)]}}
    (args.output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__ == '__main__':
    main()
