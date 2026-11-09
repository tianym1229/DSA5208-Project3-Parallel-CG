#!/usr/bin/env python3
"""DSA5208 Project 3: six reproducible MPI strong-scaling figures."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FixedLocator, FuncFormatter

ROOT = Path('data/mpi/strong_scaling')
OUT = Path('results/analysis/mpi/strong_scaling_figures')
OUT.mkdir(parents=True, exist_ok=True)
RANKS = [1, 2, 4, 8, 16, 32, 64, 128, 256]
STYLES = {'baseline': dict(color='#205781', marker='o', label='Baseline'),
          'overlap': dict(color='#D87535', marker='s', label='Overlap')}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,
                     'axes.labelsize':11,'legend.fontsize':9,'savefig.dpi':300,
                     'axes.spines.top':False,'axes.spines.right':False,
                     'pdf.fonttype':42,'ps.fonttype':42})

def read_table(path):
    if not path.is_file():
        raise SystemExit(f'Missing input: {path}')
    with path.open(newline='') as f:
        return list(csv.DictReader(f))

perf = read_table(ROOT/'performance_summary.csv')
comp = read_table(ROOT/'communication_comparison.csv')
assert len(perf)==36, f'Expected 36 performance rows; got {len(perf)}'
assert len(comp)==18, f'Expected 18 comparison rows; got {len(comp)}'
P={}
for row in perf:
    key=(int(row['dim']),int(row['ranks']),row['comm'])
    if key in P: raise ValueError(f'Duplicate performance key {key}')
    P[key]=row
C={}
for row in comp:
    key=(int(row['dim']),int(row['ranks']))
    if key in C: raise ValueError(f'Duplicate comparison key {key}')
    C[key]=row
for dim in (2,3):
    for rank in RANKS:
        assert (dim,rank) in C
        for comm in STYLES: assert (dim,rank,comm) in P
        b=float(P[dim,rank,'baseline']['median_runtime_s'])
        o=float(P[dim,rank,'overlap']['median_runtime_s'])
        assert b>0 and o>0
        assert abs(float(C[dim,rank]['improvement_percent'])-(b-o)/b*100)<1e-5

def axis_ranks(ax):
    ax.set_xscale('log',base=2)
    ax.set_xticks(RANKS)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xlim(0.88,295)
    ax.set_xlabel('Number of MPI ranks')
    ax.grid(True,alpha=.22,linewidth=.7)

def save(fig,stem):
    fig.tight_layout(pad=1.5)
    for ext in ('png',):
        path=OUT/f'{stem}.{ext}'
        fig.savefig(path,dpi=300,bbox_inches='tight')
        print('CREATED:',path)
    plt.close(fig)

def values(dim,comm,field):
    return [float(P[dim,r,comm][field]) for r in RANKS]

for dim in (2,3):
    fig,ax=plt.subplots(figsize=(7.4,4.9))
    for comm,style in STYLES.items():
        ax.plot(RANKS,values(dim,comm,'speedup'),linewidth=2,markersize=5.5,**style)
    ax.plot(RANKS,RANKS,'--',color='#808080',linewidth=1.4,label='Ideal linear speedup')
    axis_ranks(ax)
    ax.set_yscale('log',base=2)
    ax.set_ylim(.8,320)
    ax.yaxis.set_major_locator(FixedLocator([1,2,4,8,16,32,64,128,256]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f'{int(y)}'))
    ax.set_ylabel('Speedup (relative to 1 rank)')
    ax.set_title(f'{dim}D MPI Strong Scaling — Speedup')
    ax.legend(loc='upper left',frameon=False)
    save(fig,f'fig0{1 if dim==2 else 2}_{dim}d_speedup')

for dim in (2,3):
    fig,ax=plt.subplots(figsize=(7.4,4.9))
    for comm,style in STYLES.items():
        ax.plot(RANKS,values(dim,comm,'efficiency_percent'),linewidth=2,markersize=5.5,**style)
    ax.axhline(100,color='#808080',ls='--',lw=1.4,label='Ideal efficiency (100%)')
    axis_ranks(ax)
    ax.set_ylim(0,112)
    ax.set_ylabel('Parallel efficiency (%)')
    ax.set_title(f'{dim}D MPI Strong Scaling — Efficiency')
    ax.legend(loc='upper right',frameon=False)
    save(fig,f'fig0{3 if dim==2 else 4}_{dim}d_efficiency')

fig,axes=plt.subplots(1,2,figsize=(11.6,4.8),sharey=False)
for dim,ax in zip((2,3),axes):
    for comm,style in STYLES.items():
        ax.plot(RANKS,values(dim,comm,'median_runtime_s'),linewidth=2,markersize=5.2,**style)
    axis_ranks(ax)
    ax.set_yscale('log')
    ax.set_ylabel('Median runtime (s)')
    ax.set_title(f'{dim}D, grid={512 if dim==2 else 64}')
    best_rank = min(RANKS, key=lambda r: float(P[dim,r,'baseline']['median_runtime_s']))
    best_time = float(P[dim,best_rank,'baseline']['median_runtime_s'])
    ax.annotate(f'Best baseline: {best_time:.4f} s\n({best_rank} ranks)',
                xy=(best_rank,best_time), xytext=(20,22), textcoords='offset points',
                fontsize=8.5, color=STYLES['baseline']['color'],
                arrowprops=dict(arrowstyle='->', lw=0.9, color=STYLES['baseline']['color']),
                bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='none', alpha=0.85))
    ax.legend(frameon=False,loc='upper right')
fig.suptitle('MPI Strong Scaling — Baseline vs Overlap Runtime',fontsize=13)
fig.text(.5,.01,'Each panel uses its own logarithmic runtime axis.',ha='center',fontsize=8.5,color='#555555')
fig.subplots_adjust(bottom=0.14)
save(fig,'fig05_baseline_vs_overlap_runtime')

fig,axes=plt.subplots(1,2,figsize=(11.6,4.8),sharey=True)
for dim,ax in zip((2,3),axes):
    imp=[float(C[dim,r]['improvement_percent']) for r in RANKS]
    ax.plot(RANKS,imp,color='#205781',marker='o',lw=2,markersize=5.5)
    ax.axhline(0,color='#777777',ls='--',lw=1.2)
    axis_ranks(ax)
    ax.set_title(f'{dim}D, grid={512 if dim==2 else 64}')
    ax.set_ylabel('Overlap improvement (%)')
fig.suptitle('MPI Communication Mode — Relative Runtime Improvement',fontsize=13)
fig.text(.5,.005,'Improvement = (baseline median - overlap median) / baseline median × 100%; positive = overlap faster',ha='center',fontsize=8.5)
save(fig,'fig06_communication_mode_improvement')

files=list(OUT.glob('fig0*.png'))
assert len(files)==6,f'Expected 12 figure files; got {len(files)}'
print('===== ALL SIX FIGURES COMPLETED: 12 FILES =====')
print('NOTE: original C44 32-rank records remain in primary summary; C467 is independent validation.')
