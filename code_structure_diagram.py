import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

def draw_code_structure():
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # 색상 정의
    colors = {
        'main': '#E8F4FD',
        'model': '#FFE6E6', 
        'dataset': '#E6FFE6',
        'training': '#FFF2E6',
        'evaluation': '#F0E6FF'
    }
    
    # 메인 파일들
    main_files = [
        ('main.py', 1, 10, colors['main']),
        ('REIN.py', 1, 8, colors['model']),
        ('nclt_dataset_3ch.py', 1, 6, colors['dataset']),
        ('kitti_dataset.py', 1, 4, colors['dataset'])
    ]
    
    # 메인 파일들 그리기
    for name, x, y, color in main_files:
        box = FancyBboxPatch((x, y), 2, 0.8, 
                           boxstyle="round,pad=0.1", 
                           facecolor=color, 
                           edgecolor='black', 
                           linewidth=1.5)
        ax.add_patch(box)
        ax.text(x+1, y+0.4, name, ha='center', va='center', fontsize=10, fontweight='bold')
    
    # main.py 내부 구조
    main_components = [
        ('get_args()', 4, 10.5, colors['main']),
        ('train_epoch()', 4, 9.8, colors['training']),
        ('infer()', 4, 9.1, colors['evaluation']),
        ('getClusters()', 4, 8.4, colors['training']),
        ('saveCheckpoint()', 4, 7.7, colors['main'])
    ]
    
    for name, x, y, color in main_components:
        box = FancyBboxPatch((x, y), 2.5, 0.5, 
                           boxstyle="round,pad=0.05", 
                           facecolor=color, 
                           edgecolor='black', 
                           linewidth=1)
        ax.add_patch(box)
        ax.text(x+1.25, y+0.25, name, ha='center', va='center', fontsize=8)
    
    # REIN.py 내부 구조
    rein_components = [
        ('REIN', 4, 8, colors['model']),
        ('REM', 4, 7.3, colors['model']),
        ('NetVLAD', 4, 6.6, colors['model'])
    ]
    
    for name, x, y, color in rein_components:
        box = FancyBboxPatch((x, y), 2.5, 0.5, 
                           boxstyle="round,pad=0.05", 
                           facecolor=color, 
                           edgecolor='black', 
                           linewidth=1)
        ax.add_patch(box)
        ax.text(x+1.25, y+0.25, name, ha='center', va='center', fontsize=8)
    
    # Dataset 구조
    dataset_components = [
        ('TrainingDataset', 4, 6, colors['dataset']),
        ('InferDataset', 4, 5.3, colors['dataset']),
        ('evaluateResults()', 4, 4.6, colors['evaluation']),
        ('collate_fn()', 4, 3.9, colors['dataset'])
    ]
    
    for name, x, y, color in dataset_components:
        box = FancyBboxPatch((x, y), 2.5, 0.5, 
                           boxstyle="round,pad=0.05", 
                           facecolor=color, 
                           edgecolor='black', 
                           linewidth=1)
        ax.add_patch(box)
        ax.text(x+1.25, y+0.25, name, ha='center', va='center', fontsize=8)
    
    # 데이터 흐름
    data_flow = [
        ('NCLT/KITTI\nDatasets', 7, 10, colors['dataset']),
        ('BEV Images', 7, 9, colors['dataset']),
        ('Feature\nExtraction', 7, 8, colors['model']),
        ('Global\nDescriptors', 7, 7, colors['model']),
        ('Recall\nEvaluation', 7, 6, colors['evaluation'])
    ]
    
    for name, x, y, color in data_flow:
        box = FancyBboxPatch((x, y), 2, 0.6, 
                           boxstyle="round,pad=0.05", 
                           facecolor=color, 
                           edgecolor='black', 
                           linewidth=1)
        ax.add_patch(box)
        ax.text(x+1, y+0.3, name, ha='center', va='center', fontsize=8)
    
    # 연결선 그리기
    connections = [
        # main.py -> REIN.py
        ((3, 10.4), (3, 8.4)),
        # main.py -> dataset
        ((3, 10.4), (3, 6.4)),
        # 데이터 흐름
        ((7, 9.6), (7, 9)),
        ((7, 9), (7, 8.6)),
        ((7, 8), (7, 7.6)),
        ((7, 7), (7, 6.6)),
        # 메인 -> 데이터 흐름
        ((3, 10.4), (7, 10.4))
    ]
    
    for start, end in connections:
        arrow = ConnectionPatch(start, end, "data", "data",
                              arrowstyle="->", shrinkA=5, shrinkB=5,
                              mutation_scale=20, fc="black", linewidth=1.5)
        ax.add_patch(arrow)
    
    # 제목과 범례
    ax.text(5, 11.5, 'BEVPlace++ Code Structure', ha='center', va='center', 
           fontsize=16, fontweight='bold')
    
    # 범례
    legend_elements = [
        patches.Patch(color=colors['main'], label='Main Functions'),
        patches.Patch(color=colors['model'], label='Model Components'),
        patches.Patch(color=colors['dataset'], label='Dataset & Data Loading'),
        patches.Patch(color=colors['training'], label='Training Functions'),
        patches.Patch(color=colors['evaluation'], label='Evaluation Functions')
    ]
    
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    plt.tight_layout()
    plt.savefig('bevplace_code_structure.jpg', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("코드 구조 다이어그램이 'bevplace_code_structure.jpg'로 저장되었습니다!")

if __name__ == "__main__":
    draw_code_structure() 