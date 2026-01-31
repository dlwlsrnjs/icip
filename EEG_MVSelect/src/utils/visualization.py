"""
유틸리티 함수 모음
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def visualize_connectivity_selection(model, dataset, sample_idx=0, steps=2, save_path=None):
    """
    연결성 이미지 선택 과정 시각화
    
    Args:
        model: EEG-MVSelect 모델
        dataset: EEGConnectivityDataset
        sample_idx: 샘플 인덱스
        steps: 선택 스텝 수
        save_path: 저장 경로
    """
    model.eval()
    
    # 샘플 로드
    imgs, label, keep_cams = dataset[sample_idx]
    imgs = imgs.unsqueeze(0).cuda()
    keep_cams = keep_cams.unsqueeze(0).cuda()
    
    # 초기 뷰
    init_cam = 0
    init_prob = torch.nn.functional.one_hot(
        torch.tensor([init_cam]), num_classes=dataset.num_cam
    ).cuda()
    
    # Forward
    with torch.no_grad():
        feat, _ = model.get_feat(imgs, down=1)
        _, (_, _, actions, _) = model.do_steps(feat, init_prob, steps, keep_cams)
    
    # 선택된 뷰
    selected_views = [init_cam]
    for action in actions:
        selected_views.append(action.argmax(dim=-1).item())
    
    # 시각화
    fig, axes = plt.subplots(1, steps + 2, figsize=(4 * (steps + 2), 4))
    
    for i, view_idx in enumerate(selected_views):
        ax = axes[i]
        
        # 이미지 표시
        img = imgs[0, view_idx].cpu()
        img = img.permute(1, 2, 0)
        # Denormalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = img * std + mean
        img = np.clip(img, 0, 1)
        
        ax.imshow(img)
        ax.set_title(f'Step {i}: {dataset.get_view_name(view_idx)}', fontsize=10)
        ax.axis('off')
    
    # 마지막 칸: 정보
    ax = axes[-1]
    ax.axis('off')
    info_text = f'Sample: {sample_idx}\n'
    info_text += f'Label: {dataset.subjects[label]}\n\n'
    info_text += 'Selected Views:\n'
    for i, view_idx in enumerate(selected_views):
        info_text += f'{i}. {dataset.get_view_name(view_idx)}\n'
    ax.text(0.1, 0.5, info_text, fontsize=10, verticalalignment='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Visualization saved to {save_path}')
    
    plt.show()
    plt.close()


def analyze_view_selection_statistics(model, dataloader, steps=2, save_dir=None):
    """
    뷰 선택 통계 분석
    
    Args:
        model: EEG-MVSelect 모델
        dataloader: 데이터 로더
        steps: 선택 스텝 수
        save_dir: 저장 디렉토리
    """
    model.eval()
    
    N = model.num_cam
    
    # 통계 수집
    view_count = torch.zeros(N)
    view_accuracy = torch.zeros(N)
    view_total = torch.zeros(N)
    
    with torch.no_grad():
        for imgs, labels, keep_cams in dataloader:
            B = imgs.shape[0]
            imgs = imgs.cuda()
            labels = labels.cuda()
            keep_cams = keep_cams.cuda()
            
            # 각 뷰를 초기 뷰로 테스트
            for init_cam in range(N):
                feat, _ = model.get_feat(imgs, down=1)
                init_prob = torch.nn.functional.one_hot(
                    torch.tensor([init_cam]).repeat(B), num_classes=N
                ).cuda()
                
                overall_feat, (_, _, actions, _) = model.do_steps(
                    feat, init_prob, steps, keep_cams
                )
                output = model.get_output(overall_feat)
                
                # 정확도
                pred = output.argmax(dim=1)
                correct = (pred == labels).float()
                
                view_accuracy[init_cam] += correct.sum().item()
                view_total[init_cam] += B
                
                # 선택 빈도
                for action in actions:
                    view_count += action.sum(dim=0).cpu()
    
    # 결과
    view_freq = view_count / view_count.sum()
    view_acc = view_accuracy / view_total * 100
    
    # 출력
    print('\n' + '='*70)
    print('View Selection Statistics')
    print('='*70)
    
    dataset = dataloader.dataset
    for view_idx in range(N):
        view_name = dataset.get_view_name(view_idx)
        print(f'{view_idx:2d}. {view_name:20s} | '
              f'Freq: {view_freq[view_idx]*100:5.2f}% | '
              f'Acc: {view_acc[view_idx]:5.2f}%')
    
    # 시각화
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 선택 빈도
    view_names = [dataset.get_view_name(i) for i in range(N)]
    ax1.bar(range(N), view_freq * 100)
    ax1.set_xlabel('View Index')
    ax1.set_ylabel('Selection Frequency (%)')
    ax1.set_title('View Selection Frequency')
    ax1.set_xticks(range(N))
    ax1.set_xticklabels(view_names, rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3)
    
    # 정확도
    ax2.bar(range(N), view_acc)
    ax2.set_xlabel('View Index')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Accuracy by Initial View')
    ax2.set_xticks(range(N))
    ax2.set_xticklabels(view_names, rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        save_path = Path(save_dir) / 'view_selection_stats.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'\nStatistics saved to {save_path}')
    
    plt.show()
    plt.close()
    
    return view_freq, view_acc


def count_parameters(model):
    """모델 파라미터 수 계산"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f'\nModel Parameters:')
    print(f'  Total: {total_params:,}')
    print(f'  Trainable: {trainable_params:,}')
    
    # 모듈별 파라미터
    print(f'\nParameters by Module:')
    for name, module in model.named_children():
        module_params = sum(p.numel() for p in module.parameters())
        print(f'  {name}: {module_params:,}')
    
    return total_params, trainable_params


def compute_flops(model, input_shape=(1, 15, 3, 224, 224)):
    """
    FLOPs 계산 (간략화 버전)
    정확한 계산은 thop 또는 fvcore 라이브러리 사용 권장
    """
    try:
        from thop import profile
        dummy_input = torch.randn(input_shape).cuda()
        flops, params = profile(model, inputs=(dummy_input,), verbose=False)
        print(f'\nComputational Cost:')
        print(f'  FLOPs: {flops / 1e9:.2f} G')
        print(f'  Parameters: {params / 1e6:.2f} M')
        return flops, params
    except ImportError:
        print('\nthop not installed. Install with: pip install thop')
        return None, None


if __name__ == '__main__':
    # 예제
    from src.datasets import EEGConnectivityDataset
    from src.models import EEGMVSelect
    
    # 데이터셋 및 모델
    dataset = EEGConnectivityDataset(
        root_dir='/home/work/skku/icip/FCM_Images_HERMES_v2',
        split='test'
    )
    
    model = EEGMVSelect(dataset, arch='resnet18', aggregation='max')
    model = model.cuda()
    
    # 파라미터 수
    count_parameters(model)
    
    print('\nUtils test completed!')
