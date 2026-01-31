"""
EEG Multi-View Selection 메인 학습 스크립트
"""

import os
import sys
import argparse
import random
import datetime
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.datasets import EEGConnectivityDataset
from src.models import (EEGMVSelect, create_eeg_mvselect_model,
                        AttentionEEGMVSelect, create_attention_eeg_mvselect_model,
                        LightweightAttentionCNN, create_lightweight_attention_model,
                        PaperExactAttCNN, create_paper_exact_model)
from src.trainer import EEGTrainer, create_optimizer, create_scheduler


def parse_args():
    parser = argparse.ArgumentParser(description='EEG Multi-View Selection')
    
    # 데이터셋
    parser.add_argument('--data_root', type=str, 
                       default='/home/work/skku/icip/FCM_Images_HERMES_v2',
                       help='데이터 루트 디렉토리')
    parser.add_argument('--connectivity_methods', type=str, nargs='+',
                       default=None, help='사용할 연결성 방법 (FCM, PCC, PLV)')
    parser.add_argument('--frequency_bands', type=str, nargs='+',
                       default=None, help='사용할 주파수 밴드')
    parser.add_argument('--image_size', type=int, default=224,
                       help='입력 이미지 크기')
    
    # 모델
    parser.add_argument('--arch', type=str, default='resnet18',
                       choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'vgg11', 'vgg16'],
                       help='백본 아키텍처')
    parser.add_argument('--aggregation', type=str, default='max',
                       choices=['max', 'mean'], help='뷰 통합 방법')
    parser.add_argument('--pretrained', action='store_true', default=True,
                       help='사전학습 가중치 사용')
    parser.add_argument('--down', type=int, default=1,
                       help='입력 다운샘플링 비율')
    
    # Attention 관련
    parser.add_argument('--use_attention', action='store_true',
                       help='Attention 강화 모델 사용 (CBAM + MV Attention + VS Attention)')
    parser.add_argument('--use_lightweight', action='store_true',
                       help='논문 스타일 경량 모델 사용 (Channel Attention만, 3-layer CNN)')
    parser.add_argument('--use_paper_exact', action='store_true',
                       help='Neuroinformatics 2024 논문 철학 따른 모델 (224×224 input, 논문 구조)')
    parser.add_argument('--use_cbam', action='store_true', default=True,
                       help='CBAM (Channel + Spatial Attention) 사용')
    parser.add_argument('--use_mv_attention', action='store_true', default=True,
                       help='Multi-View Attention 사용')
    parser.add_argument('--use_vs_attention', action='store_true', default=True,
                       help='View Selection Attention 사용')
    
    # Cross-Validation
    parser.add_argument('--use_cv', action='store_true',
                       help='5-Fold Cross-Validation 사용')
    parser.add_argument('--cv_fold', type=int, default=0,
                       help='현재 CV fold 번호 (0~4)')
    parser.add_argument('--n_folds', type=int, default=5,
                       help='총 fold 개수')
    
    # MVSelect
    parser.add_argument('--steps', type=int, default=0,
                       help='선택할 뷰의 개수 (0이면 모든 뷰 사용)')
    parser.add_argument('--init_cam', type=str, default='random',
                       choices=['random', 'first'], help='초기 뷰 선택 방법')
    
    # 학습
    parser.add_argument('--epochs', type=int, default=50,
                       help='학습 에폭 수')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='배치 크기')
    parser.add_argument('--lr', type=float, default=5e-5,
                       help='학습률')
    parser.add_argument('--select_lr', type=float, default=1e-4,
                       help='MVSelect 학습률')
    parser.add_argument('--base_lr_ratio', type=float, default=1.0,
                       help='Base 네트워크 학습률 비율')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='가중치 감쇠')
    parser.add_argument('--scheduler', type=str, default='cosine',
                       choices=['cosine', 'step', 'onecycle', 'none'],
                       help='학습률 스케줄러')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                       help='그래디언트 클리핑 (0이면 사용 안 함)')
    
    # 강화학습
    parser.add_argument('--rl_loss_weight', type=float, default=0.1,
                       help='강화학습 손실 가중치')
    
    # 기타
    parser.add_argument('--num_workers', type=int, default=4,
                       help='데이터 로더 워커 수')
    parser.add_argument('--seed', type=int, default=42,
                       help='랜덤 시드')
    parser.add_argument('--log_interval', type=int, default=50,
                       help='로그 출력 간격')
    parser.add_argument('--eval', action='store_true',
                       help='평가 모드')
    parser.add_argument('--resume', type=str, default=None,
                       help='체크포인트 경로')
    parser.add_argument('--save_dir', type=str, default='logs',
                       help='저장 디렉토리')
    
    args = parser.parse_args()
    return args


def set_seed(seed):
    """재현성을 위한 시드 설정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    args = parse_args()
    
    # 시드 설정
    set_seed(args.seed)
    
    # 저장 디렉토리 생성
    if not args.eval:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        exp_name = f'{args.arch}_{args.aggregation}_steps{args.steps}_' \
                   f'lr{args.lr}_bs{args.batch_size}_{timestamp}'
        save_dir = os.path.join(args.save_dir, exp_name)
        os.makedirs(save_dir, exist_ok=True)
        
        # 로그 파일
        log_file = open(os.path.join(save_dir, 'log.txt'), 'w')
        sys.stdout = Logger(log_file)
    
    print('='*70)
    print('EEG Multi-View Selection')
    print('='*70)
    print('\nArguments:')
    for arg, value in sorted(vars(args).items()):
        print(f'  {arg}: {value}')
    print()
    
    # 데이터셋
    print('Loading datasets...')
    
    # 모든 모델은 224×224 이미지 사용
    image_size = args.image_size
    
    train_dataset = EEGConnectivityDataset(
        root_dir=args.data_root,
        split='train',
        connectivity_methods=args.connectivity_methods,
        frequency_bands=args.frequency_bands,
        image_size=image_size,
        use_cv=args.use_cv,
        cv_fold=args.cv_fold,
        n_folds=args.n_folds
    )
    
    val_dataset = EEGConnectivityDataset(
        root_dir=args.data_root,
        split='val',
        connectivity_methods=args.connectivity_methods,
        frequency_bands=args.frequency_bands,
        image_size=image_size,
        use_cv=args.use_cv,
        cv_fold=args.cv_fold,
        n_folds=args.n_folds
    )
    
    test_dataset = EEGConnectivityDataset(
        root_dir=args.data_root,
        split='test',
        connectivity_methods=args.connectivity_methods,
        frequency_bands=args.frequency_bands,
        image_size=image_size,
        use_cv=args.use_cv,
        cv_fold=args.cv_fold,
        n_folds=args.n_folds
    )
    
    # 데이터 로더
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    
    # 모델
    print('\nCreating model...')
    if args.use_paper_exact:
        print('Using Paper Exact Model (Neuroinformatics 2024)')
        model = create_paper_exact_model(
            dataset=train_dataset
        )
    elif args.use_lightweight:
        print('Using Lightweight Attention Model (논문 스타일)')
        model = create_lightweight_attention_model(
            dataset=train_dataset,
            pretrained=False
        )
    elif args.use_attention:
        print('Using Attention-Enhanced Model')
        model = create_attention_eeg_mvselect_model(
            dataset=train_dataset,
            arch=args.arch,
            aggregation=args.aggregation,
            pretrained=args.pretrained,
            use_cbam=args.use_cbam,
            use_mv_attention=args.use_mv_attention,
            use_vs_attention=args.use_vs_attention
        )
    else:
        print('Using Standard Model')
        model_config = {
            'arch': args.arch,
            'aggregation': args.aggregation,
            'pretrained': args.pretrained,
        }
        model = create_eeg_mvselect_model(train_dataset, model_config)
    model = model.cuda()
    
    # 체크포인트 로드
    if args.resume:
        print(f'\nLoading checkpoint: {args.resume}')
        checkpoint = torch.load(args.resume, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print('Checkpoint loaded successfully!')
    
    # Trainer
    trainer = EEGTrainer(model, args)
    
    # 평가 모드
    if args.eval:
        print('\n' + '='*70)
        print('Evaluation Mode')
        print('='*70)
        
        # 전체 뷰 사용
        print('\n[All Views]')
        trainer.evaluate(test_loader, init_cam_list=[None])
        
        if args.steps > 0:
            # MVSelect with different initial views
            print('\n[MVSelect with Different Initial Views]')
            init_cams = list(range(min(train_dataset.num_views, 5)))
            trainer.evaluate(test_loader, init_cam_list=init_cams)
            
            # Oracle performance
            print('\n[Oracle Performance]')
            trainer.test_oracle(test_loader)
        
        return
    
    # 학습 모드
    print('\n' + '='*70)
    print('Training')
    print('='*70)
    
    # 옵티마이저 및 스케줄러
    optimizer = create_optimizer(model, args)
    scheduler = create_scheduler(optimizer, args, len(train_loader)) \
                if args.scheduler != 'none' else None
    
    # 학습 루프
    best_val_acc = 0
    
    for epoch in range(1, args.epochs + 1):
        print(f'\nEpoch {epoch}/{args.epochs}')
        print('-' * 70)
        
        # 학습
        train_loss, train_acc = trainer.train_epoch(
            epoch, train_loader, optimizer, scheduler, args.log_interval
        )
        
        print(f'\nTrain - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%')
        
        # 검증
        if epoch % 5 == 0 or epoch == args.epochs:
            print('\nValidation:')
            val_losses, val_accs = trainer.evaluate(val_loader, init_cam_list=[None])
            val_acc = val_accs[0].item()
            
            # 최고 성능 모델 저장
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'args': args
                }
                torch.save(checkpoint, os.path.join(save_dir, 'best_model.pth'))
                print(f'Best model saved! (Val Acc: {val_acc:.2f}%)')
        
        # 에폭별 체크포인트
        if epoch % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'args': args
            }
            torch.save(checkpoint, os.path.join(save_dir, f'checkpoint_epoch{epoch}.pth'))
        
        # 스케줄러 업데이트 (에폭 단위)
        if scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
            scheduler.step()
    
    # 최종 테스트
    print('\n' + '='*70)
    print('Final Test with Best Model')
    print('='*70)
    
    # 최고 모델 로드
    checkpoint = torch.load(os.path.join(save_dir, 'best_model.pth'), weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 전체 뷰 테스트
    print('\n[All Views]')
    trainer.evaluate(test_loader, init_cam_list=[None])
    
    if args.steps > 0:
        # MVSelect 테스트
        print('\n[MVSelect with Different Initial Views]')
        init_cams = list(range(min(train_dataset.num_views, 5)))
        trainer.evaluate(test_loader, init_cam_list=init_cams)
        
        # Oracle
        print('\n[Oracle Performance]')
        trainer.test_oracle(test_loader)
    
    print('\n' + '='*70)
    print(f'Training completed! Results saved to: {save_dir}')
    print('='*70)


class Logger:
    """stdout을 파일과 콘솔에 동시 출력"""
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = log_file
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()


if __name__ == '__main__':
    main()
