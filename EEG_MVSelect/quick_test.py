"""
빠른 테스트 스크립트 - 데이터 로드 및 모델 초기화 테스트
"""

import sys
import torch
from src.datasets import EEGConnectivityDataset
from src.models import EEGMVSelect

def quick_test():
    """빠른 동작 확인 테스트"""
    
    print("="*70)
    print("EEG-MVSelect Quick Test")
    print("="*70)
    
    # 1. 데이터셋 테스트
    print("\n[1/4] Testing dataset loading...")
    try:
        dataset = EEGConnectivityDataset(
            root_dir='/home/work/skku/icip/FCM_Images_HERMES_v2',
            split='train',
            image_size=224
        )
        
        if len(dataset) == 0:
            print("❌ Dataset is empty! Please check your data directory.")
            return False
        
        print(f"✅ Dataset loaded successfully!")
        print(f"   Samples: {len(dataset)}")
        print(f"   Classes: {dataset.num_class}")
        print(f"   Views: {dataset.num_views}")
        
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        return False
    
    # 2. 샘플 로드 테스트
    print("\n[2/4] Testing sample loading...")
    try:
        imgs, label, keep_cams = dataset[0]
        print(f"✅ Sample loaded successfully!")
        print(f"   Images shape: {imgs.shape}")
        print(f"   Label: {label}")
        print(f"   Keep_cams: {keep_cams.shape}")
        
    except Exception as e:
        print(f"❌ Sample loading failed: {e}")
        return False
    
    # 3. 모델 생성 테스트
    print("\n[3/4] Testing model creation...")
    try:
        model = EEGMVSelect(
            dataset=dataset,
            arch='resnet18',
            aggregation='max',
            pretrained=True
        )
        
        if torch.cuda.is_available():
            model = model.cuda()
            print(f"✅ Model created and moved to GPU!")
        else:
            print(f"✅ Model created on CPU!")
            print(f"   Warning: No GPU available, training will be slow")
        
        # 파라미터 수 출력
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   Total parameters: {total_params:,}")
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 4. Forward pass 테스트
    print("\n[4/4] Testing forward pass...")
    try:
        model.eval()
        
        # 배치 생성
        B = 2
        imgs_batch = imgs.unsqueeze(0).repeat(B, 1, 1, 1, 1)
        keep_cams_batch = keep_cams.unsqueeze(0).repeat(B, 1)
        
        if torch.cuda.is_available():
            imgs_batch = imgs_batch.cuda()
            keep_cams_batch = keep_cams_batch.cuda()
        
        # Forward (전체 뷰 사용)
        with torch.no_grad():
            output, _, _ = model(imgs_batch, down=1)
        
        print(f"✅ Forward pass (all views) successful!")
        print(f"   Output shape: {output.shape}")
        
        # Forward (MVSelect 사용)
        init_prob = torch.nn.functional.one_hot(
            torch.tensor([0, 0]), num_classes=dataset.num_cam
        )
        if torch.cuda.is_available():
            init_prob = init_prob.cuda()
        
        with torch.no_grad():
            output, _, (_, _, actions, _) = model(
                imgs_batch, init_prob=init_prob, steps=2, keep_cams=keep_cams_batch
            )
        
        print(f"✅ Forward pass (MVSelect) successful!")
        print(f"   Output shape: {output.shape}")
        print(f"   Number of selection steps: {len(actions)}")
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "="*70)
    print("✅ All tests passed successfully!")
    print("="*70)
    print("\nYou can now proceed with training:")
    print("  ./train.sh")
    print("or")
    print("  python main.py --epochs 50 --steps 0")
    print("\n")
    
    return True


if __name__ == '__main__':
    success = quick_test()
    sys.exit(0 if success else 1)
