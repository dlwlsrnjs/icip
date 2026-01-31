"""
데이터셋 구조 확인 및 통계 출력 스크립트
"""

import os
import sys
import glob
from collections import defaultdict

def check_dataset_structure(root_dir):
    """데이터셋 구조 확인"""
    
    print("="*70)
    print("EEG Connectivity Dataset Structure Check")
    print("="*70)
    print(f"\nRoot directory: {root_dir}")
    
    if not os.path.exists(root_dir):
        print(f"\n❌ Error: Directory does not exist!")
        return False
    
    # 연결성 방법별 체크
    methods = ['FCM', 'PCC', 'PLV']
    bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    
    stats = {}
    
    for method in methods:
        method_dir = os.path.join(root_dir, method)
        
        print(f"\n{'='*70}")
        print(f"Method: {method}")
        print(f"{'='*70}")
        
        if not os.path.exists(method_dir):
            print(f"❌ Directory not found: {method_dir}")
            continue
        
        # 피험자 폴더 찾기
        subject_dirs = [d for d in os.listdir(method_dir) 
                       if os.path.isdir(os.path.join(method_dir, d))]
        
        print(f"Number of subjects: {len(subject_dirs)}")
        
        if len(subject_dirs) == 0:
            print(f"❌ No subject directories found!")
            continue
        
        # 각 피험자별로 이미지 확인
        subject_stats = defaultdict(lambda: defaultdict(int))
        
        for subject_id in subject_dirs:
            subject_path = os.path.join(method_dir, subject_id)
            
            # 주파수 밴드별 이미지 찾기
            for band in bands:
                # 다양한 패턴으로 검색
                patterns = [
                    f"*{band}*.png",
                    f"*{band}*.jpg",
                    f"*{band.upper()}*.png",
                    f"*{band.capitalize()}*.png",
                ]
                
                found = False
                for pattern in patterns:
                    matches = glob.glob(os.path.join(subject_path, pattern))
                    if matches:
                        subject_stats[subject_id][band] = len(matches)
                        found = True
                        break
                
                if not found:
                    subject_stats[subject_id][band] = 0
        
        # 통계 출력
        complete_subjects = 0
        incomplete_subjects = []
        
        for subject_id, band_counts in subject_stats.items():
            if all(band_counts[band] > 0 for band in bands):
                complete_subjects += 1
            else:
                missing = [band for band in bands if band_counts[band] == 0]
                incomplete_subjects.append((subject_id, missing))
        
        print(f"\nComplete subjects (all {len(bands)} bands): {complete_subjects}")
        print(f"Incomplete subjects: {len(incomplete_subjects)}")
        
        if incomplete_subjects and len(incomplete_subjects) <= 10:
            print("\nIncomplete subjects details:")
            for subject_id, missing in incomplete_subjects:
                print(f"  {subject_id}: missing {missing}")
        
        # 샘플 피험자 상세 정보
        if subject_dirs:
            sample_subject = subject_dirs[0]
            sample_path = os.path.join(method_dir, sample_subject)
            print(f"\nSample subject: {sample_subject}")
            print(f"Files:")
            for fname in sorted(os.listdir(sample_path))[:10]:
                print(f"  - {fname}")
            if len(os.listdir(sample_path)) > 10:
                print(f"  ... and {len(os.listdir(sample_path)) - 10} more files")
        
        stats[method] = {
            'total_subjects': len(subject_dirs),
            'complete_subjects': complete_subjects,
            'incomplete_subjects': len(incomplete_subjects)
        }
    
    # 전체 요약
    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")
    
    total_views = 0
    for method in methods:
        if method in stats:
            complete = stats[method]['complete_subjects']
            print(f"{method}: {complete} complete subjects")
            total_views += complete
    
    num_views_per_sample = len(methods) * len(bands)
    print(f"\nTotal views per sample: {num_views_per_sample}")
    print(f"  Methods: {len(methods)} ({', '.join(methods)})")
    print(f"  Bands: {len(bands)} ({', '.join(bands)})")
    
    print(f"\n{'='*70}")
    
    # 데이터셋 사용 가능 여부 판단
    min_complete = min([stats[m]['complete_subjects'] for m in methods if m in stats])
    
    if min_complete > 0:
        print(f"✅ Dataset is ready for training!")
        print(f"   Minimum {min_complete} subjects with complete data")
        return True
    else:
        print(f"❌ Dataset is NOT ready!")
        print(f"   No subjects with complete data across all methods")
        return False


if __name__ == '__main__':
    root_dir = '/home/work/skku/icip/FCM_Images_HERMES_v2'
    
    if len(sys.argv) > 1:
        root_dir = sys.argv[1]
    
    check_dataset_structure(root_dir)
