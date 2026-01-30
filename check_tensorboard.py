import os
import sys
sys.path.append('/home/keti/RJM_projects/myenv/lib/python3.8/site-packages')

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    import numpy as np
    
    # 이벤트 파일 경로
    log_dir = 'runs/Aug13_11-33-29'
    
    # EventAccumulator 생성
    ea = EventAccumulator(log_dir)
    ea.Reload()
    
    print("=== TensorBoard 이벤트 파일 내용 ===")
    print(f"로그 디렉토리: {log_dir}")
    print()
    
    # 사용 가능한 태그들 확인
    print("사용 가능한 태그들:")
    for tag_type in ['scalars', 'images', 'histograms']:
        tags = getattr(ea, tag_type)()
        if tags:
            print(f"{tag_type.upper()}:")
            for tag in tags:
                print(f"  - {tag}")
    print()
    
    # 스칼라 값들 확인 (Recall 값들)
    scalars = ea.Scalars('val')
    if scalars:
        print("검증 결과 (Recall 값들):")
        for scalar in scalars:
            print(f"  Step {scalar.step}: {scalar.value:.4f}")
    
    # 학습 손실 확인
    train_loss = ea.Scalars('Train/Loss')
    if train_loss:
        print("\n학습 손실:")
        for loss in train_loss[-10:]:  # 최근 10개만
            print(f"  Step {loss.step}: {loss.value:.4f}")
    
    # 평균 손실 확인
    avg_loss = ea.Scalars('Train/AvgLoss')
    if avg_loss:
        print("\n평균 학습 손실:")
        for loss in avg_loss:
            print(f"  Epoch {loss.step}: {loss.value:.4f}")
            
except ImportError as e:
    print(f"TensorBoard 모듈을 찾을 수 없습니다: {e}")
    print("가상환경을 활성화하고 tensorboard를 설치하세요:")
    print("pip install tensorboard")
except Exception as e:
    print(f"오류 발생: {e}") 