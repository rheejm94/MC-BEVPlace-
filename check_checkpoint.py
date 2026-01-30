import torch
import json

def check_checkpoint():
    checkpoint_path = 'runs/Aug13_11-33-29/checkpoint.pth.tar'
    
    try:
        # 체크포인트 로드
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        print("=== 체크포인트 파일 내용 ===")
        print(f"파일: {checkpoint_path}")
        print()
        
        # 저장된 정보들 확인
        print("저장된 키들:")
        for key in checkpoint.keys():
            print(f"  - {key}")
        print()
        
        # 에포크 정보
        if 'epoch' in checkpoint:
            print(f"현재 에포크: {checkpoint['epoch']}")
        
        # Recall 정보
        if 'recalls' in checkpoint:
            print(f"Mean Recall: {checkpoint['recalls']:.4f}")
        
        # Best Score
        if 'best_score' in checkpoint:
            print(f"Best Score: {checkpoint['best_score']:.4f}")
        
        # 옵티마이저 정보
        if 'optimizer' in checkpoint:
            print("옵티마이저 상태 저장됨")
            
    except Exception as e:
        print(f"체크포인트 로드 오류: {e}")

if __name__ == "__main__":
    check_checkpoint() 