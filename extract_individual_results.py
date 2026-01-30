#!/usr/bin/env python3
import os
import sys
import glob

# TensorBoard 경로 추가
sys.path.append('/home/keti/RJM_projects/myenv/lib/python3.8/site-packages')

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    import numpy as np
    
    def extract_individual_results():
        print("=== BEVPlace++ 개별 시퀀스 성능 ===\n")
        
        # 최신 실행 폴더
        run_folder = 'runs/Aug13_11-33-29'
        
        # EventAccumulator 생성
        ea = EventAccumulator(run_folder)
        ea.Reload()
        
        # NCLT 시퀀스 목록
        sequences = [
            '2012-01-15',
            '2012-02-04', 
            '2012-03-17',
            '2012-06-15',
            '2012-09-28',
            '2012-11-16',
            '2013-02-23'
        ]
        
        print("📊 개별 시퀀스별 최종 Recall:")
        print("-" * 50)
        
        individual_recalls = []
        
        for seq in sequences:
            tag = f'val/NCLT_{seq}'
            scalars = ea.Scalars(tag)
            
            if scalars:
                # 최종 Recall 값
                final_recall = scalars[-1].value
                individual_recalls.append(final_recall)
                
                # 에포크 정보
                final_epoch = scalars[-1].step
                
                print(f"{seq:12}: {final_recall:.4f} ({final_recall*100:.2f}%) [Epoch {final_epoch}]")
            else:
                print(f"{seq:12}: 데이터 없음")
                individual_recalls.append(0)
        
        print("-" * 50)
        
        # 통계 계산
        if individual_recalls:
            mean_recall = np.mean(individual_recalls)
            max_recall = np.max(individual_recalls)
            min_recall = np.min(individual_recalls)
            std_recall = np.std(individual_recalls)
            
            print(f"📈 통계:")
            print(f"   평균 Recall: {mean_recall:.4f} ({mean_recall*100:.2f}%)")
            print(f"   최고 Recall: {max_recall:.4f} ({max_recall*100:.2f}%)")
            print(f"   최저 Recall: {min_recall:.4f} ({min_recall*100:.2f}%)")
            print(f"   표준편차: {std_recall:.4f}")
            
            # 성능 순위
            print(f"\n🏆 성능 순위:")
            seq_performance = list(zip(sequences, individual_recalls))
            seq_performance.sort(key=lambda x: x[1], reverse=True)
            
            for i, (seq, recall) in enumerate(seq_performance, 1):
                print(f"   {i:2d}. {seq:12}: {recall:.4f} ({recall*100:.2f}%)")
        
        # 학습 손실 정보도 확인
        print(f"\n📉 학습 정보:")
        train_loss = ea.Scalars('Train/Loss')
        if train_loss:
            final_loss = train_loss[-1].value
            print(f"   최종 Loss: {final_loss:.4f}")
        
        avg_loss = ea.Scalars('Train/AvgLoss')
        if avg_loss:
            final_avg_loss = avg_loss[-1].value
            print(f"   최종 평균 Loss: {final_avg_loss:.4f}")
            
    if __name__ == "__main__":
        extract_individual_results()
        
except ImportError as e:
    print(f"TensorBoard 모듈을 찾을 수 없습니다: {e}")
    print("가상환경을 활성화하고 tensorboard를 설치하세요:")
    print("pip install tensorboard")
except Exception as e:
    print(f"오류 발생: {e}") 