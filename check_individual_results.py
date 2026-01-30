#!/usr/bin/env python3
import torch
import os
import glob
import json

def check_individual_dataset_results():
    print("=== BEVPlace++ 개별 데이터셋 성능 확인 ===\n")
    
    # 최신 실행 폴더 찾기
    run_folders = glob.glob('runs/*')
    if not run_folders:
        print("실행 폴더를 찾을 수 없습니다.")
        return
    
    latest_folder = max(run_folders, key=os.path.getctime)
    print(f"실행 폴더: {latest_folder}")
    
    # 체크포인트에서 전체 성능 확인
    checkpoint_path = os.path.join(latest_folder, 'checkpoint.pth.tar')
    if os.path.exists(checkpoint_path):
        try:
            ckpt = torch.load(checkpoint_path, map_location='cpu')
            print(f"\n📊 전체 성능:")
            print(f"   최종 에포크: {ckpt['epoch']}")
            print(f"   Mean Recall: {ckpt['recalls']:.4f}")
            print(f"   Best Score: {ckpt['best_score']:.4f}")
            print(f"   성공률: {ckpt['recalls']*100:.2f}%")
        except Exception as e:
            print(f"체크포인트 로드 오류: {e}")
    
    # TensorBoard 이벤트 파일에서 개별 성능 확인
    event_files = glob.glob(os.path.join(latest_folder, 'events*'))
    if event_files:
        latest_event = max(event_files, key=os.path.getctime)
        print(f"\n📈 TensorBoard 로그 파일: {os.path.basename(latest_event)}")
        print("   개별 시퀀스 성능은 TensorBoard에서 확인 가능합니다.")
        print("   실행 방법: tensorboard --logdir=" + latest_folder)
    
    # NCLT 시퀀스 목록
    nclt_sequences = [
        '2012-01-15',
        '2012-02-04', 
        '2012-03-17',
        '2012-06-15',
        '2012-09-28',
        '2012-11-16',
        '2013-02-23'
    ]
    
    print(f"\n🔍 NCLT 시퀀스별 성능:")
    print("   TensorBoard에서 다음 태그들로 확인 가능:")
    for seq in nclt_sequences:
        print(f"   - NCLT_{seq}")
    
    print(f"\n💡 확인 방법:")
    print("   1. TensorBoard 실행:")
    print(f"      tensorboard --logdir={latest_folder}")
    print("   2. 브라우저에서 http://localhost:6006 접속")
    print("   3. 'SCALARS' 탭에서 각 시퀀스별 Recall 확인")
    
    # 최근 실행된 학습 프로세스 확인
    print(f"\n🔄 학습 상태:")
    try:
        import psutil
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            if 'python3' in proc.info['name'] and 'main.py' in ' '.join(proc.info['cmdline'] or []):
                print(f"   학습 실행 중 (PID: {proc.info['pid']})")
                break
        else:
            print("   학습 완료됨")
    except ImportError:
        print("   psutil 모듈 없음 - 실행 상태 확인 불가")

if __name__ == "__main__":
    check_individual_dataset_results() 