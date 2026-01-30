#!/usr/bin/env python3
import torch
import os
import glob

def check_latest_results():
    print("=== BEVPlace++ 학습 결과 확인 ===\n")
    
    # 최신 실행 폴더 찾기
    run_folders = glob.glob('runs/*')
    if not run_folders:
        print("실행 폴더를 찾을 수 없습니다.")
        return
    
    latest_folder = max(run_folders, key=os.path.getctime)
    print(f"최신 실행 폴더: {latest_folder}")
    
    # 체크포인트 파일 확인
    checkpoint_path = os.path.join(latest_folder, 'checkpoint.pth.tar')
    if os.path.exists(checkpoint_path):
        try:
            ckpt = torch.load(checkpoint_path, map_location='cpu')
            print(f"\n📊 현재 상태:")
            print(f"   에포크: {ckpt['epoch']}")
            print(f"   Mean Recall: {ckpt['recalls']:.4f}")
            print(f"   Best Score: {ckpt['best_score']:.4f}")
            
            # 파일 크기로 진행 상황 추정
            file_size = os.path.getsize(checkpoint_path) / (1024*1024)  # MB
            print(f"   체크포인트 크기: {file_size:.1f} MB")
            
        except Exception as e:
            print(f"체크포인트 로드 오류: {e}")
    else:
        print("체크포인트 파일을 찾을 수 없습니다.")
    
    # TensorBoard 이벤트 파일 확인
    event_files = glob.glob(os.path.join(latest_folder, 'events*'))
    if event_files:
        latest_event = max(event_files, key=os.path.getctime)
        event_size = os.path.getsize(latest_event) / 1024  # KB
        print(f"\n📈 TensorBoard 로그:")
        print(f"   파일: {os.path.basename(latest_event)}")
        print(f"   크기: {event_size:.1f} KB")
        print(f"   웹 접속: http://localhost:6006")
    
    # 실행 중인지 확인
    print(f"\n🔄 실행 상태:")
    try:
        import psutil
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            if 'python3' in proc.info['name'] and 'main.py' in ' '.join(proc.info['cmdline'] or []):
                print(f"   학습 실행 중 (PID: {proc.info['pid']})")
                break
        else:
            print("   현재 학습이 실행되지 않음")
    except ImportError:
        print("   psutil 모듈 없음 - 실행 상태 확인 불가")

if __name__ == "__main__":
    check_latest_results() 