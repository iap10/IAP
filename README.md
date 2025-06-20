# IAP: 얼굴 및 손-물체 인식을 활용한 지능형 학습 감시 시스템

## Introduction
COVID-19 이후 비대면 교육과 자율 학습 환경이 일상화되면서, 많은 학습자들이 집중력 저하를 겪고 있다.   

지도자의 직접적인 피드백이 어려운 온라인 환경에서는 산만함, 졸음, 스마트폰 사용 등 외부 요인에 쉽게 노출되고, 이러한 주의력 결핍은 학습 효과 저하로 이어지는 사례도 존재한다. 하지만 현재 대부분의 시스템은 수동으로 캠 화면을 확인 및 기록하거나 얼굴 인식 기반의 단순 시선 추적 기능에 의존하고 있어 실질적인 집중도 확인을 정량적으로 파악하기 어렵다는 한계를 지닌다. 
  
본 프로젝트는 이러한 문제를 해결하고자, 실시간 집중도 판단이 가능한 지능형 학습 감시 시스템을 제안한다. 본 시스템은 웹캠 영상만으로 얼굴과 손 및 물체를 탐지하고, 얼굴 자세와 손에 쥔 물체의 정보를 종합 분석하여 학습자의 집중 여부를 판단한다. 플랫폼으로서 NVIDIA Jetson Nano 보드를 사용하며 엣지 환경에서 동작 가능하도록 모델을 경량화하고 실시간성 확보를 위한 최적화 기법을 적용하였다. 

## Proposed Architecture
### Final Architecture (Case 4)
Case 4) Multi-Thread with 2 Thread(1 GPU 1 CPU)
<div style="display: inline-block;">
  <img src="https://github.com/user-attachments/assets/c9fd7b2b-6e8b-443d-8ab3-8ddaf10d6f38" width="100%">
</div>
<div style="display: inline-block;">
  <img src="https://github.com/user-attachments/assets/db887f83-e872-417c-999a-83d62085a86e" width="70%">
</div>

### Case 1
Case 1) Baseline: Multi-Threading with 4 Thread
<div style="display: inline-block;">
  <img src="https://github.com/user-attachments/assets/0d00849b-b777-443c-b0d8-2ce4b94b181c" width="100%">
</div>
<div style="display: inline-block;">
  <img src="https://github.com/user-attachments/assets/305c1e5e-e4f0-4d79-ba26-af4076867f3d" width="70%">
</div>

### Case 2
Case 2) Single Thread with sequential GPU access
<p align="center">
<img width="100%" src="https://github.com/user-attachments/assets/ef4df795-e761-4ac7-a1a7-94b5a3b5cc33">
</p>

### Case 3
Case 3) Multi-Thread with 2 Thread(2 GPU)
<div style="display: inline-block;">
  <img src="https://github.com/user-attachments/assets/25cf8588-7226-43fa-8071-f55d81bde553" width="100%">
</div>
<div style="display: inline-block;">
  <img src="https://github.com/user-attachments/assets/d7d9759d-d9c4-4b38-be7a-b1cdd597b042" width="70%">
</div>
