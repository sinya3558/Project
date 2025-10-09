# Project
test 
## 해야할 것

- [x] 확률 뽑는거 제거
- [X] Project 코드 싹 날아감.. docker 지우기
- [x] rbg + depth 모드 지원
- [ ] ~~SUCTION4.0 모델 다시 받기. 아님 그냥 써야함~~
- [X] re-train model -> 과적합이다!
- [X] validation data -> train/val datasets
- [x] 샘플데이터 12 카테고리 : 이미지 5982 장, JSON 2985
- [X] dataset 100 카테고리  -> part0 file 조심
- [ ] ~~python3.7 + tf .x + cuda 11.5~~
- [ ] update requirements.txt
- [ ] python3.10 + ft 2.x + [cuda 11.5 <-> tf 2.11] (유지할 것)
- [ ] overfitting -> add normalized layers
- [ ] fix main func in train_gqcnn

기존 GQCNNTF 구조

1. conv1-1, conv1-2
2. conv2-1, conv2-2
3. conv3-1, conv3-2
4. fc3
5. pc1
6. pc2
7. fc4
8. fc5

커스텀 GQCNNTF 구조
1. conv1-1 + batch norm + conv1-2 + norm
2. conv2-1 + bn + conv2-2 + bn
3. conv3-1 + bn + conv3-2 + bn
4. fc3
7. fc4
8. fc5

