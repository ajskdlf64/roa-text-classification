# ROA Text Classification Model
8개의 Text Classification Model을 학습하고 평가하고 사용하는 모델입니다.

##  01. install
```powershell
git clone https://github.com/ajskdlf64/roa-text-classification.git
```
```powershell
pip install -r requirements.txt
```

## 02. setup dataset
- 학습에 사용할 데이터셋을 `dataset/original` dir 하위에 위치시킵니다.
- 이 때 데이터셋은 다음과 같은 3가지 조건을 반드시 충족시켜야 합니다.
  - 조건1) original 및 test 데이터셋은 xlsx 확장자를 가져야 합니다.
  - 조건2) 데이터셋 파일 안에는 아래의 3개의 칼럼이 반드시 존재해야 합니다.
     - title : 기사 제목
     - text : 기사 본문
     - label : 해당 데이터셋의 분류 label 여부 1 or 0 으로 기록
  - 조건3) 8개의 데이터셋은 아래의 파일명을 가져야 합니다.
    -  불필요 : category
    - 투자 : investment
    - 실적 : corporate_performance
    - 출시 : release
    - 인수합병 : mna
    - 제휴 : partnership
    - 상장 : go_public
    - 인사 : personnel_changes

## 02. dataprocess
- label 비율에 따라 down sampling을 진행하고, 
```powershell
python dataprocess.py --seed 1234 --val_ratio 0.1 --make_test_ratio 0.1
```

## 03. train
```powershell
python train.py --seed 1234 --max_epochs 1 --lr 3e-5 --batch_size 16 --backbone distilbert-base-multilingual-cased
```

## 04. eval
```powershell
python eval.py
```

## 06. run.sh
```powershell
python dataprocess.py --seed 1234 --val_ratio 0.1 --test_ratio 0.1 &&
python train.py --seed 1234 --max_epochs 1 --lr 3e-5 --batch_size 16 --backbone distilbert-base-multilingual-cased &&
python eval.py
```

## 07. infer