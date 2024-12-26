# 📖Solution 

## What is Anomaly Detection?

Anomaly Detection은 데이터 내에 존재하는 비정상적인 패턴이나 이상치를 탐지하는 AI 컨텐츠입니다. 별도의 레이블링 작업 없이도 통계 및 Machine learning 기법을 활용하여 데이터의 정상 범주와 비정상 범주를 자동으로 구분해 냅니다. Anomaly Detection은 개별 데이터 포인트가 정상 범주에서 벗어나는지를 탐지하는 Point anomaly detection(PAD), 정상 패턴의 범주에서 벗어난 이상 패턴을 식별하는 Contextual anomaly detection(CAD, TBD), 그리고 다변량 데이터의 관계를 종합적으로 학습하여 비정상적인 시점과 패턴을 탐지하는 Multivariate anomaly detection(MAD, TBD) 모델을 제공합니다. 이렇게 Anomaly Detection은 데이터의 특성과 목적에 맞게 다양하게 활용될 수 있습니다.

## When to use Anomaly Detection?

Anomaly Detection 적용이 가능한 분야는 다음과 같습니다.  

제조 과정 이상치 탐지: 제조 과정 모니터링 센서로 제조 과정 중 이상 발생 여부를 확인하려는 고객을 위한 기능입니다. 이상 발생 여부를 탐지하여 문제를 사전에 방지할 수 있습니다.  
시계열 이상치 탐지: 제조 과정 뿐만 아니라 주식, 각종 추세 데이터 등 시계열 데이터에 대하여 이상치를 확인하고자 하는 고객입니다. 해당 이상치를 조기에 탐지하여 사용자는 적절한 조치를 취할 수 있게 됩니다.

## Key features and benefits 

> 기술적 관점 혹은 도메인 관점에서 작성해주세요.

Anomaly Detection은 빠르고 높은 효율성을 갖춘 통계 기반의 모델을 제공하여, 학습 리소스를 크게 필요로 하지 않으면서도 뛰어난 성능을 발휘합니다.

이는 제품 생산 과정에서 얻어진 일차원 데이터 내에서 이상치를 탐지하고자 하는 고객들에게 유용합니다. 또한 주가 추세 등과 같은 시계열 데이터에서 갑작스런 특이 사항이 발생한 포인트를 탐지하고자 하는 고객들에게도 유용한 솔루션입니다.

**빠른 속도와 메모리 요구가 낮은 통계 기반 모델**  
Anomaly detection를 통해 우수하고 학습 리소스가 많이 필요하지 않은 효율적인 통계 및 머신러닝 기반의 모델을 손쉽게 사용할 수 있습니다. 빠른 속도와 메모리 요구가 낮은 DynamicThreshold, SpectralResidual과 같은 알고리즘을 기반으로 AD는 신뢰성 있고, 빠른 속도로 이상치를 탐지해낼 수 있습니다.

**복수 컬럼에 대한 이상치 탐지 및 그룹 별 이상치 탐지**  
경우에 따라 각 포인트 마다 이상치 탐지를 하고 싶은 데이터가 여러 종류가 존재하거나, 서로 다른 그룹 혹은 센서 등에서 얻어진 데이터들을 그룹별로 이상 탐지를 진행해야 할 수 있습니다. AD는 실험계획서의 argument만 변경해줌으로써 이를 쉽고 빠르게 진행할 수 있습니다.

**실험계획서를 이용한 간편한 사용성**  
Anomaly Detection 중 Point Anomaly Detection 모델은 DynamicThreshold, SpectralResidual, stl_DynamicThreshodl, stl_SpectralResidual이라는 4가지 모델을 제공합니다. 이런 모델을 사용하기 위해 실험계획서의 argument만 변경하면 쉽게 사용할 수 있어, 간편하면서도 효율적인 사용성을 제공합니다.


# 💡Features

## Pipeline

AI Contents의 pipeline은 기능 단위인 asset의 조합으로 이루어져 있습니다. AD는 총 6가지 asset의 조합으로 pipeline이 구성되어 있습니다.  

**Train pipeline**
```
Input - Readiness - Preprocess - Train
```

**Inference pipeline**
```
Input - Readiness - Preprocess - Inference - Output
```

## Assets

> asset_{}.py 외 dataset.py 등 기타 모듈 설명도 포함해주세요.

각 단계는 asset으로 구분됩니다.

**Input asset**  
Anomaly Detection은 학습할 때 이상탐지를 하고자 하는 x 컬럼이 포함된 tabular 형태의 데이터로 동작합니다. 따라서 input asset에서는 해당 csv 데이터를 읽고 다음 asset으로 전달하는 역할을 수행합니다. 위 도표에서 보듯 다음 asset인 Preprocess asset으로 전달됩니다.

**Readiness asset**  
Readniess asset에서는 전처리와 학습 및 추론을 진행하기 전 데이터 입력과 설정 등이 조건에 맞게 되어 있는지를 체크하게 됩니다.

**Preprocess asset**  
Anomaly Detection은 학습 혹은 추론이 진행되기 전 데이터를 전처리하는 과정을 먼저 거치게 되며 이 과정이 process asset에서 진행됩니다. 해당 과정에서는 각 포인트마다 빈 값 혹은 NaN과 같이 정상적이지 않은 포인트를 제거하거나 scaling을 하는 등을 과정이 진행됩니다.

**Train asset**  
Anomaly Detection에서 train asset은 이상 탐지를 위한 최적의 파라미터를 찾아내는 과정을 수행합니다. 우선 preprocess asset을 거쳐서 전처리가 완료된 데이터를 전달 받습니다.그 후 실험계획서에 미리 작성한 설정에 맞게 데이터를 그룹별, 컬럼별로 나누게 됩니다. 나누어진 데이터마다 모델별로 인스턴스를 생성하고 bayesian optimization을 통해 모델 별로 최적을 파라미터들 찾아내는 과정을 진행합니다. 이 과정이 완료되고 나면 모델 별 성능과 학습 과정에서의 추론 결과를 저장합니다.만약 x 컬럼 별로 레이블이 존재하는 경우 unsupervised 기반의 metric이 아닌, f1 score, precision, recall을 직접 maximize 하도록 설정할 수 있습니다. 이 경우 모델 별 성능이 output csv에 같이 저장되고, 모델 별 성능을 비교한 plot 또한 저장됩니다.

**Inference asset**  
inference asset에서는 input asset, preprocess asset을 거쳐 전달받은 inference 데이터를 읽고, 다시 그룹 별, 데이터 컬럼별로 나누게 됩니다. 나누어진 데이터마다 이전 학습 단계에서 저장한 모델들 중 선택된 모델들을 불러와서 이상 탐지를 진행하고 결과를 저장합니다.만약 레이블 컬럼이 존재하는 경우 모델 별 성능이 output csv에 같이 저장되고, 모델 별 성능을 비교한 plot 또한 저장됩니다. 운영을 위한 파일은 inference_summary.yaml도 저장됩니다.

## Experimental_plan.yaml

내가 갖고 있는 데이터에 AI Contents를 적용하려면 데이터에 대한 정보와 사용할 Contents 기능들을 experimental_plan.yaml 파일에 기입해야 합니다. AI Contents를 solution 폴더에 설치하면 solution 폴더 아래에 contents 마다 기본으로 작성되어있는 experimental_plan.yaml 파일을 확인할 수 있습니다. 이 yaml 파일에 '데이터 정보'를 입력하고 asset마다 제공하는 'user arugments'를 수정/추가하여 ALO를 실행하면, 원하는 세팅으로 데이터 분석 모델을 생성할 수 있습니다.

**experimental_plan.yaml 구조**  
experimental_plan.yaml에는 ALO를 구동하는데 필요한 다양한 setting값이 작성되어 있습니다. 이 setting값 중 '데이터 경로'와 'user arguments'부분을 수정하면 AI Contents를 바로 사용할 수 있습니다.

**데이터 경로 입력(external_path)**  
external_path의 parameter는 불러올 파일의 경로나 저장할 파일의 경로를 지정할 때 사용합니다. save_train_artifacts_path와 save_inference_artifacts_path는 입력하지 않으면 default 경로인 train_artifacts, inference_artifacts 폴더에 모델링 산출물이 저장됩니다.
```
external_path:
    - load_train_data_path: ./solution/sample_data/train
    - load_inference_data_path:  ./solution/sample_data/test
    - save_train_artifacts_path:
    - save_inference_artifacts_path:
```

|파라미터명|DEFAULT|설명 및 옵션|
|---|----|---|
|load_train_data_path|	./sample_data/train/|	학습 데이터가 위치한 폴더 경로를 입력합니다.(csv 파일 명 입력 X)|
|load_inference_data_path|	./sample_data/test/|	추론 데이터가 위치한 폴더 경로를 입력합니다.(csv 파일 명 입력 X)|

**사용자 파라미터(user_parameters)**  
user_parameters 아래 step은 asset 명을 의미합니다. 아래 step: input은 input asset단계임을 의미합니다.
args는 input asset(step: input)의 user arguments를 의미합니다. user arguments는 각 asset마다 제공하는 데이터 분석 관련 설정 파라미터입니다. 이에 대한 설명은 아래에 User arguments 설명을 확인해주세요.
```
user_parameters:
    - train_pipeline:
        - step: input
          args:
            - file_type
            ...
          ui_args:
            ...

```
## Algorithms and models

TCR의 Train/Inference asset에는 총 5종의 모델이 내장되어 있습니다. 데이터 분석가들이 classification/regression 모델을 만들며 자주 사용한 모델 5종과 각 모델 별 파라미터 세트를 선정하였습니다. TCR의 모델 리스트와 파라미터 세트는 아래와 같습니다

TCR 내장 모델
- rf: random forest  
(max_depth: 6, n_estimators: 300), (max_depth: 6, n_estimators: 500)
- gbm: gradient boosting machine  
(max_depth: 5, n_estimators: 300), (max_depth: 6, n_estimators: 400), (max_depth: 7, n_estimators: 500)
- lgbm: light gradient boosting machine  
(max_depth: 5, n_estimators: 300), (max_depth: 7, n_estimators: 400), (max_depth: 9, n_estimators: 500)
- cb: catboost  
(max_depth: 5, n_estimators: 100), (max_depth: 7, n_estimators: 300), (max_depth: 9, n_estimators: 500)
- xgb: Extreme Gradient Boosting  
(max_depth: 5, n_estimators: 300), (max_depth: 6, n_estimators: 400), (max_depth: 7, n_estimators: 500)

![poster](./catboost.png)


# 📂Input and Artifacts

## 데이터 준비

**학습 데이터 준비**  
1. 이상 탐지를 하고자 하는 포인트들이 컬럼으로 이루어진 csv 파일을 준비합니다.
2. 모든 csv 파일은 해당 row를 식별할 수 있는 time column이 존재해야 합니다.
3. 만약 time column이 중복되는 경우 이를 drop 하도록 설정할 수 있습니다. 만약 drop하지 않는 경우 row별로 식별이 가능하도록 하는 컬럼들이 별도로 존재해야 합니다.
4. label 컬럼은 optional 합니다. 만약 존재하는 경우 x 컬럼 별로 모두 존재해야 합니다.
5. 그룹 별로 묶는 경우 그룹 별로 묶기 위한 컬럼이 존재해야 합니다.

**학습 데이터셋 예시**

|time_col|x_col_1|x_col_2|grouupkey|
|------|---|---|---|
|time 1|value 1_1|value 1_2|group1|
|time 2|value 2_1|value2_2|group2|
|time 3|value 3_1|value3_2|group1|

**Input data directory 구조 예시**  
- ALO를 사용하기 위해서는 train과 inference 파일이 분리되어야 합니다. 아래와 같이 학습에 사용할 데이터와 추론에 사용할 데이터를 구분해주세요.
- 하나의 폴더 아래 있는 모든 파일을 input asset에서 취합해 하나의 dataframe으로 만든 후 모델링에 사용됩니다. (경로 밑 하위 폴더 안에 있는 파일도 합쳐집니다.)
- 하나의 폴더 안에 있는 데이터의 컬럼은 모두 동일해야 합니다.
```
./{train_folder}/
    └ train_data1.csv
    └ train_data2.csv
    └ train_data3.csv
./{inference_folder}/
    └ inference_data1.csv
    └ inference_data2.csv
    └ inference_data3.csv
```

## 데이터 요구사항

**필수 요구사항**  
입력 데이터는 다음 조건을 반드시 만족하여야 합니다.

|index|item|spec.|
|----|----|----|
|1|x 컬럼의 개수|1개 이상|
|2|time 컬럼의 개수|1개|
|3|x 컬럼의 타입|float|

**추가 요구사항**  
최소한의 성능을 보장하기 위한 조건입니다. 하기 조건이 만족되지 않아도 알고리즘은 돌아가지만 성능은 확인되지 않았습니다    

|index|item|spec.|
|----|-----|---|
|1|time 컬럼|time 컬럼의 값은 중복이 최대한 적어야 합니다. 중복되는 time 컬럼 값이 있는데 중복 time 컬럼을 drop 시키게 하면 원치 않게 핻들이 삭제 될 수 있습니다.|
|2|NG 데이터|NG 데이터가 아예 존재하지 않는 경우 성능에 영향을 미칠 수 있습니다.|
|3|group key|group 마다 데이터 양이 충분해야 합니다. 그렇지 않은 경우 성능에 악영향을 미칠 수 있습니다.|

## 산출물

학습/추론을 실행하면 아래와 같은 산출물이 생성됩니다.  

**Train pipeline**
```
./alo/train_artifacts/
    └ models/preprocess/
        └ train_config.pkl
        └ train_pipeline_x.pkl
    └ models/train/prep_{x 컬럼명}
        └ train_params.pkl
    └ output/
        └ train_result.csv
    └ extra_output/train/prep_{x 컬럼명}
        └ confusion_matrix.jpg (y 컬럼 입력시 생성)
        └ plot_anomaly_model_best_model.jpg (y 컬럼 입력시 생성)
        └ plot_anomaly_model_{선택한 모델 명}.jpg (y 컬럼 입력시 생성)
        └ score_compare.jpg (y 컬럼 입력시 생성)
```

**Inference pipeline**
```
 ./alo/inference_artifacts/
    └ output/inference/
        └ output.csv
    └ extra_output/inference/prep_{x 컬럼명}
        └ confusion_matrix.jpg (y 컬럼 입력시 생성)
        └ plot_anomaly_model_best_model.jpg (y 컬럼 입력시 생성)
        └ plot_anomaly_model_{선택한 모델 명}.jpg (y 컬럼 입력시 생성)
        └ score_compare.jpg (y 컬럼 입력시 생성)
    └ score/
        └ inference_summary.yaml

```

각 산출물에 대한 상세 설명은 다음과 같습니다.  

**train_config.pkl**  
preprocess asset의 argument를 담고 있는 pickle 파일입니다.

**train_pipeline_x.pkl**  
preprocess asset에서 x 컬럼을 전처리하는 모델을 저장한 pickle 파일입니다.

**train_params.pkl**  
train asset에서 학습을 진행 후 모델을 저장한 pkl 파일입니다.

**train_result.csv**    
train pipeline이 끝난 후 결과를 저장한 csv 파일입니다.

**confusion_matrix.jpg**  
y 컬럼이 주어진 경우 모델(들)의 train data를 이용한 confusion matrix를 plot한 jpg 파일입니다.

**plot_anomaly_model_best_model.jpg**  
모델(들) 중 score가 가장 높은 모델이 train data를 anomaly detection한 결과를 plot한 jpg 파일입니다.