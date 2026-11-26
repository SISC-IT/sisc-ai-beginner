"""
6주차 실습: 과거 20거래일의 주가 특징으로 다음 거래일의 상승 여부 예측

비교 대상
    1. LSTM: 시계열의 순차적 변화를 기억하는 순환 신경망
    2. Transformer: Self-Attention으로 시점 간 관계를 학습하는 신경망

전체 과정
    데이터 수집 → 특징(Feature) 생성 → 시퀀스 구성 → 시간순 분할
    → 표준화 → 모델 학습 → 평가 → 예측/가중치 저장 → 그래프 출력

주의: 데이터 수집에는 인터넷이 필요하고, 실행 시 학습과 파일 저장이 진행됩니다.
주가 예측은 실습용이며 실제 투자 수익을 보장하지 않습니다.
"""

import copy
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yfinance as yf
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


# ===== 1. 실험 설정 =====
# 예측에 사용할 종목과 다운로드할 과거 데이터의 범위
TICKER = "AAPL"
PERIOD = "10y"
# 입력 한 개에 포함되는 연속 거래일 수: (20일, 특징 6개) 형태가 됨
SEQUENCE_LENGTH = 60
# 전체 시퀀스 중 학습 70%, 검증 15%, 나머지 테스트 15%
TRAIN_RATIO = 0.70
VALID_RATIO = 0.15
# 학습 시 한 번에 처리하는 시퀀스 수
BATCH_SIZE = 64
# Adam 옵티마이저가 가중치를 수정하는 크기를 결정
LEARNING_RATE = 0.001
# 최대 50회 학습하되 검증 성능이 개선되지 않으면 일찍 종료
MAX_EPOCHS = 50
PATIENCE = 8
# 난수 시드를 고정하여 실행 간 결과 차이를 줄임
RANDOM_SEED = 42
# 주가 차트에 표시할 최근 테스트 거래일 수 (원하면 60, 150 등으로 변경)
PLOT_DAYS = 100

# ===== 백테스트 설정 =====
# 시작 투자금. 실제 계좌 규모와 관계없이 비교하기 위한 가상 금액
BACKTEST_INITIAL_CAPITAL = 10_000
# 매수와 매도 각각에 적용할 비용. 5bp = 0.05% (수수료 + 슬리피지의 가정치)
# 예: 10,000달러를 매수하면 약 5달러, 매도하면 다시 약 5달러의 비용
BACKTEST_COST_BPS = 5.0
# 상승 확률이 이 값 이상이면 다음 거래일 시가에 100% 매수, 아니면 현금 100%
# 테스트 결과에 맞춰 조정하면 과최적화되므로 기본 0.5로 고정
BACKTEST_BUY_THRESHOLD = 0.50

# Python, NumPy, PyTorch의 난수 생성기를 같은 시드로 초기화
# 단, GPU 연산과 일부 라이브러리는 시드만으로 완전한 재현성이 보장되지 않음
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# CUDA 지원 GPU가 있으면 GPU, 없으면 CPU에서 신경망 계산
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("사용 장치:", device)


# ===== 2. 주가 데이터 다운로드 및 특징 생성 =====
# yfinance에서 AAPL 일봉을 다운로드
# auto_adjust=True: 분할/배당을 반영해 조정된 가격 사용
# progress=False: 다운로드 진행 표시 숨김
df = yf.download(
    TICKER,
    period=PERIOD,
    interval="1d",
    auto_adjust=True,
    multi_level_index=False,
    progress=False,
)
if df.empty:
    # 빈 데이터로 계속 진행하면 뒤에서 학습 중 오류가 생기므로 즉시 중단
    raise RuntimeError("주가 데이터를 내려받지 못했습니다.")

# 일부 yfinance 버전/옵션에서는 열 이름이 다층 인덱스로 올 수 있음
# 이 경우 Close, Open 등의 1단계 열 이름만 사용
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# 날짜를 오름차순으로 정렬하고 중복 날짜를 제거
# 시계열 학습에서는 데이터 순서가 중요
df = df.sort_index()
df = df[~df.index.duplicated(keep="first")]

# pct_change(): 전일 또는 5거래일 전 가격 대비 변화율
# 예: 100 → 102는 0.02(2%)로 표현
df["daily_return"] = df["Close"].pct_change()
df["return_5d"] = df["Close"].pct_change(5)

# 이동평균: 최근 5/20/60거래일 종가의 평균으로 추세를 표현
# rolling(60) 때문에 첫 59개 행은 MA60 계산값이 NaN
df["ma_5"] = df["Close"].rolling(5).mean()
df["ma_20"] = df["Close"].rolling(20).mean()
df["ma_60"] = df["Close"].rolling(60).mean()

# 현재 종가가 단기/중기 이동평균보다 얼마나 높거나 낮은지 비율로 표현
# 양수: 이동평균보다 종가가 높음, 음수: 이동평균보다 종가가 낮음
df["ma_gap_5"] = df["Close"] / df["ma_5"] - 1
df["ma_gap_20"] = df["Close"] / df["ma_20"] - 1

# 20일선이 60일선보다 상대적으로 높은지 판단하는 추세 특징
df["ma_cross"] = df["ma_20"] / df["ma_60"] - 1

# 최근 20거래일 일간 수익률의 표준편차를 연율화
# sqrt(252)는 1년 거래일을 약 252일로 가정하는 관례
df["volatility_20"] = df["daily_return"].rolling(20).std() * np.sqrt(252)

# 예측 목표(target): 오늘 정보를 보고 '다음 거래일'의 상승을 맞히기
# shift(-1)은 다음 행의 종가를 현재 행에 가져옴
# target_up = 1(상승), 0(보합 또는 하락), 마지막 행은 정답이 없어 NaN
# 미래 가격은 정답 라벨 생성에만 쓰고, 아래 모델 입력 features에는 포함하지 않음
df["next_return"] = df["Close"].shift(-1) / df["Close"] - 1
df["target_up"] = np.where(
    df["next_return"].notna(),
    (df["next_return"] > 0).astype(int),
    np.nan,
)

# 실제 모델에 입력되는 6가지 특징
# 원래 주가 자체가 아니라 수익률/평균 대비 격차/변동성 등 상대적 지표를 사용
features = [
    "daily_return",
    "return_5d",
    "ma_gap_5",
    "ma_gap_20",
    "ma_cross",
    "volatility_20",
]

# 무한대와 결측값(NaN)을 제거해 학습 가능한 날짜만 남김
# 이동평균 초기 구간 및 다음 날 정답을 모르는 마지막 날은 제외됨
dataset = (
    df[features + ["target_up"]]
    .replace([np.inf, -np.inf], np.nan)
    .dropna()
    .copy()
)
dataset["target_up"] = dataset["target_up"].astype(int)

print("\n종목:", TICKER)
print("사용 가능한 거래일:", len(dataset))
print("입력 특징:", features)


# ===== 3. 20거래일 시퀀스 생성 =====
# 2차원 특징 행렬: (전체 유효 거래일 수, 6개 특징)
# 정답 배열: (전체 유효 거래일 수,)
feature_data = dataset[features].to_numpy(dtype=np.float32)
target_data = dataset["target_up"].to_numpy(dtype=np.float32)

# 각 20일 구간을 하나의 샘플로 저장할 리스트
X_sequences = []
y_sequences = []
sequence_dates = []

# 최소 20일이 모이는 지점부터 마지막까지 한 날짜씩 창을 이동
# 예: 첫 샘플은 1~20일 입력 → 21일 상승 여부 정답
# 둘째 샘플은 2~21일 입력 → 22일 상승 여부 정답
for end_index in range(SEQUENCE_LENGTH - 1, len(dataset)):
    start_index = end_index - SEQUENCE_LENGTH + 1
    # 슬라이싱의 끝 인덱스는 미포함이므로 +1을 붙임
    sequence = feature_data[start_index : end_index + 1]
    # target_data[end_index]는 이 시퀀스 마지막 날의 '다음 날' 상승 여부
    target = target_data[end_index]

    X_sequences.append(sequence)
    y_sequences.append(target)
    # 예측 신호를 내는 시점의 날짜를 보관해 나중에 결과 CSV에 사용
    sequence_dates.append(dataset.index[end_index])

# LSTM/Transformer가 받는 입력 모양: (샘플 수, 20일, 6특징)
# y는 샘플마다 0 또는 1 하나이므로 모양: (샘플 수,)
X = np.array(X_sequences, dtype=np.float32)
y = np.array(y_sequences, dtype=np.float32)
# 날짜를 pandas의 날짜 인덱스 형식으로 보관
sequence_dates = pd.DatetimeIndex(sequence_dates)

print("\n시퀀스 생성 결과")
print("입력 형태:", X.shape)
print("정답 형태:", y.shape)
print("\n첫 번째 입력 시퀀스")
print("시작일:", dataset.index[0].date())
print("종료일:", dataset.index[SEQUENCE_LENGTH - 1].date())
print("예측 대상:", "종료일 다음 거래일의 상승 여부")


# ===== 4. 시간 순서대로 학습 / 검증 / 테스트 분할 =====
# 시계열에서는 데이터를 무작위로 분할하면 미래 구간이 학습에 섞일 위험이 있음
# 따라서 과거 70%를 학습, 이후 15%를 검증, 가장 최근 15%를 테스트에 배정
# int()는 소수점을 버리므로 경계에서 1개 안팎의 반올림 차이가 생길 수 있음
train_end = int(len(X) * TRAIN_RATIO)
valid_end = int(len(X) * (TRAIN_RATIO + VALID_RATIO))

# X: 입력 시퀀스 / y: 상승 여부 정답 (0 또는 1)
# 슬라이스 표기 [:끝]은 끝 인덱스를 포함하지 않음
X_train, y_train = X[:train_end], y[:train_end]
X_valid, y_valid = X[train_end:valid_end], y[train_end:valid_end]
X_test, y_test = X[valid_end:], y[valid_end:]
# 테스트 결과와 실제 날짜를 연결하기 위한 인덱스
test_dates = sequence_dates[valid_end:]

# 정규화 이후 다시 3차원 배열로 복구하기 위해 원래 모양 보관
train_shape = X_train.shape
valid_shape = X_valid.shape
test_shape = X_test.shape

# ===== 5. 입력 특징 표준화 =====
# StandardScaler는 특징별 평균을 0, 표준편차를 1 근처로 맞춤
# 서로 단위/크기가 다른 특징을 신경망이 안정적으로 학습하도록 도움
# IMPORTANT: fit은 학습 구간에만 적용 (검증/테스트 데이터를 이용하면 정보 누출)
scaler = StandardScaler()
# (샘플 수, 20일, 6특징) → (-1, 6특징)으로 펴서 특징별 평균과 표준편차 계산
# -1은 나머지 차원에서 행 수를 자동으로 추론한다는 뜻
X_train = scaler.fit_transform(X_train.reshape(-1, len(features))).reshape(train_shape)
# 검증/테스트는 학습 구간에서 구한 평균/표준편차를 그대로 사용
X_valid = scaler.transform(X_valid.reshape(-1, len(features))).reshape(valid_shape)
X_test = scaler.transform(X_test.reshape(-1, len(features))).reshape(test_shape)

# 신경망 계산에 널리 쓰는 32비트 실수형으로 통일
X_train = X_train.astype(np.float32)
X_valid = X_valid.astype(np.float32)
X_test = X_test.astype(np.float32)

print("\n데이터 분할")
print("학습:", X_train.shape)
print("검증:", X_valid.shape)
print("테스트:", X_test.shape)
print("학습 구간:", sequence_dates[0].date(), "~", sequence_dates[train_end - 1].date())
print("검증 구간:", sequence_dates[train_end].date(), "~", sequence_dates[valid_end - 1].date())
print("테스트 구간:", sequence_dates[valid_end].date(), "~", sequence_dates[-1].date())


# ===== 6. PyTorch Dataset / DataLoader 준비 =====
def create_dataset(X_data, y_data):
    """NumPy 입력과 정답을 PyTorch 텐서로 묶어 한 쌍의 학습 데이터로 변환."""
    # X_tensor: (샘플 수, 20, 6)
    X_tensor = torch.tensor(X_data, dtype=torch.float32)
    # BCE 손실함수의 모델 출력(배치 수, 1)과 정답 형태를 맞춤
    y_tensor = torch.tensor(y_data.reshape(-1, 1), dtype=torch.float32)
    # 같은 인덱스의 입력과 정답이 함께 반환되도록 설정
    return TensorDataset(X_tensor, y_tensor)


# 학습/검증/테스트에 동일한 변환 적용
train_dataset = create_dataset(X_train, y_train)
valid_dataset = create_dataset(X_valid, y_valid)
test_dataset = create_dataset(X_test, y_test)

# DataLoader: 데이터셋을 BATCH_SIZE개씩 끊어서 반복적으로 제공
# 학습은 배치 순서를 섞되 각 시퀀스 내부 20일의 순서는 유지함
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
# 검증/테스트는 예측값과 원래 샘플의 순서를 맞추기 위해 섞지 않음
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 첫 번째 배치 하나를 꺼내 입력/정답의 모양이 맞는지 확인
# 보통 sample_X: (64, 20, 6), sample_y: (64, 1)
sample_X, sample_y = next(iter(train_loader))
print("\n첫 번째 Batch")
print("입력 형태:", sample_X.shape)
print("정답 형태:", sample_y.shape)


# ===== 7. LSTM 분류 모델 정의 =====
# nn.Module을 상속하면 PyTorch가 모델의 파라미터를 자동 관리함
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=32, dropout_rate=0.2):
        # 부모 클래스(nn.Module)를 초기화해야 신경망 계층이 등록됨
        super().__init__()
        # 입력: [배치, 시간 20, 특징 6] (batch_first=True)
        # hidden_size: 각 시점에 LSTM이 생성하는 은닉 벡터의 길이
        # num_layers=1: LSTM 계층 하나만 사용
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        # 학습 중 일부 신호를 무작위로 0으로 만들어 과적합을 완화
        # model.eval() 중에는 Dropout이 자동으로 비활성화됨
        self.dropout = nn.Dropout(dropout_rate)
        # 32차원 은닉 벡터 → 상승 여부를 나타내는 점수 1개(logit)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x):
        """입력 시퀀스를 받아 상승 여부의 원시 점수(logit)를 반환."""
        # sequence_output: (배치 수, 20, hidden_size)
        # 두 번째 반환값은 최종 은닉/셀 상태이며 여기서는 사용하지 않음
        sequence_output, _ = self.lstm(x)
        # [:, -1, :] = 모든 샘플에서 마지막(20번째) 거래일의 출력만 선택
        # 마지막 출력은 앞선 거래일의 정보를 요약한 벡터
        last_output = sequence_output[:, -1, :]
        last_output = self.dropout(last_output)
        # sigmoid를 여기서 적용하지 않음: BCEWithLogitsLoss가 내부에서 처리
        return self.output(last_output)


# ===== 8. Transformer 분류 모델 정의 =====
# LSTM과 똑같은 입력/정답을 사용해 구조에 따른 성능을 비교
class TransformerClassifier(nn.Module):
    def __init__(
        self,
        input_size,
        sequence_length,
        d_model=32,
        nhead=4,
        num_layers=2,
        dropout_rate=0.2,
    ):
        super().__init__()

        # Transformer 내부 표현 크기 d_model=32로 입력 특징 6개를 확장
        # 크기를 맞춰야 위치 임베딩과 Attention 연산을 적용할 수 있음
        self.input_projection = nn.Linear(input_size, d_model)

        # Attention만 사용하면 입력의 순서 정보를 알 수 없으므로 위치 임베딩 추가
        # 크기: (1, 20, 32), 앞의 1은 모든 배치에 공유한다는 의미
        # nn.Parameter: 학습 과정에서 값이 업데이트되는 가중치로 등록
        self.position_embedding = nn.Parameter(
            torch.zeros(1, sequence_length, d_model)
        )
        # 위치 임베딩을 작은 정규분포 난수로 초기화
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)

        # TransformerEncoderLayer의 주요 설정
        # nhead=4: 32차원을 4개의 Attention head로 나눠 관계 학습
        # dim_feedforward=64: Attention 뒤의 내부 완전연결층 크기
        # activation="gelu": 내부 비선형 활성화 함수
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=64,
            dropout=dropout_rate,
            activation="gelu",
            batch_first=True,
        )
        # 위 EncoderLayer를 num_layers=2개 직렬로 연결
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # 출력 특징의 분포를 정규화해 학습 안정성을 높임
        self.normalization = nn.LayerNorm(d_model)
        # 최종 32차원 벡터 → 다음 날 상승 점수(logit) 하나
        self.output = nn.Linear(d_model, 1)

    def forward(self, x):
        """전체 시계열을 인코딩하고 마지막 시점의 표현으로 분류."""
        # [Batch, Sequence, Feature] → [Batch, Sequence, d_model]
        x = self.input_projection(x)
        # 시점별 위치 벡터를 더해 '언제' 발생한 특징인지 구별하게 함
        # :x.size(1)은 입력 시퀀스 길이만큼 위치 임베딩을 잘라 쓰는 표현
        x = x + self.position_embedding[:, : x.size(1), :]
        # Self-Attention으로 과거 20개 시점 간 관계를 계산
        encoded = self.encoder(x)
        # 마지막 거래일의 표현(배치 수, d_model)을 예측에 사용
        last_output = encoded[:, -1, :]
        last_output = self.normalization(last_output)
        # LSTM과 마찬가지로 확률이 아닌 logit 반환
        return self.output(last_output)


# ===== 9. 학습 보조 함수 =====
def count_parameters(model):
    """학습으로 갱신되는 가중치의 개수 계산 (모델 크기 비교용)."""
    # requires_grad=False로 고정된 파라미터는 학습 대상이 아니므로 제외
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def calculate_loss(model, data_loader, loss_function):
    """검증 데이터 전체의 평균 손실을 측정 (가중치 업데이트 없음)."""
    # 평가 모드: Dropout 등을 비활성화
    model.eval()
    losses = []

    # 검증에는 역전파가 필요 없으므로 그래디언트 추적을 해제해 메모리 절약
    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            # 모델이 있는 CPU/GPU로 데이터 이동
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            # 예측 logit과 실제 0/1 정답을 비교
            logits = model(batch_X)
            loss = loss_function(logits, batch_y)
            # .item(): 텐서 속 스칼라 값을 Python 숫자로 변환
            losses.append(loss.item())

    # 각 배치의 손실을 평균 내어 검증 성능을 대표하는 값 생성
    return float(np.mean(losses))


def train_model(model, model_name):
    """하나의 모델을 학습하고 가장 검증 손실이 낮은 가중치를 반환."""
    # 모델과 데이터를 모두 같은 장치에 놓아야 연산 가능
    model = model.to(device)
    # 이진 분류용 손실. sigmoid와 Binary Cross Entropy를 합친 안정적인 구현
    loss_function = nn.BCEWithLogitsLoss()
    # Adam이 기울기를 이용해 모델 가중치를 자동으로 갱신
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Epoch별 학습 손실과 검증 손실을 따로 기록해 나중에 그래프 표시
    train_losses = []
    valid_losses = []
    # 초기 최저 검증 손실을 무한대로 설정하면 첫 Epoch는 항상 개선으로 판정
    best_valid_loss = float("inf")
    # 최고 성능 시점의 가중치/시점을 보관하기 위한 변수
    best_model_state = None
    best_epoch = 0
    # 연속으로 검증 성능이 개선되지 않은 Epoch 수
    epochs_without_improvement = 0
    start_time = time.time()

    # Epoch = 학습 데이터를 한 바퀴 모두 사용하는 과정
    for epoch in range(1, MAX_EPOCHS + 1):
        # 학습 모드: Dropout 활성화, 기울기 계산 가능
        model.train()
        batch_losses = []

        # 한 번에 64개씩 mini-batch를 가져와 업데이트
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            # 이전 배치의 기울기가 누적되지 않도록 0으로 초기화
            optimizer.zero_grad()

            # 순전파: 모델이 예측 logit을 계산
            logits = model(batch_X)
            # 정답과 예측이 다를수록 손실이 커짐
            loss = loss_function(logits, batch_y)
            # 역전파: 각 가중치가 손실에 미친 영향(기울기)을 계산
            loss.backward()

            # 기울기 벡터의 크기(norm)를 1.0 이하로 제한해 폭주 방지
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # 계산된 기울기로 실제 모델 가중치를 한 번 업데이트
            optimizer.step()
            batch_losses.append(loss.item())

        # 이번 Epoch의 배치별 평균 학습 손실
        train_loss = float(np.mean(batch_losses))
        # 검증 데이터로 일반화 성능을 확인하되 여기서는 가중치를 변경하지 않음
        valid_loss = calculate_loss(model, valid_loader, loss_function)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        # 검증 손실이 이전 최고 기록보다 0.00001 이상 작아지면 개선으로 판정
        if valid_loss < best_valid_loss - 1e-5:
            best_valid_loss = valid_loss
            best_epoch = epoch
            # deepcopy로 현재 가중치의 복사본을 저장해 이후 학습과 분리
            best_model_state = copy.deepcopy(model.state_dict())
            # 개선되었으므로 Early Stopping 카운터 초기화
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # 첫 Epoch와 매 5번째 Epoch의 학습/검증 손실 출력
        if epoch == 1 or epoch % 5 == 0:
            print(
                f"{model_name:11s} | "
                f"Epoch {epoch:2d} | "
                f"Train {train_loss:.4f} | "
                f"Valid {valid_loss:.4f}"
            )

        # PATIENCE=8번 연속으로 검증 손실이 개선되지 않으면 학습 중단
        # 불필요한 학습 시간과 과적합을 줄이기 위한 기법
        if epochs_without_improvement >= PATIENCE:
            print(f"{model_name}: Early Stopping")
            break

    # 정상 데이터라면 첫 Epoch에서 최고 모델이 지정됨
    if best_model_state is None:
        raise RuntimeError(f"{model_name}의 최적 모델이 없습니다.")

    # 마지막 Epoch의 가중치가 아니라 검증 손실이 가장 낮았던 가중치로 복원
    model.load_state_dict(best_model_state)
    # 전체 학습에 걸린 초 단위 시간
    elapsed_time = time.time() - start_time
    history = {"train_loss": train_losses, "valid_loss": valid_losses}

    # 평가, 저장 및 그래프에 필요한 정보를 사전(dict)으로 전달
    return {
        "model": model,
        "history": history,
        "best_epoch": best_epoch,
        "best_valid_loss": best_valid_loss,
        "training_time": elapsed_time,
    }


# ===== 10. 모델 생성 및 학습 =====
# 두 모델을 생성할 때 시드를 다시 고정하여 초기화 차이를 줄임
# 두 모델의 구조 자체가 달라 정확히 같은 가중치로 시작하는 것은 아님
torch.manual_seed(RANDOM_SEED)
lstm_model = LSTMClassifier(
    # 입력 특징 수 6개, 내부 은닉 표현 32차원
    input_size=len(features), hidden_size=32, dropout_rate=0.2
)

torch.manual_seed(RANDOM_SEED)
transformer_model = TransformerClassifier(
    # 모델 표현 32차원, Attention 헤드 4개, Encoder 계층 2개
    input_size=len(features),
    sequence_length=SEQUENCE_LENGTH,
    d_model=32,
    nhead=4,
    num_layers=2,
    dropout_rate=0.2,
)

# 두 모델의 trainable parameter 수를 출력해 모델 복잡도 비교
print("\n모델 파라미터 수")
print("LSTM:", count_parameters(lstm_model))
print("Transformer:", count_parameters(transformer_model))

# 동일한 train_model 함수로 두 모델을 각각 학습
# 공정한 비교를 위해 동일한 입력 특징, 분할, 손실함수, 최적화 설정 사용
print("\nLSTM 학습 시작")
lstm_result = train_model(lstm_model, "LSTM")
print("\nTransformer 학습 시작")
transformer_result = train_model(transformer_model, "Transformer")


# ===== 11. 테스트 데이터 예측 =====
def predict_model(model, data_loader, threshold=0.5):
    """테스트 시퀀스별 상승 확률과 최종 0/1 예측값을 반환."""
    # 평가 모드: 학습 중에만 쓰는 Dropout 등 비활성화
    model.eval()
    # 데이터 전체의 예측을 배치마다 모아서 리스트에 저장
    probabilities = []
    predictions = []

    # 추론 단계에서는 가중치를 갱신하지 않으므로 그래디언트 계산 생략
    with torch.no_grad():
        for batch_X, _ in data_loader:
            batch_X = batch_X.to(device)
            # 원시 점수(logit)는 -무한대 ~ +무한대 값일 수 있음
            logits = model(batch_X)
            # sigmoid는 logit을 0~1 사이의 상승 확률로 변환
            batch_probabilities = torch.sigmoid(logits)
            # 상승 확률이 0.5 이상이면 1(상승), 미만이면 0(비상승)
            batch_predictions = (batch_probabilities >= threshold).int()

            # .cpu(): GPU 텐서를 CPU로 이동
            # .numpy(): NumPy 배열로 변환, .ravel(): 1차원으로 펼침
            # extend(): 배치별 결과를 전체 결과 리스트에 이어 붙임
            probabilities.extend(batch_probabilities.cpu().numpy().ravel())
            predictions.extend(batch_predictions.cpu().numpy().ravel())

    # 확률 배열과 이진 예측 배열을 각각 반환 (길이는 테스트 샘플 수)
    return np.array(probabilities), np.array(predictions)


# 학습이 끝난 두 모델에 같은 테스트 데이터 적용
# test_loader는 shuffle=False이므로 정답 y_test와 예측의 순서가 일치
lstm_probability, lstm_prediction = predict_model(lstm_result["model"], test_loader)
transformer_probability, transformer_prediction = predict_model(
    transformer_result["model"], test_loader
)


# ===== 12. 성능지표 계산 =====
def calculate_classification_metrics(actual, prediction):
    """정확도와 균형 정확도를 계산해 한국어 이름의 사전으로 반환."""
    return {
        # 전체 예측 중 맞힌 비율: (TP + TN) / 전체
        "정확도": accuracy_score(actual, prediction),
        # (상승 재현율 + 비상승 재현율) / 2
        # 한쪽 클래스가 많을 때 단순 정확도에 비해 편향을 덜 받음
        "균형 정확도": balanced_accuracy_score(actual, prediction),
    }


# 각 모델의 예측값과 테스트 구간의 실제 정답 비교
lstm_metrics = calculate_classification_metrics(y_test, lstm_prediction)
transformer_metrics = calculate_classification_metrics(y_test, transformer_prediction)

# 기준 모델(Baseline): 테스트 구간에서 더 자주 나온 클래스로만 예측한 경우
# 예: 상승 55%, 하락·보합 45%면 무조건 상승 예측 정확도는 55%
# 이 값보다 정확도가 높아야 단순 다수 클래스 예측보다 낫다고 볼 수 있음
up_ratio = y_test.mean()
baseline_accuracy = max(up_ratio, 1 - up_ratio)

# 성능 비교 표 생성: 각 행이 하나의 모델, 각 열이 하나의 평가 항목
comparison_table = pd.DataFrame(
    [
        {
            "모델": "기준 모델",
            "정확도": baseline_accuracy,
            # 두 클래스가 모두 존재하는 경우 다수 클래스만 예측하면 균형 정확도 0.5
            "균형 정확도": 0.5,
            # 기준 모델은 학습하지 않으므로 Epoch/시간 정보가 없음
            "최적 Epoch": np.nan,
            "학습 시간": np.nan,
            "파라미터 수": 0,
        },
        {
            "모델": "LSTM",
            "정확도": lstm_metrics["정확도"],
            "균형 정확도": lstm_metrics["균형 정확도"],
            # Early Stopping 동안 가장 낮은 검증 손실을 기록한 시점
            "최적 Epoch": lstm_result["best_epoch"],
            # 학습 시간(초)과 학습 가능한 가중치 수
            "학습 시간": lstm_result["training_time"],
            "파라미터 수": count_parameters(lstm_result["model"]),
        },
        {
            "모델": "Transformer",
            # LSTM과 똑같은 항목을 저장하여 표에서 나란히 비교
            "정확도": transformer_metrics["정확도"],
            "균형 정확도": transformer_metrics["균형 정확도"],
            "최적 Epoch": transformer_result["best_epoch"],
            "학습 시간": transformer_result["training_time"],
            "파라미터 수": count_parameters(transformer_result["model"]),
        },
    ]
)

# 숫자 4자리 출력 등 보기 좋은 형식으로 비교 표 출력
print("\n모델 비교")
print(
    comparison_table.to_string(
        index=False,
        formatters={
            "정확도": "{:.4f}".format,
            "균형 정확도": "{:.4f}".format,
            "학습 시간": lambda value: "-" if pd.isna(value) else f"{value:.2f}초",
        },
    )
)

# 혼동행렬: 행 = 실제 클래스, 열 = 예측 클래스 (기본 순서 0, 1)
# 왼쪽 위 TN / 오른쪽 위 FP / 왼쪽 아래 FN / 오른쪽 아래 TP
print("\nLSTM 혼동행렬")
print(confusion_matrix(y_test, lstm_prediction))
print("\nTransformer 혼동행렬")
print(confusion_matrix(y_test, transformer_prediction))

# classification_report: 정밀도(Precision), 재현율(Recall),
# F1-score 및 클래스별 샘플 수(Support)를 함께 출력
# zero_division=0: 분모가 0인 경우 경고/예외 대신 지표를 0으로 처리
print("\nLSTM 분류 보고서")
print(classification_report(y_test, lstm_prediction, digits=4, zero_division=0))
print("\nTransformer 분류 보고서")
print(
    classification_report(y_test, transformer_prediction, digits=4, zero_division=0)
)


# ===== 13. 예측 결과 및 모델 파일 저장 =====
# 테스트 구간 각 날짜의 실제 정답/두 모델의 확률/0·1 판정 결과를 표로 저장
prediction_result = pd.DataFrame(
    {
        # 실제 상승 여부
        "actual": y_test.astype(int),
        # 모델이 계산한 상승 확률 (0~1)
        "lstm_probability": lstm_probability,
        # 확률 0.5 기준 LSTM 예측 클래스
        "lstm_prediction": lstm_prediction,
        "transformer_probability": transformer_probability,
        "transformer_prediction": transformer_prediction,
    },
    # 날짜별 결과를 그래프/CSV에서 추적할 수 있도록 시퀀스 종료일 사용
    index=test_dates,
)
# CSV 첫 열(날짜 인덱스)의 이름 지정
prediction_result.index.name = "signal_date"

# CSV 저장: utf-8-sig는 Windows Excel에서 한글 열 이름이 깨질 위험을 줄임
# 파일은 실행 중인 작업 디렉터리에 생성됨
prediction_result.to_csv("week06_predictions.csv", encoding="utf-8-sig")
comparison_table.to_csv("week06_model_comparison.csv", index=False, encoding="utf-8-sig")

# torch.save: 재학습 없이 예측 모델을 불러올 수 있도록 가중치와 설정을 저장
# state_dict에는 각 레이어의 학습된 파라미터가 들어 있음
# scaler_mean/scale도 저장해야 추론 시 학습 때와 같은 표준화가 가능
torch.save(
    {
        "model_state_dict": lstm_result["model"].state_dict(),
        "features": features,
        "sequence_length": SEQUENCE_LENGTH,
        "input_size": len(features),
        "hidden_size": 32,
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
    },
    "week06_lstm_model.pt",
)

# Transformer 역시 가중치와 하이퍼파라미터/표준화 계수를 함께 저장
torch.save(
    {
        "model_state_dict": transformer_result["model"].state_dict(),
        "features": features,
        "sequence_length": SEQUENCE_LENGTH,
        "input_size": len(features),
        "d_model": 32,
        "nhead": 4,
        "num_layers": 2,
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
    },
    "week06_transformer_model.pt",
)

print("\n저장 완료")
print("- week06_predictions.csv")
print("- week06_model_comparison.csv")
print("- week06_lstm_model.pt")
print("- week06_transformer_model.pt")


# ===== 14. 학습 손실 및 테스트 예측 확률 시각화 =====
# 세로 3행, 가로 1열의 그래프를 생성 (너비 12인치 × 높이 13인치)
fig, axes = plt.subplots(3, 1, figsize=(12, 13))

# 1번 그래프: LSTM Epoch별 학습/검증 손실
# 학습 손실만 줄고 검증 손실이 다시 올라가면 과적합을 의심할 수 있음
axes[0].plot(lstm_result["history"]["train_loss"], label="Train Loss")
axes[0].plot(lstm_result["history"]["valid_loss"], label="Validation Loss")
# 가장 좋은 검증 손실을 기록한 Epoch 위치에 빨간 점선
# 그래프 x축 인덱스가 0부터 시작하므로 best_epoch에서 1을 뺌
axes[0].axvline(
    lstm_result["best_epoch"] - 1,
    color="red",
    linestyle="--",
    label="Best Epoch",
)
axes[0].set_title("LSTM Loss")
# 그래프 제목, x/y축 이름, 보조 격자 및 범례 표시
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].grid(alpha=0.3)
axes[0].legend()

# 2번 그래프: Transformer Epoch별 학습/검증 손실
# 위 LSTM과 동일한 형식으로 그려 학습 양상을 비교
axes[1].plot(transformer_result["history"]["train_loss"], label="Train Loss")
axes[1].plot(transformer_result["history"]["valid_loss"], label="Validation Loss")
axes[1].axvline(
    transformer_result["best_epoch"] - 1,
    color="red",
    linestyle="--",
    label="Best Epoch",
)
axes[1].set_title("Transformer Loss")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].grid(alpha=0.3)
axes[1].legend()

# 3번 그래프: 테스트 앞부분 최대 100개 샘플의 상승 예측 확률
# 데이터가 100개보다 적으면 있는 만큼만 선택
display_count = min(100, len(prediction_result))
display_result = prediction_result.iloc[:display_count]

# 같은 날짜에 대한 LSTM과 Transformer 상승 확률을 선으로 비교
axes[2].plot(
    display_result.index,
    display_result["lstm_probability"],
    label="LSTM",
)
axes[2].plot(
    display_result.index,
    display_result["transformer_probability"],
    label="Transformer",
)
# 실제 정답(0 또는 1)을 검은 점으로 표시
# 검은 점과 각 모델의 확률선이 일치하는 경향인지 시각적으로 확인 가능
axes[2].scatter(
    display_result.index,
    display_result["actual"],
    color="black",
    s=12,
    alpha=0.5,
    label="Actual",
)
# 예측 확률 0.5가 상승/비상승을 가르는 결정 경계
axes[2].axhline(0.5, color="red", linestyle="--", label="Threshold 0.5")
axes[2].set_title("Test Prediction Probability")
axes[2].set_ylabel("Probability")
axes[2].grid(alpha=0.3)
axes[2].legend()

# 그래프 제목/라벨이 서로 겹치지 않도록 간격 자동 조정
plt.tight_layout()
# 그래프 PNG 저장: 150 DPI, 바깥 여백 최소화
plt.savefig("week06_result.png", dpi=150, bbox_inches="tight")
# 백테스트까지 실행되도록 첫 번째 그래프는 창을 띄운 뒤 코드 진행
# 마지막 백테스트 그래프에서 plt.show()를 호출해 모든 창을 표시
plt.show(block=False)


# ===== 15. 실제 주가에 LSTM / Transformer 예측 신호 표시 =====
# 이 모델은 '다음 날 종가' 자체를 예측하는 회귀 모델이 아님
# 예측값은 다음 거래일의 상승 여부(0/1)와 상승 확률(0~1)임
# 따라서 실제 종가 위에 예측 방향을 ▲/▼로 표시하여 확인
#
# 신호는 입력 시퀀스의 마지막 거래일 '종가 시점'에 발생한다고 가정
# 예: 10월 20일 위치의 ▲는 10월 21일 상승을 예측했다는 뜻
# 차트에 표시되는 종가 선은 실제 관측 가격이며 '예측 주가'가 아님

# test_dates는 입력 시퀀스 마지막 날(신호 발생일)의 날짜
# 기존 prediction_result와 원본 df의 종가를 날짜 기준으로 결합
price_signals = prediction_result.tail(PLOT_DAYS).copy()
price_signals["Close"] = df.loc[price_signals.index, "Close"]

# LSTM과 Transformer를 각각 한 패널에 보여 차이를 비교
fig_price, price_axes = plt.subplots(
    2, 1, figsize=(15, 10), sharex=True
)

for axis, model_name, prediction_column in [
    (price_axes[0], "LSTM", "lstm_prediction"),
    (price_axes[1], "Transformer", "transformer_prediction"),
]:
    # 해당 날짜 입력으로 다음 날 상승을 예측했으면 True
    predicted_up = price_signals[prediction_column] == 1
    predicted_down = price_signals[prediction_column] == 0

    # 공통 배경: 실제 AAPL 종가(미래 가격 예측선이 아님)
    axis.plot(
        price_signals.index,
        price_signals["Close"],
        color="black",
        linewidth=1.3,
        alpha=0.75,
        label="Actual Close",
    )

    # 녹색 ▲: 해당 날짜 종가 기준으로 '다음 거래일 상승'을 예측
    axis.scatter(
        price_signals.index[predicted_up],
        price_signals.loc[predicted_up, "Close"],
        marker="^",
        s=48,
        color="green",
        alpha=0.8,
        label="Predict UP (next day)",
        zorder=3,
    )

    # 빨간색 ▼: 해당 날짜 종가 기준으로 '다음 거래일 비상승'을 예측
    axis.scatter(
        price_signals.index[predicted_down],
        price_signals.loc[predicted_down, "Close"],
        marker="v",
        s=48,
        color="red",
        alpha=0.8,
        label="Predict DOWN (next day)",
        zorder=3,
    )

    axis.set_title(f"AAPL Close with {model_name} Next-Day Direction Signals")
    axis.set_ylabel("Actual Close ($)")
    axis.grid(alpha=0.25)
    axis.legend(loc="best")

# 두 패널의 가로축을 실제 신호 발생 날짜로 표시
price_axes[1].set_xlabel("Signal Date (predicting the following trading day)")
fig_price.autofmt_xdate()
plt.tight_layout()

# 기존 학습/확률 그래프와 별도로 주가+예측신호 이미지도 저장
plt.savefig("week06_price_signals.png", dpi=150, bbox_inches="tight")
# 여기서 창이 닫히기를 기다리면 아래 백테스트 출력이 지연됨
plt.show(block=False)
print("- week06_price_signals.png")


# ===== 16. 테스트 구간의 모의투자 백테스트 =====
# 중요한 시점 구분:
#   t일 종가: LSTM/Transformer가 t+1일 상승 확률을 계산 (이때 t일 종가까지 사용)
#   t+1일 시가: 해당 확률로 매수 또는 매도 (이 시점까지는 미래 가격을 모름)
#   t+2일 시가: 보유한 1거래일 동안의 수익률을 평가하고 다음 신호를 반영
# 따라서 't일 종가에 예측하고 똑같은 종가에 체결'하는 미래정보 누출을 피함.
#
# 전략: 확률 >= 0.5이면 AAPL 100% 보유, 미만이면 100% 현금 (공매도 없음)
# 비교: 테스트 기간 처음부터 끝까지 AAPL을 보유하는 Buy & Hold
# 재투자: 매일 수익을 복리로 반영, 레버리지/세금/현금 이자는 고려하지 않음.
# 비용: 포지션을 0→1 또는 1→0으로 바꿀 때마다 1회 거래비용 적용.
#
# 참고: '다음 날 종가 상승 여부'로 학습했지만 백테스트는 다음 날 시가에 체결됨.
# 종가 방향 예측 정확도가 높아도 실제 시가 진입 수익은 다를 수 있음.


def calculate_backtest_metrics(daily_returns):
    """거래일별 순수익률로 누적 수익, 연율 수익, 샤프, MDD를 계산."""
    daily_returns = np.asarray(daily_returns, dtype=float)
    if len(daily_returns) == 0:
        raise ValueError("백테스트 수익률 배열이 비어 있습니다.")

    # 예: 수익률 +2%, -1%일 때 누적 성장배수는 1.02 * 0.99
    equity_multiple = np.cumprod(1.0 + daily_returns)
    total_return = float(equity_multiple[-1] - 1.0)

    # 1년 252거래일 가정. 실제 테스트 기간에 따라 연율화 값이 달라질 수 있음
    if equity_multiple[-1] > 0:
        cagr = float(equity_multiple[-1] ** (252 / len(daily_returns)) - 1)
    else:
        cagr = -1.0

    # 무위험수익률을 0으로 놓은 일간 Sharpe (일별 수익률 표준편차 이용)
    std = float(np.std(daily_returns, ddof=1)) if len(daily_returns) > 1 else 0
    sharpe = (
        float(np.sqrt(252) * np.mean(daily_returns) / std)
        if std > 1e-12
        else np.nan
    )

    # 최고점 대비 하락률. 초기 투자금 1.0도 최고점 후보에 포함
    equity_with_initial = np.r_[1.0, equity_multiple]
    running_peak = np.maximum.accumulate(equity_with_initial)
    drawdown = equity_with_initial / running_peak - 1.0
    max_drawdown = float(drawdown.min())  # 예: -0.2는 최대낙폭 -20%

    return {
        "누적 수익률": total_return,
        "연환산 수익률": cagr,
        "Sharpe": sharpe,
        "MDD": max_drawdown,
        "최종 자산": BACKTEST_INITIAL_CAPITAL * float(equity_multiple[-1]),
    }


def backtest_probabilities(price_data, signal_dates, probabilities, model_name):
    """테스트 확률 → 다음 날 시가 체결 → 그 다음 시가 평가.

    반환값:
        daily: 일별 포지션·수익률·자산 곡선 (인덱스는 신호 발생일)
        trades: 매수부터 매도까지 묶은 개별 거래 내역
    """
    if len(signal_dates) != len(probabilities):
        raise ValueError("날짜 개수와 모델 확률 개수가 다릅니다.")

    # 시가를 한 줄/두 줄 앞으로 당겨 t일에 t+1, t+2의 시가를 연결
    # shift(-1), shift(-2)는 '수익 평가'에만 쓰며 모델 입력에는 절대 사용 안 함
    future_prices = pd.DataFrame(index=price_data.index)
    future_prices["execution_date"] = price_data.index.to_series().shift(-1)
    future_prices["valuation_date"] = price_data.index.to_series().shift(-2)
    future_prices["entry_open"] = price_data["Open"].shift(-1)
    future_prices["exit_open"] = price_data["Open"].shift(-2)

    daily = future_prices.reindex(signal_dates).copy()
    daily.index.name = "signal_date"
    daily["up_probability"] = np.asarray(probabilities, dtype=float)

    # 마지막 1~2일은 그다음 거래일의 시가가 아직 존재하지 않을 수 있음
    # 미래 시가를 임의로 생성하지 않고 그 시점은 평가 대상에서 제외
    daily = daily.dropna(
        subset=["execution_date", "valuation_date", "entry_open", "exit_open"]
    )
    daily = daily[(daily["entry_open"] > 0) & (daily["exit_open"] > 0)].copy()
    if daily.empty:
        raise ValueError(f"{model_name}: 실제 체결 가능한 구간이 없습니다.")

    if not (
        (daily["execution_date"] > daily.index).all()
        and (daily["valuation_date"] > daily["execution_date"]).all()
    ):
        raise AssertionError("예측 날짜와 체결/평가 날짜의 선후 관계가 잘못되었습니다.")

    # 신호가 1이면 다음 시가에 전액 매수 / 0이면 주식을 보유하지 않음
    daily["position"] = (daily["up_probability"] >= BACKTEST_BUY_THRESHOLD).astype(int)
    # 전날 보유량과 오늘 보유량의 차이가 1이면 실제 주문 발생
    previous_position = daily["position"].shift(1).fillna(0).astype(int)
    daily["turnover"] = (daily["position"] - previous_position).abs()

    # 각 시가부터 다음 거래일 시가까지의 실제 AAPL 수익률
    daily["market_return"] = daily["exit_open"] / daily["entry_open"] - 1.0
    # 현금 보유 시 수익률 0 (현금 이자는 0으로 가정)
    daily["gross_return"] = daily["position"] * daily["market_return"]

    # 1bp = 0.01% = 0.0001. 비용은 매수/매도 각 1회 발생
    one_way_cost = BACKTEST_COST_BPS / 10_000.0
    daily["cost_rate"] = daily["turnover"] * one_way_cost
    # 수익률에 비용을 곱으로 적용해 복리 수익 계산 시 일관성을 유지
    daily["net_return"] = (
        (1 + daily["gross_return"]) * (1 - daily["cost_rate"]) - 1
    )

    # 마지막 평가일 시가에는 다음 매수 신호가 없으므로 남은 보유분을 청산
    # 청산 비용을 빠뜨리면 마지막 날 주식을 계속 들고 있는 전략이 유리해짐
    if daily["position"].iloc[-1] == 1:
        last_idx = daily.index[-1]
        daily.loc[last_idx, "net_return"] = (
            (1 + daily.loc[last_idx, "net_return"]) * (1 - one_way_cost) - 1
        )
        daily.loc[last_idx, "cost_rate"] += one_way_cost

    # Buy & Hold: 첫 거래일 시가에 1회 매수해 마지막 평가일 시가에 1회 매도
    # 모델과 정확히 같은 시가 구간 / 동일한 거래 비용을 사용하여 비교
    daily["buy_hold_net_return"] = daily["market_return"].copy()
    first_idx, last_idx = daily.index[0], daily.index[-1]
    daily.loc[first_idx, "buy_hold_net_return"] = (
        (1 + daily.loc[first_idx, "buy_hold_net_return"]) * (1 - one_way_cost) - 1
    )
    daily.loc[last_idx, "buy_hold_net_return"] = (
        (1 + daily.loc[last_idx, "buy_hold_net_return"]) * (1 - one_way_cost) - 1
    )

    # 일간 수익률을 누적 곱해 자산 가치로 환산
    daily["strategy_equity"] = BACKTEST_INITIAL_CAPITAL * (1 + daily["net_return"]).cumprod()
    daily["buy_hold_equity"] = BACKTEST_INITIAL_CAPITAL * (
        1 + daily["buy_hold_net_return"]
    ).cumprod()

    # 매수부터 매도까지를 '거래 1회'로 집계. 매도는 다음 번 현금 신호 시가
    # 또는 테스트 마지막 평가 시가에 발생함
    trade_records = []
    open_trade = None
    for _, row in daily.iterrows():
        position = int(row["position"])
        if position == 1 and open_trade is None:
            open_trade = {
                "모델": model_name,
                "매수일": row["execution_date"],
                "매수가": float(row["entry_open"]),
                "보유 거래일": 0,
            }
        elif position == 0 and open_trade is not None:
            exit_price = float(row["entry_open"])
            trade_records.append(
                {
                    **open_trade,
                    "매도일": row["execution_date"],
                    "매도가": exit_price,
                    "거래 수익률": (
                        exit_price / open_trade["매수가"] * (1 - one_way_cost) ** 2 - 1
                    ),
                }
            )
            open_trade = None

        if position == 1:
            # 이 행은 실제로 한 번의 시가→다음 시가 보유 구간을 의미
            open_trade["보유 거래일"] += 1

    # 열린 거래가 남았다면 마지막 평가 가능한 시가에 실제 청산 처리
    if open_trade is not None:
        final_row = daily.iloc[-1]
        exit_price = float(final_row["exit_open"])
        trade_records.append(
            {
                **open_trade,
                "매도일": final_row["valuation_date"],
                "매도가": exit_price,
                "거래 수익률": (
                    exit_price / open_trade["매수가"] * (1 - one_way_cost) ** 2 - 1
                ),
            }
        )

    trade_columns = [
        "모델", "매수일", "매수가", "매도일", "매도가", "보유 거래일", "거래 수익률"
    ]
    trades = pd.DataFrame(trade_records, columns=trade_columns)
    return daily, trades


# 동일한 테스트 날짜와 시장 데이터에서 두 신경망의 매매 결과를 평가
lstm_bt, lstm_trades = backtest_probabilities(
    df, test_dates, lstm_probability, "LSTM"
)
transformer_bt, transformer_trades = backtest_probabilities(
    df, test_dates, transformer_probability, "Transformer"
)

# 두 모델은 같은 날짜의 예측 확률을 사용하므로 평가 날짜도 일치해야 함
if not lstm_bt.index.equals(transformer_bt.index):
    raise AssertionError("LSTM과 Transformer의 백테스트 날짜가 일치하지 않습니다.")

# 누적 수익률, 연율 수익률, 샤프지수, 최대낙폭 및 거래 횟수 비교
backtest_rows = []
for model_name, daily, trades in [
    ("LSTM", lstm_bt, lstm_trades),
    ("Transformer", transformer_bt, transformer_trades),
]:
    metrics = calculate_backtest_metrics(daily["net_return"])
    metrics["모델"] = model_name
    metrics["완료 거래 수"] = len(trades)
    # 거래 단위 승률: 수수료/슬리피지를 제외한 순수익률이 0보다 큰 거래 비율
    metrics["거래 승률"] = float((trades["거래 수익률"] > 0).mean()) if len(trades) else np.nan
    metrics["시장 노출 비율"] = float(daily["position"].mean())
    backtest_rows.append(metrics)

# 주식 보유 자체의 효과와 신경망 예측 신호의 효과를 구분하기 위한 기준
baseline_metrics = calculate_backtest_metrics(lstm_bt["buy_hold_net_return"])
baseline_metrics.update(
    {
        "모델": "AAPL Buy & Hold",
        "완료 거래 수": 1,
        "거래 승률": float(baseline_metrics["누적 수익률"] > 0),
        "시장 노출 비율": 1.0,
    }
)
backtest_rows.append(baseline_metrics)
backtest_summary = pd.DataFrame(backtest_rows).set_index("모델")

print("\n" + "=" * 65)
print("테스트 구간 백테스트 (다음 시가에 체결, 롱/현금)")
print("=" * 65)
print(f"평가 시작: {lstm_bt['execution_date'].iloc[0].date()}")
print(f"평가 종료: {lstm_bt['valuation_date'].iloc[-1].date()}")
print(f"평가 거래일 수: {len(lstm_bt)}")
print(f"초기 자산: ${BACKTEST_INITIAL_CAPITAL:,.2f}")
print(f"편도 거래 비용: {BACKTEST_COST_BPS:.2f}bp")
print(f"매수 기준: 상승 확률 >= {BACKTEST_BUY_THRESHOLD:.2f}")

# 퍼센트/달러 단위로 사람이 읽기 편하게 표시 (원본 CSV에는 실수 저장)
summary_display = backtest_summary.copy()
for col in ["누적 수익률", "연환산 수익률", "MDD", "거래 승률", "시장 노출 비율"]:
    summary_display[col] = summary_display[col].map(
        lambda val: "-" if pd.isna(val) else f"{val * 100:.2f}%"
    )
summary_display["Sharpe"] = summary_display["Sharpe"].map(
    lambda val: "-" if pd.isna(val) else f"{val:.2f}"
)
summary_display["최종 자산"] = summary_display["최종 자산"].map(
    lambda val: f"${val:,.2f}"
)
print(summary_display.to_string())

# 테스트 구간 '일별 상세'에는 신호 날짜 / 체결 날짜 / 실제 가격 / 수익률 포함
# Excel에서 전략 간 결과를 나란히 비교할 수 있도록 하나의 CSV로 저장
backtest_daily = lstm_bt[
    ["execution_date", "valuation_date", "entry_open", "exit_open", "market_return"]
].copy()
for model_name, daily in [("lstm", lstm_bt), ("transformer", transformer_bt)]:
    backtest_daily[f"{model_name}_probability"] = daily["up_probability"]
    backtest_daily[f"{model_name}_position"] = daily["position"]
    backtest_daily[f"{model_name}_cost_rate"] = daily["cost_rate"]
    backtest_daily[f"{model_name}_net_return"] = daily["net_return"]
    backtest_daily[f"{model_name}_equity"] = daily["strategy_equity"]
backtest_daily["buy_hold_net_return"] = lstm_bt["buy_hold_net_return"]
backtest_daily["buy_hold_equity"] = lstm_bt["buy_hold_equity"]

backtest_trades = pd.concat([lstm_trades, transformer_trades], ignore_index=True)
backtest_daily.to_csv("week06_backtest_daily.csv", encoding="utf-8-sig")
backtest_trades.to_csv("week06_backtest_trades.csv", index=False, encoding="utf-8-sig")
backtest_summary.to_csv("week06_backtest_summary.csv", encoding="utf-8-sig")

# ===== 17. 백테스트 누적 자산과 낙폭(MDD) 시각화 =====
fig_backtest, bt_axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
portfolio_lines = [
    ("LSTM", lstm_bt["strategy_equity"]),
    ("Transformer", transformer_bt["strategy_equity"]),
    ("AAPL Buy & Hold", lstm_bt["buy_hold_equity"]),
]
plot_dates = lstm_bt["valuation_date"]

for name, portfolio in portfolio_lines:
    # 테스트 시작 투자금에서 실제 매매/보유로 자산이 어떻게 변했는지 확인
    bt_axes[0].plot(plot_dates, portfolio.to_numpy(), label=name)
    # 고점 갱신 이후의 자산 하락폭을 백분율로 표시
    peaks = np.maximum.accumulate(np.r_[BACKTEST_INITIAL_CAPITAL, portfolio.to_numpy()])[1:]
    drawdown_percent = (portfolio.to_numpy() / peaks - 1) * 100
    bt_axes[1].plot(plot_dates, drawdown_percent, label=name)

bt_axes[0].axhline(BACKTEST_INITIAL_CAPITAL, color="gray", linestyle="--", alpha=0.7)
bt_axes[0].set_title("Out-of-Sample Backtest: LSTM vs Transformer vs AAPL Buy & Hold")
bt_axes[0].set_ylabel("Portfolio Value ($)")
bt_axes[0].grid(alpha=0.3)
bt_axes[0].legend()

bt_axes[1].set_title("Drawdown from Previous Peak")
bt_axes[1].set_ylabel("Drawdown (%)")
bt_axes[1].set_xlabel("Valuation Date (next available market open)")
bt_axes[1].grid(alpha=0.3)
bt_axes[1].legend()

fig_backtest.autofmt_xdate()
plt.tight_layout()
plt.savefig("week06_backtest_equity.png", dpi=150, bbox_inches="tight")
plt.show()

print("\n백테스트 파일 저장 완료")
print("- week06_backtest_daily.csv          (일별 포지션·가격·수익률)")
print("- week06_backtest_trades.csv         (모델별 개별 거래 내역)")
print("- week06_backtest_summary.csv        (수익률·Sharpe·MDD 비교)")
print("- week06_backtest_equity.png         (누적 자산·낙폭 그래프)")
