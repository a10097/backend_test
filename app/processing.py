#processing.py
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, List, Tuple, Any
from keras.models import load_model as tf_load_model
from sklearn.preprocessing import RobustScaler, PowerTransformer # 타입 힌팅용

# --- 아티팩트 경로 설정 ---
BASE_DIR = Path(__file__).resolve().parent.parent # mealfit_prediction_api/ 디렉토리
ARTIFACTS_DIR = BASE_DIR / "trained_artifacts"

MODEL_PATH = ARTIFACTS_DIR / "mealfit_tf_dnn_model.keras"
SCALER_PATH = ARTIFACTS_DIR / "robust_scaler_final.pkl"
POWER_TRANSFORMER_PATH = ARTIFACTS_DIR / "power_transformer_final.pkl"
FINAL_FEATURES_LIST_PATH = ARTIFACTS_DIR / "final_feature_list.pkl"
# 선택 사항: NaN 대체를 위해 학습 데이터의 중간값을 저장한 경우
# MEDIANS_PATH = ARTIFACTS_DIR / "training_medians.pkl"

# --- 로드된 아티팩트를 위한 전역 캐시 ---
# 매 요청마다 다시 로드하는 것을 방지
_artifacts_cache: Dict[str, Any] = {}

def load_artifacts_once():
    """모델, 전처리기, 특성 목록을 한 번만 로드합니다."""
    if not _artifacts_cache:
        print("아티팩트를 처음으로 로드합니다...")
        try:
            _artifacts_cache["model"] = tf_load_model(MODEL_PATH)
            _artifacts_cache["scaler"] = joblib.load(SCALER_PATH)
            _artifacts_cache["power_transformer"] = joblib.load(POWER_TRANSFORMER_PATH)
            _artifacts_cache["final_feature_list"] = joblib.load(FINAL_FEATURES_LIST_PATH)
            # 선택 사항: 중간값을 저장한 경우 로드
            # _artifacts_cache["medians"] = joblib.load(MEDIANS_PATH)
            model_name = getattr(_artifacts_cache["model"], 'name', 'TensorFlow 모델')
            print(f"{model_name} 로드 완료.")
            print(f"스케일러 로드 완료: {type(_artifacts_cache['scaler'])}")
            print(f"PowerTransformer 로드 완료: {type(_artifacts_cache['power_transformer'])}")
            print(f"최종 특성 목록 로드 완료 (개수: {len(_artifacts_cache['final_feature_list'])})")
        except FileNotFoundError as e:
            print(f"아티팩트 로드 오류: {e}. 모든 아티팩트 파일이 {ARTIFACTS_DIR}에 있는지 확인하세요.")
            raise
        except Exception as e:
            print(f"아티팩트 로드 중 예상치 못한 오류 발생: {e}")
            raise
    return _artifacts_cache

# --- 파생 변수 생성 함수 (script2_preprocessing.py에서 복사 및 조정) ---
def create_derived_features(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()
    # 연산 전 필요한 컬럼 존재 여부 확인
    if 'recipe_base_protein' in df.columns and 'recipe_base_fat' in df.columns:
        df["prot_fat_ratio"] = df["recipe_base_protein"] / (df["recipe_base_fat"] + 1e-6)
    if 'recipe_base_kcal' in df.columns and 'budget_mid' in df.columns:
        df["kcal_budget_ratio"] = df["recipe_base_kcal"] / (df["budget_mid"] + 1e-6)
    if 'recipe_base_kcal' in df.columns:
        df["log_kcal"] = np.log1p(df["recipe_base_kcal"])
    return df

# app/processing.py

# ... (파일 상단 임포트 및 경로 설정은 그대로) ...

# app/processing.py

# ... (파일 상단 임포트, 경로 설정, load_artifacts_once, create_derived_features 함수는 동일) ...

def preprocess_input_for_prediction(input_data: pd.DataFrame) -> pd.DataFrame:
    artifacts = load_artifacts_once()
    scaler: RobustScaler = artifacts["scaler"]
    power_transformer: PowerTransformer = artifacts["power_transformer"]
    final_feature_list_from_training: List[str] = artifacts["final_feature_list"]

    print("--- preprocess_input_for_prediction 시작 ---")
    print(f"입력 데이터 컬럼 ({len(input_data.columns)}개): {input_data.columns.tolist()}")

    df_processed = input_data.copy()
    df_processed = create_derived_features(df_processed)
    print(f"파생 변수 생성 후 컬럼 ({len(df_processed.columns)}개): {df_processed.columns.tolist()}")

    # --- PowerTransform 적용 ---
    # PowerTransformer가 학습 시 사용했던 컬럼 이름들을 가져옵니다.
    # 이 컬럼들은 파생 변수가 포함된 상태의 이름이어야 합니다.
    if hasattr(power_transformer, 'feature_names_in_'):
        cols_for_power_transform = power_transformer.feature_names_in_
        print(f"PowerTransformer가 기대하는 컬럼 ({len(cols_for_power_transform)}개): {cols_for_power_transform.tolist()}")

        # df_processed에 PowerTransformer가 기대하는 컬럼이 모두 있는지 확인
        missing_for_pt = [col for col in cols_for_power_transform if col not in df_processed.columns]
        if missing_for_pt:
            print(f"경고: df_processed에 PowerTransformer가 기대하는 다음 컬럼이 누락되었습니다: {missing_for_pt}")
            # 실제 운영에서는 이 부분에서 오류를 발생시키거나 적절히 처리해야 합니다.
            # 임시로 누락된 컬럼을 0으로 채워넣어 볼 수 있습니다. (권장하지는 않음)
            for col in missing_for_pt:
                df_processed[col] = 0
            print(f"PowerTransformer 누락 컬럼 0으로 채운 후 컬럼: {df_processed.columns.tolist()}")

        # PowerTransformer가 기대하는 컬럼만, 그리고 그 순서대로 전달합니다.
        # 변환 결과는 원래 컬럼 이름을 유지하도록 DataFrame으로 만듭니다.
        df_to_transform_pt = df_processed[cols_for_power_transform]
        transformed_data_pt = power_transformer.transform(df_to_transform_pt)
        df_transformed_pt = pd.DataFrame(transformed_data_pt, columns=cols_for_power_transform,
                                         index=df_processed.index)

        # 변환된 컬럼만 df_processed에 업데이트합니다.
        for col in cols_for_power_transform:
            df_processed[col] = df_transformed_pt[col]
        print(f"PowerTransform 적용 후 컬럼 ({len(df_processed.columns)}개): {df_processed.columns.tolist()}")
    else:
        print("경고: PowerTransformer에 feature_names_in_ 속성이 없습니다. PowerTransform 단계를 건너뜁니다 (또는 다른 방식으로 처리 필요).")

    # --- RobustScaler 적용 ---
    # RobustScaler가 학습 시 사용했던 컬럼 이름들을 가져옵니다.
    # 이 컬럼들은 PowerTransform이 적용된 후의 상태이며, final_feature_list_from_training과 동일해야 합니다.
    if hasattr(scaler, 'feature_names_in_'):
        cols_for_scaling = scaler.feature_names_in_
        print(f"RobustScaler가 기대하는 컬럼 ({len(cols_for_scaling)}개): {cols_for_scaling.tolist()}")
        print(
            f"모델이 기대하는 최종 특성 목록 (final_feature_list_from_training) ({len(final_feature_list_from_training)}개): {final_feature_list_from_training}")

        # RobustScaler가 기대하는 컬럼과 final_feature_list가 동일한지 확인 (디버깅용)
        if not np.array_equal(np.sort(cols_for_scaling), np.sort(final_feature_list_from_training)):
            print(
                f"경고: RobustScaler의 feature_names_in_ ({cols_for_scaling.tolist()}) 와 final_feature_list ({final_feature_list_from_training}) 가 일치하지 않을 수 있습니다. 확인 필요!")
            # 이 경우, final_feature_list_from_training을 기준으로 스케일링 대상을 정하는 것이 더 안전할 수 있습니다.
            # cols_for_scaling = final_feature_list_from_training # 강제 할당 (주의!)

        # df_processed에 RobustScaler가 기대하는 컬럼이 모두 있는지 확인
        missing_for_scaler = [col for col in cols_for_scaling if col not in df_processed.columns]
        if missing_for_scaler:
            print(f"경고: df_processed에 RobustScaler가 기대하는 다음 컬럼이 누락되었습니다: {missing_for_scaler}")
            for col in missing_for_scaler:  # 임시 조치
                df_processed[col] = 0
            print(f"RobustScaler 누락 컬럼 0으로 채운 후 컬럼: {df_processed.columns.tolist()}")

        # RobustScaler가 기대하는 컬럼만, 그리고 그 순서대로 전달합니다.
        df_to_scale = df_processed[cols_for_scaling]
        scaled_data = scaler.transform(df_to_scale)
        df_scaled = pd.DataFrame(scaled_data, columns=cols_for_scaling, index=df_processed.index)
        print(f"RobustScaler 적용 후 컬럼 (df_scaled) ({len(df_scaled.columns)}개): {df_scaled.columns.tolist()}")
    else:
        print("경고: RobustScaler에 feature_names_in_ 속성이 없습니다. RobustScaler 단계를 건너뜁니다 (또는 다른 방식으로 처리 필요).")
        df_scaled = df_processed.copy()  # 스케일링 건너뛰고 원본 사용 (문제가 될 수 있음)

    # --- 최종 컬럼 선택 및 순서 정렬 ---
    try:
        # final_feature_list_from_training에 있는 순서대로 컬럼을 선택하고 정렬합니다.
        # 이 때 df_scaled에는 final_feature_list_from_training의 모든 컬럼이 존재해야 합니다.
        missing_in_df_scaled_final = [col for col in final_feature_list_from_training if col not in df_scaled.columns]
        if missing_in_df_scaled_final:
            print(f"심각: 최종 정렬 전, df_scaled에 모델이 기대하는 다음 컬럼들이 누락되었습니다: {missing_in_df_scaled_final}")
            for col in missing_in_df_scaled_final:  # 임시 조치
                df_scaled[col] = 0
            print(f"최종 정렬 전 누락 컬럼 임시 조치 후 df_scaled 컬럼: {df_scaled.columns.tolist()}")

        df_final_for_model = df_scaled[final_feature_list_from_training]
        print(
            f"최종 정렬 후 컬럼 (df_final_for_model) ({len(df_final_for_model.columns)}개): {df_final_for_model.columns.tolist()}")
    except KeyError as e:
        # 이 오류는 df_scaled에 final_feature_list_from_training에 있는 컬럼 중 일부가 누락되었을 때 발생합니다.
        missing_cols_at_final = set(final_feature_list_from_training) - set(df_scaled.columns)
        print(f"처리 중 오류 (KeyError): 최종 모델 입력 데이터 생성 중 컬럼 누락 또는 불일치: {missing_cols_at_final}")
        print(f"df_scaled의 현재 컬럼: {df_scaled.columns.tolist()}")
        print(f"모델이 기대하는 컬럼 (final_feature_list.pkl): {final_feature_list_from_training}")
        raise ValueError(f"모델 입력에 필요한 일부 특성이 준비되지 않았습니다 (KeyError). 서버 로그를 확인하세요. 원인: {e}")
    except Exception as ex:
        print(f"최종 정렬 중 예상치 못한 오류: {ex}")
        raise ValueError(f"데이터 처리 중 예상치 못한 오류 발생 (최종 정렬 단계). 서버 로그를 확인하세요.")

    if df_final_for_model.isnull().values.any():
        print(f"경고: 모델로 전송되는 데이터에 NaN 값이 발견되었습니다. 0으로 대체합니다.")
        print(df_final_for_model[df_final_for_model.isnull().any(axis=1)])
        df_final_for_model = df_final_for_model.fillna(0)

    print("--- preprocess_input_for_prediction 종료 ---")
    return df_final_for_model