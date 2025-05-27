# recommendation_api.py
from datetime import datetime
import mysql.connector
import pandas as pd
import numpy as np
import tensorflow as tf
import jwt
import os
from fastapi import APIRouter, HTTPException, Depends, Header, Body
from typing import Dict, Any
from pydantic import BaseModel
from typing import List
from enum import Enum

from . import models
from . import processing

#터미널 체크용
import pprint

router = APIRouter()

# JWT 설정 (main.py와 동일)
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "supersecret")
ALGORITHM = "HS256"


class IngredientRequest(BaseModel):
    ingredient_name: str
    isPreferred: bool
    isDisliked: bool

class NutritionLevel(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"

class CalorieLevel(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class RecipeRecommendationRequest(BaseModel):
    budget_mid: float
    ingredients: List[IngredientRequest]  # 재료 정보 추가
    nutrition_level: NutritionLevel 
    calorie_level: CalorieLevel 



def get_artifacts() -> Dict[str, Any]:
    try:
        return processing.load_artifacts_once()
    except Exception as e:
        raise RuntimeError(f"아티팩트 로드 실패: {e}") from e

def get_user_profile(user_id: int):
    """user_profiles 테이블에서 사용자 프로필 정보 조회"""
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="1509",
        database="test"
    )
    cursor = conn.cursor(dictionary=True)
    cursor.execute("""
        SELECT height_cm, weight_kg, birth_year, gender, pal_value, tdee,
               target_protein_total_g, target_fat_total_g
        FROM user_profiles
        WHERE user_id = %s
        ORDER BY created_at DESC
        LIMIT 1
    """, (user_id,))
    row = cursor.fetchone()
    cursor.close()
    conn.close()
    
    if not row:
        return None

    # age 계산
    now_year = datetime.now().year
    age = now_year - row["birth_year"]

    # gender 변환 (M=0, F=1)
    gender_int = 0 if row["gender"] == "M" else 1

    return {
        "gender": gender_int,
        "age": age,
        "height_cm": float(row["height_cm"]),
        "weight_kg": float(row["weight_kg"]),
        "pal_value": float(row["pal_value"]),
        "tdee": row["tdee"],
        "target_protein_total_g": float(row["target_protein_total_g"]),
        "target_fat_total_g": float(row["target_fat_total_g"])
    }

@router.post("/predict_top10/", summary="DB 내 모든 레시피에 대해 예측하고 상위 10개 반환")
async def predict_top10_recipes(
    request: RecipeRecommendationRequest = Body(...),  
    Authorization: str = Header(...),
    artifacts: Dict[str, Any] = Depends(get_artifacts)
):
    try:
        # 1. JWT 토큰에서 user_id 추출
        token = Authorization.split(" ")[1]
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            user_email = payload.get("sub")
        except jwt.PyJWTError:
            raise HTTPException(status_code=401, detail="유효하지 않은 토큰입니다.")
        
        # 2. user_email로 user_id 조회
        conn = mysql.connector.connect(
            host="localhost",
            user="root", 
            password="1509",
            database="test"
        )
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT id FROM users WHERE email = %s", (user_email,))
        result = cursor.fetchone()
        if not result:
            cursor.close()
            conn.close()
            raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다.")
        
        user_id = result["id"]
        cursor.close()
        conn.close()

        # 3. user_profiles에서 사용자 정보 조회
        user_profile = get_user_profile(user_id)
        if not user_profile:
            raise HTTPException(status_code=404, detail="사용자 프로필이 없습니다.")

        # 4.1 DB에서 메뉴별 재료정보 집계
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="1509",
            database="test"
        )
        cursor = conn.cursor(dictionary=True)
        ingredient_query = """
        SELECT 
            menu_일련번호 AS menu_id,
            메뉴명 AS recipe_name,
            GROUP_CONCAT(재료정보 SEPARATOR ',') AS ingredients
        FROM ingredients
        GROUP BY menu_일련번호, 메뉴명
        """
        cursor.execute(ingredient_query)
        ingredient_rows = cursor.fetchall()
        ingredient_df = pd.DataFrame(ingredient_rows)
        cursor.close()
        conn.close()

    # 4.2 DB에서 menus 테이블의 영양 정보 가져오기
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="1509",
            database="test"
        )
        cursor = conn.cursor(dictionary=True)
        nutrition_query = """
        SELECT `일련번호` AS menu_id, `메뉴명` AS recipe_name,
            `열량` AS recipe_base_kcal, `탄수화물` AS recipe_base_carb,
            `단백질` AS recipe_base_protein, `지방` AS recipe_base_fat,
            `나트륨` AS recipe_base_sodium
        FROM menus LIMIT 155;
        """
        cursor.execute(nutrition_query)
        nutrition_rows = cursor.fetchall()
        nutrition_df = pd.DataFrame(nutrition_rows)
        cursor.close()
        conn.close()

        # 4.3 병합   
        full_recipe_df = pd.merge(nutrition_df, ingredient_df, on=["menu_id", "recipe_name"], how="left")

        # 4.4 재료명 리스트 추출
        def parse_ingredient_names(ingredients_str):
            if not isinstance(ingredients_str, str):
                return []
            return [item.split()[0] for item in ingredients_str.split(',') if item.strip()]
        full_recipe_df["ingredient_names"] = full_recipe_df["ingredients"].apply(parse_ingredient_names)
                    
        # 5. 사용자 정보 + 예산 값으로 레시피 feature 계산
        def compute_recipe_features(recipe_df: pd.DataFrame, user_info: dict, budget: float,
                                    ingredients: list, nutrition_level: NutritionLevel, 
                                    calorie_level: CalorieLevel) -> list:
            features_list = []


            #사용자 개인정보 추출
            user_gender = user_info["gender"]  # 0=남성, 1=여성
            user_age = user_info["age"]
            user_weight = user_info["weight_kg"]
            user_height = user_info["height_cm"]
            user_pal = user_info["pal_value"]
            user_tdee = user_info["tdee"]
            user_target_protein = user_info["target_protein_total_g"]
            user_target_fat = user_info["target_fat_total_g"]

            # ✅ ingredient_data에서 재료 가격 정보 미리 조회
            def get_ingredient_prices():
                """ingredient_data 테이블에서 모든 재료의 가격 정보를 가져와서 딕셔너리로 반환"""
                conn = mysql.connector.connect(
                    host="localhost",
                    user="root", 
                    password="1509",
                    database="test"
                )
                cursor = conn.cursor(dictionary=True)
                        
                # ✅ NULL 값이 있는 행을 제외하고 조회
                cursor.execute("""
                    SELECT ingredient_name, ingredient_price
                    FROM ingredient_data
                    WHERE ingredient_price IS NOT NULL
                """)   

                price_data = cursor.fetchall()
                cursor.close()
                conn.close()

                # ✅ 추가적인 None 체크와 함께 딕셔너리 생성
                price_dict = {}
                for row in price_data:
                    ingredient_name = row['ingredient_name']
                    ingredient_price = row['ingredient_price']
                    
                    if ingredient_name:  # 재료명이 있는 경우만
                        if ingredient_price is not None:
                            try:
                                price_dict[ingredient_name] = float(ingredient_price)
                            except (ValueError, TypeError):
                                price_dict[ingredient_name] = 0.0  # 변환 실패 시 기본값
                        else:
                            price_dict[ingredient_name] = 0.0  # NULL 값일 때 기본값
                
                print(f"가격 정보가 로드된 재료 수: {len(price_dict)}")
                return price_dict

            # 재료 가격 정보 한 번만 조회
            ingredient_prices = get_ingredient_prices()

            # ✅ 개인별 맞춤 영양 기준 설정
            def get_personalized_nutrition_targets(gender, age, weight, tdee, pal_value):
                """성별, 연령, 활동량에 따른 개인별 영양 기준 설정"""
        
                #  1끼 기준 칼로리 (TDEE의 1/3)
                meal_kcal_target = tdee / 3
        
                # 성별별 기본 영양 기준
                if gender == 0:  # 남성
                    base_protein_ratio = 1.0 if age < 50 else 1.2  # 50세 이상 단백질 증가
                    sodium_limit = 2300 if age < 65 else 2000      # 고령자 나트륨 제한
                    fat_limit_percent = 30                          # 지방 칼로리 비율 상한
                else:  # 여성
                    base_protein_ratio = 0.9 if age < 50 else 1.1  # 폐경 후 단백질 증가
                    sodium_limit = 2300 if age < 65 else 1800      # 여성 고령자 나트륨 더 제한
                    fat_limit_percent = 25                          # 여성 지방 비율 더 제한
        
                    # 활동량(PAL)에 따른 조정
                if pal_value >= 1.9:      # 매우 활동적
                    protein_multiplier = 1.3
                    carb_importance = 1.2
                elif pal_value >= 1.6:   # 활동적
                    protein_multiplier = 1.15
                    carb_importance = 1.1
                else:                    # 저활동
                    protein_multiplier = 1.0
                    carb_importance = 0.9
        
                # 연령별 가중치
                if age >= 65:
                    # 고령자: 단백질, 칼슘 중요도 증가, 나트륨 제한 강화
                    age_weights = {"protein": 0.4, "kcal": 0.2, "fat": 0.2, "sodium": 0.2}
                elif age >= 50:
                 # 중년: 균형잡힌 영양, 대사질환 예방
                    age_weights = {"protein": 0.3, "kcal": 0.3, "fat": 0.2, "sodium": 0.2}
                elif age >= 30:
                    # 성인: 체중관리, 영양균형
                    age_weights = {"protein": 0.25, "kcal": 0.35, "fat": 0.2, "sodium": 0.2}
                else:
                    # 청년: 칼로리, 단백질 중시
                    age_weights = {"protein": 0.3, "kcal": 0.4, "fat": 0.15, "sodium": 0.15}
        
                return {
                    "meal_kcal_target": meal_kcal_target,
                    "protein_target_per_meal": (user_target_protein / 3) * base_protein_ratio * protein_multiplier,
                    "fat_limit_per_meal": (meal_kcal_target * fat_limit_percent / 100) / 9,  # g 단위
                    "sodium_limit_per_meal": sodium_limit / 3,  # 1끼 기준
                    "carb_importance": carb_importance,
                    "age_weights": age_weights
                }

            nutrition_targets = get_personalized_nutrition_targets(
                user_gender, user_age, user_weight, user_tdee, user_pal
            )

            # ▶▶▶ 영양/칼로리 수준에 따른 가중치 매핑 ◀◀◀
            level_mapping = {
                "low": 0.3,
                "medium": 0.5,
                "high": 0.7
            }
            pref_nutrition_score = level_mapping[nutrition_level.value]
            pref_kcal_target_accuracy = level_mapping[calorie_level.value]
            pref_macro_target_accuracy = level_mapping[nutrition_level.value]

             # 재료 정보 처리
            owned_ingredients = [ing.ingredient_name for ing in ingredients if not ing.isDisliked]
            preferred_ingredients = [ing.ingredient_name for ing in ingredients if ing.isPreferred]
            disliked_ingredients = [ing.ingredient_name for ing in ingredients if ing.isDisliked]

            # ✅ 디버깅: 플러터에서 받은 재료 정보 출력
            print(f"\n=== 개인별 맞춤 영양 기준 ===")
            print(f"사용자: {user_age}세, {'남성' if user_gender == 0 else '여성'}, {user_weight}kg, PAL: {user_pal}")
            print(f"1끼 칼로리 목표: {nutrition_targets['meal_kcal_target']:.0f}kcal")
            print(f"1끼 단백질 목표: {nutrition_targets['protein_target_per_meal']:.1f}g")
            print(f"전체 재료(비선호 제외): {owned_ingredients}")
            print(f"선호 재료: {preferred_ingredients}")
            print(f"비선호 재료: {disliked_ingredients}")
            print(f"ingredient_data에서 조회된 재료 가격 개수: {len(ingredient_prices)}")

            # 총 재료 개수 계산 (피처에서 사용)
            total_owned = len(owned_ingredients)
            total_preferred = len(preferred_ingredients)
            total_disliked = len(disliked_ingredients)

            for idx, row in recipe_df.iterrows():
                kcal = float(row["recipe_base_kcal"])
                carb = float(row["recipe_base_carb"])
                protein = float(row["recipe_base_protein"])
                fat = float(row["recipe_base_fat"])
                sodium = float(row["recipe_base_sodium"])
                # ✅ 개인별 맞춤 영양 점수 계산

                # 1. 개인별 단백질 점수
                protein_score = min(protein / nutrition_targets["protein_target_per_meal"], 1.0)

                # 2. 개인별 칼로리 점수 (TDEE 기반)
                ideal_kcal = nutrition_targets["meal_kcal_target"]
                kcal_diff_ratio = abs(kcal - ideal_kcal) / ideal_kcal
                kcal_score = max(0.0, 1.0 - kcal_diff_ratio)

                # 3. 개인별 지방 점수
                fat_limit = nutrition_targets["fat_limit_per_meal"]
                fat_score = max(0.0, 1.0 - (fat / fat_limit)) if fat_limit > 0 else 0.5

                # 4. 개인별 나트륨 점수
                sodium_limit = nutrition_targets["sodium_limit_per_meal"]
                sodium_score = max(0.0, 1.0 - (sodium / sodium_limit)) if sodium_limit > 0 else 0.5

                # 5. 탄수화물 점수 (활동량 반영)
                # 활동량이 높을수록 탄수화물 중요도 증가
                carb_optimal = ideal_kcal * 0.5 / 4  # 칼로리의 50%를 탄수화물로 가정
                carb_diff_ratio = abs(carb - carb_optimal) / carb_optimal if carb_optimal > 0 else 0
                carb_score = max(0.0, 1.0 - carb_diff_ratio) * nutrition_targets["carb_importance"]

                # 6. 연령별 가중치 적용한 최종 영양 점수
                age_weights = nutrition_targets["age_weights"]
                cand_nutrition_score = (
                    protein_score * age_weights["protein"] +
                    kcal_score * age_weights["kcal"] +
                    fat_score * age_weights["fat"] +
                    sodium_score * age_weights["sodium"]
                )

                # 탄수화물 점수 추가 (가중치 조정)
                cand_nutrition_score = (cand_nutrition_score * 0.8) + (carb_score * 0.2)

                # 0~1 범위로 정규화
                cand_nutrition_score = min(1.0, max(0.0, cand_nutrition_score))

                # ✅ 재료 매칭 로직
                recipe_ingredients = row.get("ingredient_names", [])
                if not isinstance(recipe_ingredients, list):
                    recipe_ingredients = []

                if idx == 0:
                    print(f"\n=== 첫 번째 레시피 분석 ===")
                    print(f"레시피명: {row['recipe_name']}")
                    print(f"영양정보: {kcal}kcal, 단백질 {protein}g, 지방 {fat}g, 나트륨 {sodium}mg")
                    print(f"개인별 영양점수: {cand_nutrition_score:.3f}")
                    print(f"  - 단백질점수: {protein_score:.3f}")
                    print(f"  - 칼로리점수: {kcal_score:.3f}")
                    print(f"  - 지방점수: {fat_score:.3f}")
                    print(f"  - 나트륨점수: {sodium_score:.3f}")

                # ✅ 교집합 계산
                owned_match = set(recipe_ingredients) & set(owned_ingredients)
                liked_match = set(recipe_ingredients) & set(preferred_ingredients)
                disliked_match = set(recipe_ingredients) & set(disliked_ingredients)
        
                num_owned_match = len(owned_match)
                num_liked_match = len(liked_match)
                num_disliked_match = len(disliked_match)


                # ✅ 보유 재료 중 ingredient_data에 있는 재료들의 가격 총합 계산
                recipe_total_price = 0.0
                matched_ingredients_with_price = []
                for ingredient_name in recipe_ingredients:
                    if ingredient_name not in owned_ingredients:  # 보유하지 않은 재료만
                        if ingredient_name in ingredient_prices:
                            price = ingredient_prices[ingredient_name]
                            recipe_total_price += price
                            matched_ingredients_with_price.append((ingredient_name, price))
                        else:
                            # 가격 정보가 없는 경우 기본값(0원) 처리 (선택적)
                            matched_ingredients_with_price.append((ingredient_name, 0.0))

                # ✅ 첫 번째 레시피의 매칭 결과 출력
                if idx == 0:
                    print(f"보유재료 매칭: {owned_match}")
                    print(f"선호재료 매칭: {liked_match}")
                    print(f"비선호재료 매칭: {disliked_match}")
                    print(f"가격이 있는 매칭 재료: {matched_ingredients_with_price}")
                    print(f"총 레시피 가격: {recipe_total_price}원")

                 # ✅ 강화된 비선호 패널티 계산 (방법 3: 단계별 강한 패널티)
                if num_disliked_match == 0:
                    cand_dislike_penalty = 0.0
                elif num_disliked_match == 1:
                    cand_dislike_penalty = 0.6  # 1개만 있어도 60% 패널티
                elif num_disliked_match == 2:
                    cand_dislike_penalty = 0.9  # 2개면 90% 패널티
                else:
                    cand_dislike_penalty = 1.0  # 3개 이상이면 최대 패널티    

                # # 비선호재료 패널티 (0.0 ~ 1.0, 많이 매치될수록 높음 = 페널티 증가)
                # cand_dislike_penalty = min(1.0, (
                #     (num_disliked_match / len(recipe_ingredients)) * penalty_multiplier
                #     if len(recipe_ingredients) > 0 else 0.0
                # ))


                # ✅ 피처 계산
                # 보유재료 매치 스코어 (0.0 ~ 1.0, 많이 매치될수록 높음)
                cand_ingredient_match_score = (
                    num_owned_match / len(recipe_ingredients) 
                    if len(recipe_ingredients) > 0 else 0.0
                )
        
                # 선호재료 매치 스코어 (0.0 ~ 1.0, 많이 매치될수록 높음)
                cand_like_match_score = (
                    num_liked_match / len(recipe_ingredients) 
                    if len(recipe_ingredients) > 0 else 0.0
                )
                
        
                # ✅ 개인별 맞춤 pref_like_match_importance 계산
                # 연령이 높을수록, 건강을 더 중시할수록 선호재료 중요도 감소
                if user_age >= 65:
                    pref_like_match_importance = 0.3  # 고령자는 건강 우선
                elif user_age >= 50:
                    pref_like_match_importance = 0.4  # 중년은 건강과 맛의 균형
                else:
                    pref_like_match_importance = 0.6  # 젊은층은 맛 

                features = {
                    "cand_kcal_gap_ratio": abs(kcal - user_info["tdee"] / 3) / (user_info["tdee"] / 3),
                    "cand_macro_gap_ratio": abs(kcal - user_info["tdee"] / 3) / (user_info["tdee"] / 3),
                    "cand_dislike_penalty": cand_dislike_penalty, # 강화된 비선호 패널티 계산 (0.0~1.0, 레시피 내 비선호 재료 비율이 높을수록 값 증가)
                    "cand_ingredient_match_score": cand_ingredient_match_score, # 보유 재료 매칭 점수 (0.0~1.0, 레시피 재료 중 사용자가 보유한 재료 비율)
                    "cand_like_match_score": cand_like_match_score, # 선호 재료 매칭 점수 (0.0~1.0, 레시피 재료 중 사용자가 선호하는 재료 비율)
                    "num_ingredients_owned": num_owned_match, # 레시피 내 보유 재료 개수 (정수, 비선호 제외)
                    "num_liked_ingredients_in_recipe": num_liked_match, # 레시피 내 선호 재료 개수 (정수)
                    "num_disliked_ingredients_in_recipe": num_disliked_match, # 레시피 내 비선호 재료 개수 (정수)
                    "user_total_disliked_ingredients": total_disliked, # 사용자 총 비선호 재료 수 (전체 설정 중 비선호로 표시된 재료 수)
                    "user_total_liked_ingredients": total_preferred, # 사용자 총 선호 재료 수 (전체 설정 중 선호로 표시된 재료 수)
                    "pref_budget_match_strictness": 0.5,#- 0.5로 고정 
                    "pref_dislike_penalty_sensitivity": 0.5, #- 0.5로 고정 
                    "pref_ingredient_match_importance": 0.5, #- 0.5로 고정 
                    "pref_like_match_importance": pref_like_match_importance,  # 연령별 동적 조정
                    "cand_nutrition_score": cand_nutrition_score,  # 개인별 맞춤 영양점수
                    "pref_kcal_target_accuracy": pref_kcal_target_accuracy,       
                    "pref_macro_target_accuracy": pref_macro_target_accuracy, 
                    "pref_nutrition_score": pref_nutrition_score,
                    "target_fat_total_g": user_profile["target_fat_total_g"],
                    "target_protein_total_g": user_profile["target_protein_total_g"],
                    "recipe_base_kcal": kcal,
                    "recipe_base_carb": float(row["recipe_base_carb"]),
                    "recipe_base_protein": float(row["recipe_base_protein"]),
                    "recipe_base_fat": float(row["recipe_base_fat"]),
                    "recipe_base_sodium": float(row["recipe_base_sodium"]),
                    "recipe_price": recipe_total_price,  # 레시피 가격은 임시, 재료의 총합으로 가격제공
                    "owned_match_names": list(owned_match),
                    "liked_match_names": list(liked_match),
                }
                features_list.append(features)
            return features_list

        features = compute_recipe_features(
        full_recipe_df,
        user_profile,
        request.budget_mid,
        request.ingredients,
        request.nutrition_level,  
        request.calorie_level     
    )



        # 터미널 검사용
        first_recipe_features = features[0]
        selected_features = {
        "gender": user_profile["gender"],
        "age": user_profile["age"],
        "height_cm": user_profile["height_cm"],
        "weight_kg": user_profile["weight_kg"],
        "pal_value": user_profile["pal_value"],
        "tdee": user_profile["tdee"],
        "target_protein_total_g": user_profile["target_protein_total_g"],
        "target_fat_total_g": user_profile["target_fat_total_g"],
        "budget_mid": first_recipe_features["recipe_price"]  # 또는 budget_mid
        }


        print("\n=== 첫 번째 레시피의 선택된 피처 ===")
        pprint.pprint(selected_features, width=120)

        if features:
            first_recipe_features = features[0]
            print(f"\n=== 재료 매칭 결과 (첫 번째 레시피) ===")
            print(f"레시피 재료 개수: {first_recipe_features.get('num_ingredients_owned', 0)}")
            print(f"선호 재료 매치: {first_recipe_features.get('num_liked_ingredients_in_recipe', 0)}")
            print(f"비선호 재료 매치: {first_recipe_features.get('num_disliked_ingredients_in_recipe', 0)}")
            print(f"보유재료 매치 스코어: {first_recipe_features.get('cand_ingredient_match_score', 0):.3f}")
            print(f"선호재료 매치 스코어: {first_recipe_features.get('cand_like_match_score', 0):.3f}")
            print(f"비선호재료 패널티: {first_recipe_features.get('cand_dislike_penalty', 0):.3f}")


        # 6. 전처리 및 예측
        input_df = pd.DataFrame([{**user_profile, **f} for f in features])
        processed_df = processing.preprocess_input_for_prediction(input_df)
        model: tf.keras.models.Model = artifacts["model"]
        proba = model.predict(processed_df).flatten()

        # 7. 상위 10개 선택
        top_indices = np.argsort(-proba)[:10]
        top_results = []
        for idx in top_indices:
            row = nutrition_df.iloc[idx]
            top_results.append({
                "menu_id": int(row["menu_id"]),
                "recipe_name": str(row["recipe_name"]),
                "prediction_proba": float(proba[idx]),
                "recipe_kcal": float(features[idx]["recipe_base_kcal"]),
                "recipe_protein": float(features[idx]["recipe_base_protein"]),
                "recipe_fat": float(features[idx]["recipe_base_fat"]),
                "recipe_sodium": float(features[idx]["recipe_base_sodium"]),
                "ingredient_match": float(features[idx]["cand_ingredient_match_score"]),
                "like_match": float(features[idx]["cand_like_match_score"]),
                "dislike_penalty": float(features[idx]["cand_dislike_penalty"]),
                "recipe_price": float(features[idx]["recipe_price"]),
                "owned_match_names": features[idx]["owned_match_names"],
                "liked_match_names": features[idx]["liked_match_names"],
    
                
            })

        return {"top_10": top_results}

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"예측 실패: {e}")

