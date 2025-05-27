# recommendation_api.py (업데이트: models 없이 사용자 재료 기반 추천)

from datetime import datetime
import mysql.connector
import pandas as pd
import numpy as np
import tensorflow as tf
import jwt
import os
from fastapi import APIRouter, HTTPException, Depends, Header, Request
from typing import Dict, Any, List

from . import models
from . import processing

router = APIRouter()

SECRET_KEY = os.getenv("JWT_SECRET_KEY", "supersecret")
ALGORITHM = "HS256"

# --- 공통 유틸 함수들 ---
def get_connection():
    return mysql.connector.connect(
        host="localhost", user="root", password="kjm10091009@M@", database="test"
    )

def get_user_profile(user_id: int):
    conn = get_connection()
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

    now_year = datetime.now().year
    age = now_year - row["birth_year"]
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

def get_recipe_ingredients(menu_id: int) -> List[str]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT 재료명 FROM ingredients WHERE menu_번호 = %s", (menu_id,))
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return [row[0] for row in rows]

def calculate_ingredient_features(recipe_ingredients: List[str], user_ingredients: List[Dict]) -> Dict:
    owned = {i['ingredient_name'] for i in user_ingredients if i.get('is_owned')}
    preferred = {i['ingredient_name'] for i in user_ingredients if i.get('is_preferred')}
    disliked = {i['ingredient_name'] for i in user_ingredients if i.get('is_disliked')}
    
    recipe_set = set(recipe_ingredients)

    return {
        'cand_ingredient_match_score': len(recipe_set & owned) / len(recipe_set or [1]),
        'cand_like_match_score': len(recipe_set & preferred) / len(preferred or [1]),
        'cand_dislike_penalty': len(recipe_set & disliked) / len(recipe_set or [1]),
        'num_ingredients_owned': len(recipe_set & owned),
        'num_liked_ingredients_in_recipe': len(recipe_set & preferred),
        'num_disliked_ingredients_in_recipe': len(recipe_set & disliked),
        'user_total_liked_ingredients': len(preferred),
        'user_total_disliked_ingredients': len(disliked)
    }

@router.post("/predict_top10_with_ingredients/", summary="사용자 재료 기반 추천")
async def predict_top10_recipes_with_ingredients(
    request: Request,
    Authorization: str = Header(...),
    artifacts: Dict[str, Any] = Depends(processing.load_artifacts_once)
):
    try:
        token = Authorization.split(" ")[1]
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_email = payload.get("sub")
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="유효하지 않은 토큰입니다.")

    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT id FROM users WHERE email = %s", (user_email,))
    user = cursor.fetchone()
    if not user:
        raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다.")
    user_id = user["id"]
    cursor.close()
    conn.close()

    user_profile = get_user_profile(user_id)
    if not user_profile:
        raise HTTPException(status_code=404, detail="프로필 없음")

    # 요청 데이터 파싱
    json_data = await request.json()
    budget_mid = json_data.get("budget_mid")
    user_ingredients = json_data.get("user_ingredients", [])
    nutrition_level = json_data.get("nutrition_level", 0.5)
    calorie_level = json_data.get("calorie_level", 0.5)

    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("""
        SELECT `일련번호` AS menu_id, `메뉴명` AS recipe_name,
               `열량` AS recipe_base_kcal, `탄수화물` AS recipe_base_carb,
               `단백질` AS recipe_base_protein, `지방` AS recipe_base_fat,
               `나트륨` AS recipe_base_sodium
        FROM menus LIMIT 155
    """)
    recipe_df = pd.DataFrame(cursor.fetchall())
    cursor.close()
    conn.close()

    def compute_recipe_features_enhanced():
        features_list = []
        for _, row in recipe_df.iterrows():
            menu_id = row['menu_id']
            recipe_ingredients = get_recipe_ingredients(menu_id)
            ing_feats = calculate_ingredient_features(recipe_ingredients, user_ingredients)

            features_list.append({
                **ing_feats,
                "cand_kcal_gap_ratio": abs(row['recipe_base_kcal'] - user_profile["tdee"] / 3) / (user_profile["tdee"] / 3),
                "cand_macro_gap_ratio": abs(row['recipe_base_kcal'] - user_profile["tdee"] / 3) / (user_profile["tdee"] / 3),
                "cand_nutrition_score": 0.7,
                "pref_budget_match_strictness": 0.5,
                "pref_dislike_penalty_sensitivity": 0.5,
                "pref_ingredient_match_importance": 0.5,
                "pref_kcal_target_accuracy": 0.5,
                "pref_like_match_importance": calorie_level,
                "pref_macro_target_accuracy": nutrition_level,
                "pref_nutrition_score": nutrition_level,
                "target_fat_total_g": user_profile["target_fat_total_g"],
                "target_protein_total_g": user_profile["target_protein_total_g"],
                "recipe_base_kcal": row['recipe_base_kcal'],
                "recipe_base_carb": row['recipe_base_carb'],
                "recipe_base_protein": row['recipe_base_protein'],
                "recipe_base_fat": row['recipe_base_fat'],
                "recipe_base_sodium": row['recipe_base_sodium'],
                "recipe_price": budget_mid
            })
        return features_list

    features = compute_recipe_features_enhanced()
    input_df = pd.DataFrame([{**user_profile, **f} for f in features])
    processed_df = processing.preprocess_input_for_prediction(input_df)
    model: tf.keras.models.Model = artifacts["model"]
    proba = model.predict(processed_df).flatten()

    top_indices = np.argsort(-proba)[:10]
    top_results = [
        {
            "menu_id": int(recipe_df.iloc[i]["menu_id"]),
            "recipe_name": recipe_df.iloc[i]["recipe_name"],
            "prediction_proba": float(proba[i])
        }
        for i in top_indices
    ]
    return {"top_10": top_results}
