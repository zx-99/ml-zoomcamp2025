
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import randint, uniform

# 常量定义
SEED = 1

def load_data(data_path: str) -> pd.DataFrame:
    
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    df = df.drop(columns='id')
    
    return df

def train_model(X_train: pd.DataFrame , y_train: pd.Series) -> xgb.XGBClassifier:
    np.random.seed(seed=SEED)
    
    # 初始化模型
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='aucpr',  # 优先优化PR-AUC
        random_state=SEED
    )
    
    # 超参数分布
    param_dist = {
        'n_estimators': [50, 100, 150, 200],
        'max_depth': randint(3, 10),
        'eta': uniform(0.01, 0.3),
        'min_child_weight': randint(1, 30)
    }

    # 随机搜索最佳参数,并重新训练
    random_search = RandomizedSearchCV(
        estimator= model,
        param_distributions=param_dist,
        n_iter=20,
        cv=5,
        scoring='average_precision',
        refit = True,
        random_state=SEED
    )

    print('training.....')
    random_search.fit(X_train, y_train)
    # 输出最佳参数
    print("Best Parameters:", random_search.best_params_)
    print("Best PR-AUC Score: ", random_search.best_score_)
    return random_search.best_estimator_

def evaluate_metrics(model, X_val: pd.DataFrame, y_val: pd.Series):
    y_pred = model.predict_proba(X_val)[:,1]
    rocauc = roc_auc_score(y_val, y_pred)
    prauc = average_precision_score(y_val, y_pred)
    print('ROC-auc Val score: %.4f' %rocauc)
    print('PR-auc Val score: %.4f' %prauc)
    
def save_model(model, model_path: str):
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {model_path}")

def main():
    df_full_train = load_data('./data/train.csv')

    X_full_train = df_full_train.drop(columns='smoking')
    y_full_train = df_full_train['smoking']

    X_train, X_val, y_train, y_val = train_test_split(X_full_train, y_full_train, test_size=0.2, random_state=SEED)

    assert len(X_full_train) == len(X_train) + len(X_val)
    assert len(y_full_train) == len(y_train) + len(y_val)
    
    # Train it
    best_model = train_model(X_train, y_train)
    print("------")
    # Validate it
    evaluate_metrics(best_model, X_val, y_val)
    print("------")
    # Final full training
    final_model = train_model(X_full_train, y_full_train)
    print("------")
    # Save trained model
    save_model(final_model, './models/model.pkl')
    
if __name__ == "__main__":
    main()

