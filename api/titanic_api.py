from fastapi import FastAPI
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
import uvicorn

# --- 1. SETUP THE "CHEF" (The ML Model) ---
# Let us train the model once when the script starts
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)[['Survived', 'Pclass', 'Age', 'SibSp', 'Parch']].dropna()
X = df.drop('Survived', axis=1)
y = df['Survived']
model = LogisticRegression()
model.fit(X, y)

# KMeans model for clustering survivors
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X)

# --- 2. SETUP THE "WAITER" (FastAPI) ---
app = FastAPI()

@app.get("/")
def home():
    return {"message": "Titanic Survival Prediction API is Online!"}

from fastapi import Query

@app.get("/summary")
def summary(prompt: str = Query(None, description="Summary prompt text")):
    # Dictionary of prompt handlers
    def luxury_class():
        luxury_class = df[df['Pclass'] == 1]
        luxury_survived = luxury_class[luxury_class['Survived'] == 1]
        luxury_survived_count = len(luxury_survived)
        luxury_total = len(luxury_class)
        luxury_survived_rate = luxury_survived_count / luxury_total if luxury_total > 0 else 0
        return {"luxury_class_survived_rate": f"{luxury_survived_rate:.2%}"}

    def age_20_30():
        age_group = df[(df['Age'] >= 20) & (df['Age'] <= 30)]
        age_group_survived = age_group[age_group['Survived'] == 1]
        age_group_survived_count = len(age_group_survived)
        age_group_total = len(age_group)
        age_group_survived_rate = age_group_survived_count / age_group_total if age_group_total > 0 else 0
        return {"age_20_30_survived_rate": f"{age_group_survived_rate:.2%}"}

    def cluster_summary():
        cluster_labels = kmeans.labels_
        cluster_summary = {}
        for cluster_id in range(kmeans.n_clusters):
            cluster_indices = (cluster_labels == cluster_id)
            cluster_df = df[cluster_indices]
            avg_age = cluster_df['Age'].mean() if not cluster_df.empty else None
            class_dist = cluster_df['Pclass'].value_counts().to_dict() if not cluster_df.empty else {}
            survived_count = cluster_df[cluster_df['Survived'] == 1].shape[0]
            total_count = cluster_df.shape[0]
            survived_rate = survived_count / total_count if total_count > 0 else None
            cluster_summary[f"cluster_{cluster_id}"] = {
                "count": total_count,
                "average_age": round(avg_age, 2) if avg_age is not None else None,
                "class_distribution": class_dist,
                "survived_rate": f"{survived_rate:.2%}" if survived_rate is not None else None
            }
        return {"kmeans_cluster_summary": cluster_summary}

    def children():
        children = df[df['Age'] < 12]
        children_survived = children[children['Survived'] == 1]
        children_rate = len(children_survived) / len(children) if len(children) > 0 else 0
        return {"children_survived_rate": f"{children_rate:.2%}"}

    def seniors():
        seniors = df[df['Age'] > 60]
        seniors_survived = seniors[seniors['Survived'] == 1]
        seniors_rate = len(seniors_survived) / len(seniors) if len(seniors) > 0 else 0
        return {"seniors_survived_rate": f"{seniors_rate:.2%}"}

    def gender():
        gender_rates = {}
        if 'Sex' in df.columns:
            for gender in df['Sex'].unique():
                gender_df = df[df['Sex'] == gender]
                survived = gender_df[gender_df['Survived'] == 1]
                rate = len(survived) / len(gender_df) if len(gender_df) > 0 else 0
                gender_rates[f"{gender}_survived_rate"] = f"{rate:.2%}"
        return {"gender_survival_rates": gender_rates}

    def family():
        with_family = df[(df['SibSp'] + df['Parch']) > 0]
        alone = df[(df['SibSp'] + df['Parch']) == 0]
        with_family_survived = with_family[with_family['Survived'] == 1]
        alone_survived = alone[alone['Survived'] == 1]
        with_family_rate = len(with_family_survived) / len(with_family) if len(with_family) > 0 else 0
        alone_rate = len(alone_survived) / len(alone) if len(alone) > 0 else 0
        return {"family_survival_rates": {
            "with_family": f"{with_family_rate:.2%}",
            "alone": f"{alone_rate:.2%}"
        }}

    def class_survival():
        class_rates = {}
        for pclass in sorted(df['Pclass'].unique()):
            class_df = df[df['Pclass'] == pclass]
            survived = class_df[class_df['Survived'] == 1]
            rate = len(survived) / len(class_df) if len(class_df) > 0 else 0
            class_rates[f"class_{pclass}_survived_rate"] = f"{rate:.2%}"
        return {"class_survival_rates": class_rates}

    def average_age():
        avg_age_survived = df[df['Survived'] == 1]['Age'].mean()
        avg_age_nonsurvived = df[df['Survived'] == 0]['Age'].mean()
        return {
            "average_age_survived": round(avg_age_survived, 2),
            "average_age_nonsurvived": round(avg_age_nonsurvived, 2)
        }

    def feature_importance():
        feature_importance = {}
        if hasattr(model, 'coef_'):
            for idx, col in enumerate(X.columns):
                feature_importance[col] = float(model.coef_[0][idx])
        return {"feature_importance": feature_importance}


    # Dictionary mapping prompt keys to handler functions
    prompt_dict = {
        "luxury class": luxury_class,
        "age 20-30": age_20_30,
        "cluster summary": cluster_summary,
        "children": children,
        "seniors": seniors,
        "gender": gender,
        "family": family,
        "class survival": class_survival,
        "average age": average_age,
        "feature importance": feature_importance
    }

    # Map natural language questions/statements to prompt keys
    question_map = {
        "luxury class": [
            "luxury class", "first class", "survival rate for luxury class", "what is the survival rate for luxury class", "how many survived in luxury class"
        ],
        "age 20-30": [
            "age 20-30", "survival rate for age 20-30", "what is the survival rate for passengers aged 20 to 30", "how many survived between ages 20 and 30"
        ],
        "cluster summary": [
            "cluster summary", "show cluster summary", "kmeans cluster summary", "passenger clusters", "what are the clusters"
        ],
        "children": [
            "children", "survival rate for children", "how many children survived", "passengers under 12"
        ],
        "seniors": [
            "seniors", "survival rate for seniors", "how many seniors survived", "passengers over 60"
        ],
        "gender": [
            "gender", "survival rate by gender", "male vs female survival", "how many males and females survived"
        ],
        "family": [
            "family", "survival rate by family", "with family", "alone", "how many survived with family", "how many survived alone"
        ],
        "class survival": [
            "class survival", "survival rate by class", "what is the survival rate for each class", "how many survived in each class"
        ],
        "average age": [
            "average age", "average age of survivors", "average age of non-survivors", "what is the average age of survivors"
        ],
        "feature importance": [
            "feature importance", "top features influencing survival", "what features influence survival", "model coefficients"
        ]
    }

    if prompt:
        prompt_lower = prompt.lower()
        for key, phrases in question_map.items():
            for phrase in phrases:
                if phrase in prompt_lower:
                    return prompt_dict[key]()
        return {"error": "Prompt not recognized. Please use a supported summary question or statement."}

    # Default: return all summaries
    # ...existing code...

@app.get("/predict")
def predict_survival(pclass: int, age: float, sibsp: int, parch: int):
    # Prepare the input for the model
    input_data = [[pclass, age, sibsp, parch]]
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    
    status = "Survived" if prediction == 1 else "Did not survive"
    
    return {
        "prediction": status,
        "probability_of_survival": f"{probability:.2%}"
    }

# --- 3. RUN THE SERVER ---
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)