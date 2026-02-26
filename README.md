# Titanic Survival Prediction & Summary App

This project provides a complete solution for predicting Titanic passenger survival and generating insightful summaries using machine learning. It consists of two main components:

- **API**: Built with FastAPI, exposes endpoints for survival prediction and summary statistics.
- **UI**: Built with Streamlit, offers an interactive web interface for predictions and summary queries.

## Features

- Predict survival status for any passenger based on class, age, family, etc.
- View summary statistics (survival rates, cluster analysis, feature importance, etc.)
- Query summary endpoint with natural language questions (e.g., "What is the survival rate for luxury class?")
- Select summary prompts from UI or type your own

## Folder Structure

```
api/
    titanic_api.py
ui/
    titanic_ui.py
requirements.txt
```

## Setup Instructions

1. **Clone the repository**
2. **Install dependencies**

    ```bash
    pip install -r requirements.txt
    ```

3. **Run the API server**

    ```bash
    cd api
    uvicorn titanic_api:app --host 127.0.0.1 --port 8000
    ```

4. **Run the UI**

    ```bash
    cd ui
    streamlit run titanic_ui.py
    ```

## API Endpoints

- `/predict`: Predict survival status for a passenger
    - Query params: `pclass`, `age`, `sibsp`, `parch`
- `/summary`: Get summary statistics
    - Query param: `prompt` (natural language question or select from UI)

## Example Prompts for Summary Endpoint

- What is the survival rate for luxury class?
- What is the survival rate for passengers aged 20 to 30?
- Show cluster summary for passengers.
- How many children survived?
- How many seniors survived?
- Show survival rate by gender.
- Show survival rate by family.
- What is the survival rate for each class?
- What is the average age of survivors?
- What features influence survival?

## Data Source

- Titanic dataset from [Data Science Dojo](https://github.com/datasciencedojo/datasets/blob/master/titanic.csv)

## License

MIT License

---

For questions or improvements, please open an issue or pull request.
