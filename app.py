import streamlit as st
import pandas as pd
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

def preprocess_data(df, encoder=None, fitted_columns=None):
    df = df.copy()
    if "selling_price" in df.columns:
        df = df.drop(columns=["selling_price"])

    df = df.drop_duplicates().reset_index(drop=True)
    df = df.dropna().reset_index(drop=True)
    kept_index = df.index

    cat_cols = ['fuel', 'seller_type', 'transmission', 'owner', 'seats']
    cat_cols = [c for c in cat_cols if c in df.columns]
    num_cols = [c for c in df.columns if c not in cat_cols and c not in ['name']]

    num_imputer = SimpleImputer(strategy="median")
    df[num_cols] = num_imputer.fit_transform(df[num_cols])

    cat_imputer = SimpleImputer(strategy="most_frequent")
    df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

    if encoder is None:
        encoder = OneHotEncoder(drop='first', sparse_output=False, dtype=int, handle_unknown="ignore")
        encoded = encoder.fit_transform(df[cat_cols])
    else:
        encoded = encoder.transform(df[cat_cols])

    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_cols), index=df.index)
    final_df = pd.concat([df[num_cols], encoded_df], axis=1)

    if fitted_columns is not None:
        for col in fitted_columns:
            if col not in final_df.columns:
                final_df[col] = 0
        final_df = final_df[fitted_columns]

    final_df = final_df.fillna(0)
    return final_df, encoder, kept_index

st.title('Предсказание стоимости автомобиля 🚗')
st.image('cars.jpg')
st.sidebar.title('Загрузка файлов')

model_status = st.sidebar.empty()
data_status = st.sidebar.empty()

model_file = st.sidebar.file_uploader("Файл моделей (.pkl, .pickle)", type=["pkl", "pickle"], key="model_file_widget")
if model_file is not None:
    artifacts = pickle.load(model_file)
    st.session_state["artifacts"] = artifacts
    model_status.success(f"Файл с моделями: {model_file.name}")
elif "artifacts" not in st.session_state:
    model_status.info("Файл с моделями не загружен!")

data_file = st.sidebar.file_uploader("CSV-файл с данными", type=["csv"], key="data_file_widget")
if data_file is not None:
    df = pd.read_csv(data_file)
    df = df.reset_index(drop=True)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    st.session_state["data"] = df
    data_status.success(f"Файл с данными: {data_file.name}")
elif "data" not in st.session_state:
    data_status.info("Файл с данными не загружен!")

if "data" in st.session_state:
    df = st.session_state["data"]

    st.subheader('Разведочный анализ набора данных')
    st.dataframe(df)

    st.subheader('Визуализация зависимостей между признаками')
    fig = sns.pairplot(df.select_dtypes(include=['int64', 'float64']))
    st.pyplot(fig)

    if "artifacts" in st.session_state:
        st.subheader('Применение модели для прогноза цены автомобиля')

        models = st.session_state["artifacts"]['models']
        fitted_columns = st.session_state["artifacts"]['feature_columns']
        encoder = st.session_state["artifacts"].get("encoder", None)

        selected_model_name = st.selectbox("Выберите модель", options=list(models.keys()))
        selected_model = models[selected_model_name]

        if st.button("Предсказать"):
            df_processed, encoder, kept_idx = preprocess_data(df, encoder=encoder, fitted_columns=fitted_columns)
            y_pred = selected_model.predict(df_processed)
            df_result = df.loc[kept_idx].copy()
            df_result["PredictedPrice"] = y_pred

            st.session_state["df_result"] = df_result
            st.session_state["selected_model"] = selected_model
            st.session_state["fitted_columns"] = fitted_columns

            st.success(f"Предсказания модели '{selected_model_name}' выполнены")
            st.dataframe(df_result[["PredictedPrice"]])

        if "selected_model" in st.session_state:
            if hasattr(st.session_state["selected_model"], "coef_"):
                if st.button("Показать веса модели"):
                    coefs = st.session_state["selected_model"].coef_
                    coef_df = pd.DataFrame({
                        "feature": st.session_state["fitted_columns"],
                        "weight": coefs
                    }).sort_values("weight", key=lambda x: x.abs(), ascending=False)

                    st.subheader("Веса признаков модели")
                    st.dataframe(coef_df)

                    plt.figure(figsize=(10, 6))
                    plt.barh(coef_df["feature"], coef_df["weight"])
                    plt.gca().invert_yaxis()
                    plt.title(f"Веса модели: {selected_model_name}")
                    plt.xlabel("Коэффициент")
                    plt.ylabel("Признак")
                    st.pyplot(plt)
