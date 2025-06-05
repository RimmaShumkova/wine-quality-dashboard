import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle

st.set_page_config(page_title="ML Dashboard", layout="wide") #настройка страницы

#загрузка всех доступных моделей
models = {
    "CatBoost": "catboost_model.pkl",
    "GBoost": "gboost_model.pkl",
    "Polynomial Regression": "wine_quality_poly_model.pkl",
    "Bagging": "bagging_model.pkl",
    "Stacking": "stacking_model.pkl",
    "MLP Adam": "mlp_adam_model.pkl"
}

loaded_models = {}

for model_name, model_path in models.items():
    try:
        with open(model_path, 'rb') as file:
            loaded_models[model_name] = pickle.load(file)
    except FileNotFoundError:
        st.warning(f"Файл модели {model_name} не найден ({model_path}). Эта модель будет недоступна.")
    except Exception as e:
        st.warning(f"Ошибка при загрузке модели {model_name}: {e}. Эта модель будет недоступна!")

if not loaded_models:
    st.error("Ни одна из моделей не загружена. Пожалуйста, убедитесь, что хотя бы одна модель доступна.")
    st.stop()

#сайдбар для навигации
page = st.sidebar.selectbox("Выберите страницу", ["Информация об авторе", "Информация о наборе данных", "Визуализация данных", "Прогнозирование качества вина"])

if page == "Информация об авторе":
    st.title("Информация об авторе")
    st.header("ФИО: Шумкова Римма Сергеевна")
    st.subheader("Номер учебной группы: ФИТ-231")
    st.image("images\photo.jpeg", 
             width=300)

if page == "Информация о наборе данных":
    st.title("Информация о наборе данных: Wine Quality")

    st.header("Описание предметной области")
    st.write("""
    Набор данных "Wine Quality" содержит информацию о физико-химических свойствах вин и их субъективной оценке качества.
    Данные были собраны для исследования взаимосвязи между объективными измерениями состава вина и субъективными
    оценками его качества экспертами.
    """)

    st.subheader("Применение данных")
    st.write("""
    Понимание этих взаимосвязей может помочь в:
    - Контроле качества производства вина
    - Прогнозировании качества на основе химических показателей
    - Оптимизации процессов виноделия
    """)

    st.header("Описание признаков датасета")
    st.write("Датасет содержит следующие признаки (для красных и белых вин):")

    cols = st.columns(2)
    with cols[0]:
        st.markdown("""
        - **Фиксированная кислотность (fixed acidity)**: Основные нелетучие кислоты в вине (г/л)
        - **Летучая кислотность (volatile acidity)**: Количество уксусной кислоты (г/л)
        - **Лимонная кислота (citric acid)**: Содержание лимонной кислоты (г/л)
        - **Остаточный сахар (residual sugar)**: Количество сахара после брожения (г/л)
        - **Хлориды (chlorides)**: Содержание соли в вине (г/л)
        - **Свободный диоксид серы (free sulfur dioxide)**: Свободная форма SO₂ (мг/л)
        """)

    with cols[1]:
        st.markdown("""
        - **Общий диоксид серы (total sulfur dioxide)**: Общее количество SO₂ (мг/л)
        - **Плотность (density)**: Плотность вина (г/мл)
        - **pH**: Показатель кислотности вина (0-14)
        - **Сульфаты (sulphates)**: Добавка диоксида калия (г/л)
        - **Алкоголь (alcohol)**: Содержание алкоголя (% об.)
        - **Качество (quality)**: Оценка качества (0-10)
        """)

    st.header("Особенности предобработки данных")
    st.markdown("""
    1. **Обработка пропущенных значений**: В датасете отсутствуют пропущенные значения
    2. **Масштабирование признаков**: Требуется стандартизация из-за разных единиц измерения
    3. **Балансировка классов**: Оценки качества распределены неравномерно (преобладают 5-6)
    4. **Выбросы**: Некоторые химические показатели содержат выбросы
    5. **Корреляции**: Некоторые признаки сильно коррелированы между собой
    """)

    st.header("Результаты EDA (Exploratory Data Analysis)")
    st.markdown("""
    - **Распределение качества**: Большинство вин получили оценки 5-6
    - **Корреляция с качеством**:
      - Алкоголь (положительная)
      - Летучая кислотность (отрицательная)
      - Сульфаты (положительная)
    - **Различия между типами вин**:
      - Белые вина содержат больше сахара
      - Красные вина имеют более высокую летучую кислотность
    """)

if page == "Визуализация данных":
    st.title("Визуализация данных")

    data = pd.read_csv('winequality.csv')
    data.drop(['wine_type'], axis=1, inplace=True)

    #выбор визуализации
    visualization_option = st.selectbox(
        "Выберите визуализацию:",
        [
            "Распределение качества вина",
            "Корреляция между признаками",
            "Зависимость качества от алкоголя",
            "Зависимость качества от летучей кислотности",
            "Зависимость качества от сульфатов",
            "Распределение алкоголя",
            "Распределение летучей кислотности",
            "Распределение сульфатов"
        ]
    )

    if visualization_option == "Распределение качества вина":
        st.header("Распределение качества вина")
        fig, ax = plt.subplots()
        sns.histplot(data['quality'], bins=10, kde=True, ax=ax)
        st.pyplot(fig)

    elif visualization_option == "Корреляция между признаками":
        st.header("Корреляция между признаками")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(data.corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

    elif visualization_option == "Зависимость качества от алкоголя":
        st.header("Зависимость качества от алкоголя")
        fig, ax = plt.subplots()
        sns.regplot(x='alcohol', y='quality', data=data, ax=ax, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
        st.pyplot(fig)

    elif visualization_option == "Зависимость качества от летучей кислотности":
        st.header("Зависимость качества от летучей кислотности")
        fig, ax = plt.subplots()
        sns.regplot(x='volatile acidity', y='quality', data=data, ax=ax, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
        st.pyplot(fig)

    elif visualization_option == "Зависимость качества от сульфатов":
        st.header("Зависимость качества от сульфатов")
        fig, ax = plt.subplots()
        sns.regplot(x='sulphates', y='quality', data=data, ax=ax, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
        st.pyplot(fig)

    elif visualization_option == "Распределение алкоголя":
        st.header("Распределение алкоголя")
        fig, ax = plt.subplots()
        sns.histplot(data['alcohol'], bins=20, kde=True, ax=ax)
        st.pyplot(fig)

    elif visualization_option == "Распределение летучей кислотности":
        st.header("Распределение летучей кислотности")
        fig, ax = plt.subplots()
        sns.histplot(data['volatile acidity'], bins=20, kde=True, ax=ax)
        st.pyplot(fig)

    elif visualization_option == "Распределение сульфатов":
        st.header("Распределение сульфатов")
        fig, ax = plt.subplots()
        sns.histplot(data['sulphates'], bins=20, kde=True, ax=ax)
        st.pyplot(fig)




elif page == "Прогнозирование качества вина":
    st.title("Прогнозирование качества вина")
    
    #выбор модели
    selected_model = st.selectbox("Выберите модель для предсказания", list(loaded_models.keys()))
    model = loaded_models[selected_model]
    
    st.info(f"Выбрана модель: {selected_model}")

    #выбор способа ввода данных
    input_method = st.radio("Выберите способ ввода данных:",
                          ["Загрузить CSV файл", "Ввести вручную"])

    if input_method == "Загрузить CSV файл":
        st.subheader("Загрузка CSV файла")
        uploaded_file = st.file_uploader("Загрузите файл с данными о вине (CSV)", type="csv")

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.write("Загруженные данные:")
                st.dataframe(df)

                #проверка наличия необходимых столбцов
                required_columns = [
                    'fixed acidity', 'volatile acidity', 'citric acid',
                    'residual sugar', 'chlorides', 'free sulfur dioxide',
                    'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol'
                ]

                if all(col in df.columns for col in required_columns):
                    try:
                        predictions = model.predict(df[required_columns])
                        df['Predicted Quality'] = np.round(predictions, 1)

                        def interpret_quality(score):
                            if score < 4: return "Очень плохое качество вина"
                            elif 4 <= score < 6: return "Среднее качество вина"
                            elif 6 <= score < 8: return "Хорошее качество вина"
                            else: return "Отличное качество вина"

                        df['Quality Interpretation'] = df['Predicted Quality'].apply(interpret_quality)
                        df['Used Model'] = selected_model

                        st.subheader("Результаты предсказания")
                        st.dataframe(df)

                        #создаем кнопку для скачивания результатов
                        csv = df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Скачать результаты с предсказаниями",
                            data=csv,
                            file_name='wine_quality_predictions.csv',
                            mime='text/csv'
                        )

                    except Exception as e:
                        st.error(f"Ошибка при предсказании: {e}")
                else:
                    missing_cols = [col for col in required_columns if col not in df.columns]
                    st.error(f"В файле отсутствуют необходимые столбцы: {', '.join(missing_cols)}")
                    st.write(f"Требуемые столбцы: {', '.join(required_columns)}")

            except Exception as e:
                st.error(f"Произошла ошибка при загрузке файла: {e}")

    elif input_method == "Ввести вручную":
        st.subheader("Ввод данных вручную")
        
        #форма для ввода данных
        with st.form("input_form"):
            fixed_acidity = st.number_input("Фиксированная кислотность (г/л)", min_value=0.0, format="%.2f")
            volatile_acidity = st.number_input("Летучая кислотность (г/л)", min_value=0.0, format="%.2f")
            citric_acid = st.number_input("Лимонная кислота (г/л)", min_value=0.0, format="%.2f")
            residual_sugar = st.number_input("Остаточный сахар (г/л)", min_value=0.0, format="%.2f")
            chlorides = st.number_input("Хлориды (г/л)", min_value=0.0, format="%.2f")
            free_sulfur_dioxide = st.number_input("Свободный диоксид серы (мг/л)", min_value=0.0, format="%.2f")
            total_sulfur_dioxide = st.number_input("Общий диоксид серы (мг/л)", min_value=0.0, format="%.2f")
            density = st.number_input("Плотность (г/мл)", min_value=0.0, format="%.4f")
            ph = st.number_input("pH", min_value=0.0, format="%.2f")
            sulphates = st.number_input("Сульфаты (г/л)", min_value=0.0, format="%.2f")
            alcohol = st.number_input("Алкоголь (% об.)", min_value=0.0, format="%.2f")

            submitted = st.form_submit_button("Предсказать качество вина")

            if submitted:
                #предсказание качества вина на основе введенных данных
                input_data = np.array([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,
                                      free_sulfur_dioxide, total_sulfur_dioxide, density, ph, sulphates, alcohol]])
                try:
                    prediction = model.predict(input_data)
                    st.success(f"Предсказанное качество вина: {np.round(prediction[0], 1)}")
                    st.info(f"Использована модель: {selected_model}")
                    
                    score = np.round(prediction[0], 1)
                    if score < 4: interpretation = "Очень плохое"
                    elif 4 <= score < 6: interpretation = "Среднее"
                    elif 6 <= score < 8: interpretation = "Хорошее"
                    else: interpretation = "Отличное"
                    
                    st.write(f"Интерпретация: {interpretation}")
                    
                except Exception as e:
                    st.error(f"Ошибка при предсказании: {e}")
