import os
import joblib
import pandas as pd
import streamlit as st

# --- Инициализация состояния страницы ---
if 'page' not in st.session_state:
    st.session_state.page = 'main'

# --- Загрузка feature_map сразу при старте ---
if 'feature_map' not in st.session_state:
    try:
        fm_df = pd.read_csv('feature_map.csv')
        st.session_state.feature_map = dict(zip(fm_df['label'], fm_df['code']))
    except FileNotFoundError:
        st.session_state.feature_map = {
            'Тип клюва (beak_type)': 'beak_type',
            'Цвет оперения (plumage_color)': 'plumage_color',
            'Цвет глаз (eye_color)': 'eye_color',
            'Среда обитания (habitat)': 'habitat',
            'Тип питания (feeding_type)': 'feeding_type',
            'Размер тела (body_size)': 'body_size'
        }

# --- Вспомогательные функции для загрузки/сохранения настроек признаков ---
def load_value_settings():
    if 'val_types' not in st.session_state:
        st.session_state.val_types = {}
        st.session_state.ranges   = {}
        st.session_state.enums    = {}
        # Загрузить перечислимые значения
        for label, code in st.session_state.feature_map.items():
            fn = f"{code}_values.csv"
            if os.path.exists(fn):
                df = pd.read_csv(fn)
                st.session_state.enums[code] = df['label'].tolist()
                st.session_state.val_types[code] = 'Перечислимые'
        # Загрузить диапазоны
        if os.path.exists('ranges.csv'):
            rd = pd.read_csv('ranges.csv')
            for _, row in rd.iterrows():
                st.session_state.ranges[row['code']] = (int(row['min']), int(row['max']))
                st.session_state.val_types[row['code']] = 'Числовые'

def save_value_settings():
    # Сохранить перечислимые
    for code, enum_list in st.session_state.enums.items():
        pd.DataFrame({'label': enum_list}).to_csv(f"{code}_values.csv", index=False)
    # Сохранить диапазоны
    if st.session_state.ranges:
        rd = [{'code': c, 'min': mn, 'max': mx} for c, (mn, mx) in st.session_state.ranges.items()]
        pd.DataFrame(rd).to_csv('ranges.csv', index=False)
    # Сохранить типы
    types = [{'code': code, 'type': typ} for code, typ in st.session_state.val_types.items()]
    pd.DataFrame(types).to_csv('val_types.csv', index=False)

# --- Загрузка модели и энкодера ---
@st.cache_resource
def load_resources():
    bird_model = joblib.load('model.pkl')
    encoder_bird = joblib.load('label_encoder.pkl')
    return bird_model, encoder_bird

model, encoder = load_resources()

# --- Загрузка сохранённых настроек значений признаков ---
load_value_settings()

# --- Страницы ---
def main_page():
    st.title("Система классификации птиц")
    st.write("Выберите действие:")
    if st.button("Редактировать базу знаний"):
        st.session_state.page = 'edit'
    if st.button("Решить задачу"):
        st.session_state.page = 'classify'


def edit_knowledge_base():
    st.header("Редактирование базы знаний")
    tabs = st.tabs(["Классы", "Признаки", "Значения признаков", "Описание свойств вида", "Признаки классов"])

    # --- Таб "Классы" ---
    with tabs[0]:
        st.subheader("Управление классами (species)")
        if 'species_list' not in st.session_state:
            try:
                sp_df = pd.read_csv('classes.csv')
                st.session_state.species_list = sp_df['species'].astype(str).tolist()
            except FileNotFoundError:
                st.session_state.species_list = []
        new_class = st.text_input("Добавить новый класс (вид)")
        if st.button("Добавить класс", key="add_class") and new_class:
            if new_class not in st.session_state.species_list:
                st.session_state.species_list.append(new_class)
                pd.DataFrame({'species': st.session_state.species_list}).to_csv('classes.csv', index=False)
                st.success(f"Класс '{new_class}' добавлен")
        st.markdown("### Текущий список классов:")
        for i, sp in enumerate(st.session_state.species_list):
            c1, c2 = st.columns([0.9, 0.1])
            c1.write(sp)
            if c2.button("❌", key=f"del_class_{i}"):
                st.session_state.species_list.pop(i)
                pd.DataFrame({'species': st.session_state.species_list}).to_csv('classes.csv', index=False)
                st.experimental_rerun()

    # --- Таб "Признаки" ---
    with tabs[1]:
        st.subheader("Управление признаками")
        if 'features' not in st.session_state:
            try:
                feat_df = pd.read_csv('features.csv')
                st.session_state.features = feat_df['feature'].tolist()
            except FileNotFoundError:
                st.session_state.features = list(st.session_state.feature_map.keys())
        new_feature = st.text_input("Добавить новый признак")
        if st.button("Добавить признак", key="add_feature") and new_feature:
            if new_feature not in st.session_state.features:
                st.session_state.features.append(new_feature)
                pd.DataFrame({'feature': st.session_state.features}).to_csv('features.csv', index=False)
                code = new_feature[new_feature.find('(')+1:new_feature.find(')')] if '(' in new_feature else new_feature.replace(' ', '_').lower()
                pd.DataFrame(columns=['label']).to_csv(f"{code}_values.csv", index=False)
                st.session_state.feature_map[new_feature] = code
                pd.DataFrame(list(st.session_state.feature_map.items()), columns=['label','code']).to_csv('feature_map.csv', index=False)
                st.success(f"Признак '{new_feature}' добавлен")
        st.markdown("### Текущий список признаков:")
        for i, f in enumerate(st.session_state.features):
            c1, c2 = st.columns([0.9, 0.1])
            c1.write(f)
            if c2.button("❌", key=f"del_feat_{i}"):
                rem = st.session_state.features.pop(i)
                pd.DataFrame({'feature': st.session_state.features}).to_csv('features.csv', index=False)
                code = st.session_state.feature_map.pop(rem)
                pd.DataFrame(list(st.session_state.feature_map.items()), columns=['label','code']).to_csv('feature_map.csv', index=False)
                if os.path.exists(f"{code}_values.csv"): os.remove(f"{code}_values.csv")
                st.experimental_rerun()

    # --- Таб "Значения признаков" ---
    with tabs[2]:
        st.subheader("Управление значениями признаков")
        selected = st.selectbox("Выберите признак", list(st.session_state.feature_map.keys()), key="val_feat")
        code = st.session_state.feature_map[selected]
        typ = st.session_state.val_types.get(code, 'Перечислимые')
        typ = st.radio("Тип значений", ['Числовые','Перечислимые'], index=0 if typ=='Числовые' else 1, key=f"rt_{code}")
        st.session_state.val_types[code] = typ
        if typ=='Числовые':
            mn, mx = st.session_state.ranges.get(code,(0,100))
            mn_new = st.number_input("От", value=mn, key=f"mn_{code}")
            mx_new = st.number_input("До", value=mx, key=f"mx_{code}")
            if st.button("Сохранить диапазон", key=f"svr_{code}"):
                st.session_state.ranges[code] = (int(mn_new),int(mx_new))
                save_value_settings()
                st.success("Диапазон сохранён")
        else:
            vals = st.session_state.enums.get(code, [])
            st.markdown("**Текущие значения:**")
            for i,v in enumerate(vals):
                c1,c2 = st.columns([0.9,0.1])
                c1.write(v)
                if c2.button("❌", key=f"del_{code}_{i}"):
                    vals.pop(i)
                    st.session_state.enums[code] = vals
                    save_value_settings()
                    st.experimental_rerun()
            new_v = st.text_input("Новое значение", key=f"nv_{code}")
            if st.button("Добавить значение", key=f"add_{code}") and new_v:
                vals.append(new_v)
                st.session_state.enums[code] = vals
                save_value_settings()
                st.success("Значение добавлено")

    # --- Таб "Описание свойств вида" ---
    with tabs[3]:
        st.subheader("Описание свойств вида")
        try:
            sf_df = pd.read_csv('species_features.csv')
        except FileNotFoundError:
            sf_df = pd.DataFrame(columns=['species']+list(st.session_state.feature_map.values()))
        sp = st.selectbox("Выберите вид птицы", st.session_state.species_list, key="sp_feat")
        row = sf_df[sf_df['species']==sp]
        current = row.iloc[0].to_dict() if not row.empty else {}
        edited = {'species':sp}
        for label, code in st.session_state.feature_map.items():
            checked = st.checkbox(label, value=bool(current.get(code, False)), key=f"cb_{code}")
            edited[code] = checked
        if st.button("Сохранить настройки", key="save_sf"):
            if not row.empty:
                for k,v in edited.items(): sf_df.loc[sf_df['species']==sp, k] = v
            else:
                sf_df = pd.concat([sf_df, pd.DataFrame([edited])], ignore_index=True)
            sf_df.to_csv('species_features.csv', index=False)
            st.success("Описание видов сохранено")

    # --- Таб "Признаки классов" ---
    with tabs[4]:
        st.subheader("Управление значениями признаков для видов")
        try:
            bd = pd.read_csv('birds_data.csv')
        except FileNotFoundError:
            bd = pd.DataFrame(columns=['species']+list(st.session_state.feature_map.values()))
        if not st.session_state.species_list:
            st.warning("Сначала добавьте класс")
        else:
            sp2 = st.selectbox("Выберите вид", st.session_state.species_list, key="sp_cls")
            sf_df = pd.read_csv('species_features.csv') if os.path.exists('species_features.csv') else pd.DataFrame()
            allowed = sf_df[sf_df['species']==sp2].iloc[0].to_dict() if not sf_df.empty and (sf_df['species']==sp2).any() else {}
            rec = bd[bd['species']==sp2].iloc[0].to_dict() if not bd[bd['species']==sp2].empty else {code:None for code in st.session_state.feature_map.values()}
            edited2 = {}
            for label,code in st.session_state.feature_map.items():
                if not allowed.get(code, False): continue
                typ2 = st.session_state.val_types.get(code,'Перечислимые')
                cur = rec.get(code)
                if typ2=='Числовые':
                    mn,mx = st.session_state.ranges.get(code,(0,100))
                    default = cur if pd.notna(cur) else mn
                    val = st.slider(label, mn, mx, int(default), key=f"sl_{code}")
                else:
                    opts = st.session_state.enums.get(code, [])
                    default = cur if cur in opts else (opts[0] if opts else "")
                    val = st.selectbox(label, opts, index=opts.index(default) if default in opts else 0, key=f"sb_{code}")
                edited2[code] = val
            if st.button("Сохранить изменения", key="save_bd"):
                missing = [label for label,code in st.session_state.feature_map.items() if allowed.get(code, False) and (edited2.get(code) is None or (isinstance(edited2.get(code), str) and edited2.get(code)==''))]
                if missing:
                    st.error(f"Заполните значения для признаков: {', '.join(missing)}")
                else:
                    if bd[bd['species']==sp2].empty:
                        bd = pd.concat([bd, pd.DataFrame([{'species':sp2, **edited2}])], ignore_index=True)
                    else:
                        for k,v in edited2.items(): bd.loc[bd['species']==sp2, k] = v
                    bd.to_csv('birds_data.csv', index=False)
                    st.success("Данные сохранены")

    if st.button("Назад"):
        st.session_state.page = 'main'


def classify_bird():
    st.header("Классификация птицы по фенотипу")
    feature_map = st.session_state.feature_map
    enums       = st.session_state.enums
    ranges      = st.session_state.ranges
    val_types   = st.session_state.val_types
    inputs = {}
    # собираем признаки
    for label, code in feature_map.items():
        typ = val_types.get(code, 'Перечислимые')
        if typ == 'Числовые':
            mn, mx = ranges.get(code, (0, 100))
            default = (mn + mx) // 2
            val = st.slider(label, mn, mx, default, key=f"cl_{code}")
            inputs[code] = val
        else:
            options = enums.get(code, [])
            if not options:
                st.warning(f"Список значений для '{label}' пуст. Добавьте их в базе знаний.")
                return
            sel = st.selectbox(label, options, key=f"cl_{code}")
            # преобразуем выбор в числовой код по индексу
            inputs[code] = options.index(sel)
    if st.button("Определить вид"):
        # формируем массив признаков в порядке feature_map.values()
        x = [[inputs[code] for code in feature_map.values()]]
        y = model.predict(x)[0]
        sp = encoder.inverse_transform([y])[0]
        st.success(f"Вид птицы: {sp}")
    if st.button("Назад", key="back2"):
        st.session_state.page = 'main'            

# --- Рендер страниц ---
if st.session_state.page == 'main':
    main_page()
elif st.session_state.page == 'edit':
    edit_knowledge_base()
else:
    classify_bird()
