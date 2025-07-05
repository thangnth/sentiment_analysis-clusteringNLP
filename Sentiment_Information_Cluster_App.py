# rm -rf myenv
# python3 -m venv myenv
# source myenv/bin/activate
# pip3 install -r requirements.txt
# streamlit run Sentiment_Information_Cluster_App.py
import streamlit as st
import pickle
import re, regex
import numpy as np
from underthesea import word_tokenize, pos_tag, sent_tokenize
from scipy.sparse import csr_matrix, hstack
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import io
import plotly.graph_objects as go  # <-- thêm Plotly cho radar chart tương tác
# ========== Sidebar ==========
# ====== Sidebar MENU  ======
st.markdown("""
    <style>
        /* Thay đổi chiều rộng sidebar */
        [data-testid="stSidebar"] {
            width: 350px;
        }

        /* Điều chỉnh vùng nội dung chính để tránh đè */
        [data-testid="stSidebar"] > div:first-child {
            width: 350px;
        }
    </style>
""", unsafe_allow_html=True)
with st.sidebar:
    with st.container():
        st.title("📚 Menu")
        menu_choice = st.radio("Chọn chức năng:", (
            "📌 Business Objective",
            "🏗️ Build Model",
            "💬 Sentiment Analysis",
            "🧩 Information Clustering"
        ))
# ===== Thông tin tác giả =====
st.sidebar.markdown("""
<div style='margin-top: auto; padding-bottom: 20px;'>
    <hr style="border: none; height: 1px; background-color: #ccc;">
    <h4>🎓 Tác giả đồ án:</h4>
    👨‍🎓 <b>Nguyễn Ngọc Huân</b><br>
    📧 nguyenngochuan992@gmail.com<br><br>
    👩‍🎓 <b>Nguyễn T. Hoa Thắng</b><br>
    📧 thangnth0511@gmail.com
</div>
""", unsafe_allow_html=True)

# ========== Load mô hình và vectorizer từ .pkl ==========
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("model_lr.pkl", "rb") as f:
    model_lr = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# ========== Load dictionary ==========
def load_dict_from_txt(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()
    return dict(line.split('\t') for line in lines if '\t' in line)

def load_list_from_txt(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

emoji_dict = load_dict_from_txt("files/emojicon.txt")
teen_dict = load_dict_from_txt("files/teencode.txt")
wrong_lst = load_list_from_txt("files/wrong-word.txt")
stopwords_lst = load_list_from_txt("files/vietnamese-stopwords.txt")
positive_words = load_list_from_txt("files/positive_VN.txt")
negative_words = load_list_from_txt("files/negative_VN.txt")
positive_emojis = load_list_from_txt("files/positive_emoji.txt")
negative_emojis = load_list_from_txt("files/negative_emoji.txt")
correct_dict = load_list_from_txt("files/phrase_corrections.txt")
english_dict = load_list_from_txt("files/english-vnmese.txt")
# ========== Tiền xử lý ==========
def covert_unicode(txt):
    return txt.encode('utf-8').decode('utf-8')

def normalize_repeated_characters(text):
    return re.sub(r'(.)\1+', r'\1', text)

def process_text(text):
    document = text.lower().replace("’", '')
    document = regex.sub(r'\.+', ".", document)
    new_sentence = ''
    for sentence in sent_tokenize(document):
        sentence = ''.join(emoji_dict.get(c, c) for c in sentence)
        sentence = ' '.join(teen_dict.get(w, w) for w in sentence.split())
        pattern = r'(?i)\b[a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ]+\b'
        sentence = ' '.join(regex.findall(pattern, sentence))
        sentence = ' '.join(w for w in sentence.split() if w not in wrong_lst)
        new_sentence += sentence + '. '
    return regex.sub(r'\s+', ' ', new_sentence).strip()

def process_postag_thesea(text):
    new_document = ''
    for sentence in sent_tokenize(text):
        sentence = sentence.replace('.', '')
        lst_word_type = ['N','Np','A','AB','V','VB','VY','R']
        sentence = ' '.join(
            word[0] if word[1] in lst_word_type else ''
            for word in pos_tag(word_tokenize(sentence, format="text"))
        )
        new_document += sentence + ' '
    return regex.sub(r'\s+', ' ', new_document).strip()

def remove_stopword(text):
    return regex.sub(r'\s+', ' ', ' '.join(w for w in text.split() if w not in stopwords_lst)).strip()

def count_sentiment_items(text):
    text = str(text).lower()
    pos_word = sum(1 for word in positive_words if word in text)
    pos_emoji = sum(text.count(emoji) for emoji in positive_emojis)
    neg_word = sum(1 for word in negative_words if word in text)
    neg_emoji = sum(text.count(emoji) for emoji in negative_emojis)
    return pos_word, neg_word, pos_emoji, neg_emoji

# ========== Dự đoán ==========
def predict_sentiment(text_input, recommend_num):
    text = covert_unicode(text_input)
    text = normalize_repeated_characters(text)
    text = process_text(text)
    text = process_postag_thesea(text)
    text = remove_stopword(text)

    tfidf_vector = vectorizer.transform([text])
    pos_word, neg_word, pos_emoji, neg_emoji = count_sentiment_items(text_input)
    numeric_features = scaler.transform([[pos_word, neg_word, pos_emoji, neg_emoji]])
    recommend_feature = csr_matrix([[recommend_num]])

    final_features = hstack([tfidf_vector, csr_matrix(numeric_features), recommend_feature])
    y_pred = model_lr.predict(final_features)[0]
    label = le.inverse_transform([y_pred])[0]
    return label



# ========== Các Trang Ứng Dụng ==========
# 🔧 Chèn CSS toàn trang (ngay từ đầu file)
st.set_page_config(page_title="Phân tích cảm xúc và phân cụm", layout="wide")
st.image("images/NaturalLanguageProcessing.png", width=1000)
st.markdown("""
    <style>
        .stMain {
            padding-left: 5% !important ;
            padding-right: 5% !important;
           
        }
        p, li {
            font-size:18px !important;
            text-align: justify;
            }
    </style>
""", unsafe_allow_html=True)


# Header ảnh dùng chung
if menu_choice == "📌 Business Objective":
    # st.title("📌 Giới thiệu đồ án: Sentiment Analysis & Information Clustering")
    col_sentiment_intro, col_sentiment_img = st.columns([2, 1])  # tỷ lệ chiều rộng 2:1
    with col_sentiment_intro:
        st.markdown("""
            <p> 
                <strong> Sentiment Analysis </strong> và <strong> Clustering </strong> 
                là các kỹ thuật trong lĩnh vực <strong> xử lý ngôn ngữ tự nhiên (Natural Language Processing - NLP) </strong>. </p>
            <p>
                <strong> Sentiment Analysis </strong>, hay còn gọi là <em> phân tích cảm xúc </em>, <em> phân tích quan điểm </em> , <em> Opinion Mining </em>) là kỹ thuật trong học máy có giám sát , nhằm xác định, trích xuất và định lượng các thông tin chủ quan từ văn bản. Nói một cách đơn giản, đây là quá trình máy tính hiểu và phân loại cảm xúc, thái độ, ý kiến hay quan điểm được thể hiện trong nội dung văn bản.</p>
            </p>
            <p>
                <strong> Clustering </strong>, hay còn gọi là <em> phân cụm </em>, là một kỹ thuật trong học máy không giám sát, dùng để nhóm các đối tượng dữ liệu tương tự nhau vào cùng một cụm (cluster). Mục tiêu là để các đối tượng trong cùng một cụm có tính tương đồng cao, trong khi các cụm khác nhau thì có tính khác biệt. </p>
                    
        """, unsafe_allow_html=True)
    with col_sentiment_img:
        st.image("images/img1_sentimen_intro.png", use_container_width=True) 
    st.markdown("""
           <div class="info-box">
           </div>
        """, unsafe_allow_html=True)
    st.markdown("""
            <h4> 🎯 Mục tiêu đồ án </h4>
                <p> 📝 Sử dụng kỹ thuật Sentiment Analysis xây dựng ứng dụng phân tích cảm xúc từ các đánh giá công ty của những người tham gia khảo sát ở trang ITViec. Từ đó: 
                    <ul style= "list-style-type: none" >
                        <li> 📍 Doanh nghiệp xây dựng các báo cáo nội bộ để cải tiến môi trường làm việc. </li>
                        <li> 📍 Tự động phát hiện làn sóng đánh giá tiêu cực, giúp cảnh báo sớm khủng hoảng truyền thông cho doanh nghiệp. </li>
                        <li> 📍 Gợi ý công ty phù hợp cho ứng viên. </li>
                    </ul>
                </p>
                <p> 📝 Sử dụng kỹ thuật Clustering xây dựng ứng dụng phân cụm các đánh giá công ty của những người tham gia khảo sát ở trang ITViec. Từ đó: 
                    <ul style= "list-style-type: none" >
                        <li> 📍 Giúp doanh nghiệp tìm ra các vấn đề nổi bậc của công ty : lương thấp, công nghệ cũ, quản lý tệ... </li>
                        <li> 📍 Giúp ứng viên so sánh các công ty có nhóm văn hoá khác nhau nhằm tìm kiếm môi trường làm việc phù hợp. </li>
                        <li> 📍 Khám phá các insight ẩn từ các đánh giá không dán nhãn.
                    </ul>
                </p>
    """, unsafe_allow_html=True)
    st.markdown("""
        <h4> 🔑 Các phương pháp xây dựng ứng dụng: </h4>
        <p> 📝 Sentiment Analysis:
            <ul style= "list-style-type: none" >
                <li> 📍 Sử dụng <strong> Tfidf </strong> vectorize dữ liệu văn bản. </li>
                <li> 📍 Sử dụng <strong> MinMaxScaler </strong> để chuẩn hoá dữ liệu kiểu số. </li>
                <li> 📍 Sử dụng <strong> hstack (Horizontal Stack) </strong> để nối các ma trận dense và spare thành ma trận đặc trưng X. </li>
                <li> 📍 Sử dụng <strong> Label Encoder </strong> để mã hoá nhãn Sentiment (positive, neutral, negative). </li>
                <li> 📍 Sử dụng các mô hình phân loại của thư viện sklearn : <strong> Logistic Regression, Random Forest, Decision Tree </strong> để huấn luyện train data.</li>
            </ul>
        </p>
        <p> 📝 Clustering Analysis:
            <ul style= "list-style-type: none" >
                <li> 📍 Sử dụng <strong> CountVectorizer </strong> vectorize dữ liệu văn bản.</li>
                <li> 📍 Sử dụng biểu đồ <strong> wordcould </strong> biểu diễn các cụm từ phổ biến. </li>
                <li> 📍 Sử dụng mô hình phân chủ đề (topic modeling)  <strong> LDA </strong> để trích xuất các chủ đề tìm ẩn từ tập văn bản.  </li>
                <li> 📍 Sử dụng các mô hình <strong> Kmean, Birch, SpectralClustering,  AgglomerativeClustering  </strong> để phân cụm. </li>
                <li> 📍 Trực quan hoá phân cụm bằng biểu đồ <strong> Scatter </strong> kết hợp giảm chiều dữ liệu bằng <strong> phương pháp PCA </strong>. </li>
            </ul>
        </p>
    """, unsafe_allow_html=True)

elif menu_choice == "🏗️ Build Model":
    st.title("Xây dựng mô hình")
    st.write("### I. Sentiment Analysis")
    st.write("##### 1. Data EDA")
    st.image("images/Sentiment_EDA.JPG")
    st.write("##### 2. Visualization")
    st.image("images/sentiment_distributed_data.JPG")
    st.write("##### 3. Build model and Evaluation")
    st.write("###### - Huấn luyện mô hình phân loại Logistic Regression , Random Forest , Decision Tree")
    st.write("###### - Đánh giá kết quả dựa trên Presicion , ReCall , F1-Score , Accuracy")
    st.image("images/sentiment_evaluation.JPG")
    st.write("###### Confusion Matrix")
    st.image("images/Confusion Matrix.JPG")
    st.markdown("Chọn mô hình <span style='color: red; font-weight: bold; text-decoration: underline'>Logistic Regression</span> là tối ưu nhất.",
    unsafe_allow_html=True)
    st.write("### II. Information Clustering")
    st.write("##### 1. Data EDA")
    st.image("images/Clustering_EDA.JPG")
    st.write("##### 2. Visualization")
    st.image("images/Cluster_wordcloud.JPG")
    st.write("##### 3. Build model and Evaluation")
    st.write("###### - Huấn luyện mô hình phân cụm với các thuật toán KMeans, AgglomerativeClustering, SpectralClustering, Birch")
    st.write("###### - Đánh giá kết quả dựa trên Sihouette score")
    st.image("images/k_evaluation.JPG", width=600)
    st.write("###### Trực quan hoá Elbow theo Sihouette score")
    st.image("images/ellbow.JPG")
    st.write("###### Trực quan hoá phân cụm")
    st.image("images/Cluster_distributed.JPG")
    st.markdown(" Kết luận : Chọn mô hình <span style='color: red; font-weight: bold; text-decoration: underline'>KMeans</span> với k=4 là mô hình tối ưu nhất vì:",unsafe_allow_html=True)
    st.markdown(""" 
    - Silhouette Score ≈  0.775 cao nhất với k=4, rất ổn định.
    - Các điểm còn lại giảm nhẹ nhưng vẫn khá cao → ổn định tốt.
    - Biểu đồ phân cụm (LDA + KMeans): Nhóm dữ liệu được chia rõ ràng, trực quan.
    - Ranh giới giữa các cụm rõ ràng, gần như không có điểm chồng lấn.
    """)
                
    st.write("##### 4. Interpreting and Visualizing Cluster Analysis Results")
    st.write("###### ✅ Chủ đề #1: Lương và chế độ đãi ngộ : Cụm này nhấn mạnh về các yếu tố về lương và phúc lợi , sự cân bằng giữa công việc và cuộc sống cá nhân, đặc  biệt có đề cập đến vấn đề bất cập là lương_chậm")
    st.write("###### 🔑 Key words: lương_thưởng, lương_tốt, lương_chậm, cân_bằng_cuộc_sống, cuộc_sống_công_việc, làm_việc_linh_hoạt, sức_khoẻ. sức_khoẻ")
    st.image("images/wordcloud_0.JPG")
    st.write("######  ✅ Chủ đề #2:  Môi trường làm việc chuyên nghiệp: ứng viên quan tâm đến môi trường làm việc chuyên nghiệp với các dự án lớn , công ty lớn. Cụm này có đề cập vấn đề bất cập là ‘công nghệ cũ’")
    st.write("###### 🔑 Key words: đồng_nghiệp_vui_vẻ, đồng_nghiệp_thân_thiện, nhà_quản_lý_thân_thiện, môi_trường_làm_việc_vui_vẻ, quy_trình_làm_việc, lương_đầy_đủ, khối_lượng_công_việc, chế_độ_phúc_lợi, chế_độ_đãi_ngộ")
    st.image("images/wordcloud_1.JPG")
    st.write("######  ✅ Chủ đề #3: Trãi nghiệm không gian làm việc hiện đại và môi trường làm việc trẻ trung")
    st.write("###### 🔑 Key words: dcông_ty_lớn, môi_trường_làm_việc_chuyên_nghiệp, công_ty_tốt, dự_án_lớn, môi_trường_làm_việc_năng_động, công_nghệ_cũ.")
    st.image("images/wordcloud_2.JPG")
    st.write("######  ✅ Chủ đề #4: Mối quan hệ trong công việc , cơ sở vật chất , quy trình làm việc, chế độ làm việc.")
    st.write("###### 🔑 Key words: đồng_nghiệp_vui_vẻ, đồng_nghiệp_thân_thiện, nhà_quản_lý_thân_thiện, môi_trường_làm_việc_vui_vẻ, quy_trình_làm_việc, lương_đầy_đủ, khối_lượng_công_việc, chế_độ_phúc_lợi, chế_độ_đãi_ngộ")
    st.image("images/wordcloud_3.JPG")
elif menu_choice == "💬 Sentiment Analysis":
    st.title("💬 Ứng dụng phân tích cảm xúc review công ty")

    input_text = st.text_area("✍️ Nhập câu đánh giá của bạn:", height=150)
    recommend_input = st.checkbox("✅ Bạn có recommend công ty này không?", value=False)
    recommend_num = 1 if recommend_input else 0

    if st.button("🚀 Dự đoán cảm xúc"):
        if not input_text.strip():
            st.warning("⛔ Vui lòng nhập nội dung review!")
        else:
            with st.spinner("🔍 Đang xử lý..."):
                result = predict_sentiment(input_text, recommend_num)
            st.success(f"✅ Kết quả dự đoán: **{result.upper()}**")
    st.markdown("---")
    st.subheader("📥 Phân tích file đánh giá hàng loạt")
    uploaded_file = st.file_uploader("Tải lên file Excel (.xlsx) có cột 'Name', 'review' và 'recommend'", type=["xlsx"])

    if uploaded_file:
        try:
            df_file = pd.read_excel(uploaded_file, engine="openpyxl")
            if 'review' not in df_file.columns or 'recommend' not in df_file.columns or 'Name' not in df_file.columns:
                st.error("❌ File cần có đủ 3 cột 'Name', 'review' và 'recommend'. Vui lòng kiểm tra lại.")
            else:
                with st.spinner("🔍 Đang xử lý dự đoán hàng loạt..."):
                    df_file['sentiment'] = df_file.apply(lambda row: predict_sentiment(row['review'], row['recommend']), axis=1)

                st.success("✅ Phân tích hoàn tất! Bạn có thể tải file kết quả và xem thống kê theo từng người đánh giá.")
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df_file.to_excel(writer, index=False)
                st.download_button(
                    label="📥 Tải kết quả (.xlsx)",
                    data=output.getvalue(),
                    file_name="sentiment_result.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

                st.markdown("---")
                st.subheader("📊 Thống kê cảm xúc theo người đánh giá (Name)")
                df_counts = df_file.groupby(['Name', 'sentiment']).size().unstack(fill_value=0)
                df_percent = df_counts.div(df_counts.sum(axis=1), axis=0) * 100
                st.dataframe(df_percent.style.format("{:.1f}%"))

                fig, ax = plt.subplots(figsize=(10, 5))
                df_percent[['positive', 'neutral', 'negative']].plot(
                    kind='bar', stacked=True, ax=ax,
                    color=['red', 'skyblue', 'blue']
                )
                ax.set_ylabel("Tỷ lệ (%)")
                ax.set_title("Tỷ lệ cảm xúc theo từng người đánh giá")
                ax.set_xticklabels(ax.get_xticklabels(),rotation=360)
                st.pyplot(fig)

        except Exception as e:
            st.error(f"Lỗi xử lý file: {e}")
elif menu_choice == "🧩 Information Clustering":
    st.title("🧩 Information Clustering")
    
    try:
        df = pd.read_csv("clustered_reviews.csv", encoding='utf-8')

        company_list = sorted(df["Company Name"].dropna().unique())
        selected_company = st.selectbox("🔎 Chọn công ty để phân tích:", company_list)
        df = df[df["Company Name"] == selected_company]
        # Radar chart cho các thuộc tính đánh giá
        st.markdown("---")
        st.subheader("📈 Đánh giá tổng quan theo các khía cạnh")
        radar_cols = [
            "Salary & benefits",
            "Training & learning",
            "Management cares about me",
            "Culture & fun",
            "Office & workspace"
        ]
        if all(col in df.columns for col in radar_cols):
            avg_scores = df[radar_cols].mean().values

            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=avg_scores,
                theta=radar_cols,
                fill='toself',
                name=selected_company,
                text=[f"{col}: {score:.2f}" for col, score in zip(radar_cols, avg_scores)],
                hoverinfo="text",
                marker=dict(color='royalblue')
            ))

            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
                showlegend=False,
                title=f"Biểu đồ Radar đánh giá - {selected_company}"
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ Dữ liệu không đầy đủ để vẽ biểu đồ radar.")

        # Bar chart & pie chart cho cột Sentiment
        if "Sentiment" in df.columns:
            st.markdown("---")
            st.subheader("📊 Phân phối cảm xúc từ đánh giá")

            sentiment_counts = df['Sentiment'].value_counts()

            color_map = {
                'positive': '#90ee90',
                'neutral': '#87cefa',
                'negative': '#ffb6c1'
            }
            colors = [color_map.get(sent, 'gray') for sent in sentiment_counts.index]

            col1, col2 = st.columns(2)
            with col1:
                fig_bar, ax = plt.subplots()
                sentiment_counts.plot(kind='bar', color=colors, ax=ax)
                ax.set_title("Số lượng bình chọn theo cảm xúc")
                ax.set_xlabel("Sentiment")
                ax.set_ylabel("Số lượng")
                ax.set_xticklabels(sentiment_counts.index, rotation=0)
                st.pyplot(fig_bar)

            with col2:
                fig_pie, ax = plt.subplots()
                sentiment_counts.plot(kind='pie', autopct='%1.1f%%', colors=colors, ax=ax)
                ax.set_title("Tỷ lệ cảm xúc theo phần trăm")
                ax.set_ylabel("")
                st.pyplot(fig_pie)        
    
        # Vector hóa văn bản
        vectorizer_cluster = CountVectorizer(max_features=1000)
        X_vec = vectorizer_cluster.fit_transform(df["binh_luan"])

        # Phân cụm với KMeans
        kmeans = KMeans(n_clusters=4, random_state=42)
        df["Cluster"] = kmeans.fit_predict(X_vec)

        def get_top_words_in_cluster(dataframe, cluster_id, n_words=10):
            cluster_text = " ".join(dataframe[dataframe['cluster'] == cluster_id]['clean_text'].dropna().astype(str).tolist())
            if not cluster_text:
                return []
            vectorizer = CountVectorizer(max_features=n_words)
            X = vectorizer.fit_transform([cluster_text])
            word_counts = X.sum(axis=0).A1
            words = vectorizer.get_feature_names_out()
            word_freq = pd.Series(word_counts, index=words).sort_values(ascending=False)
            return word_freq.index.tolist(), cluster_text

        cluster_stats = df['cluster'].value_counts().sort_index()
        st.markdown(f"### 📊 Công ty `{selected_company}` có các cụm như sau:")
                # Hàm lấy từ khóa toàn công ty
        def get_top_keywords_company(df, n_keywords=20):
            all_text = " ".join(df['clean_text'].dropna().astype(str).tolist())
            if not all_text:
                return []
            vectorizer = CountVectorizer()
            X = vectorizer.fit_transform([all_text])
            words = vectorizer.get_feature_names_out()
            counts = X.toarray().flatten()
            word_freq = pd.Series(counts, index=words).sort_values(ascending=False)
            return word_freq.head(n_keywords)
            
        for cluster_id in cluster_stats.index:
            top_words, cluster_text = get_top_words_in_cluster(df, cluster_id)
            st.markdown(f"- Cụm **#{cluster_id}**: 🔑 Từ khóa: _{', '.join(top_words)}_")
            if cluster_text:
                wordcloud = WordCloud(width=1000, height=500, background_color='white',max_words=10).generate(cluster_text)
                st.image(wordcloud.to_array(), caption=f"WordCloud cho cụm #{cluster_id}", use_container_width=True)
                            # === Thống kê tổng hợp các từ khóa từ tất cả cụm ===
        # all_keywords = []
        # for cluster_id in cluster_stats.index:
        #     top_words, _ = get_top_words_in_cluster(df, cluster_id)
        #     all_keywords.extend(top_words)

        # if all_keywords:
        #     st.markdown("---")
        #     st.markdown("### 🧠 Tổng hợp vấn đề nổi bật từ các cụm đánh giá")

        #     keyword_counts = pd.Series(all_keywords).value_counts()
        #     top_keywords = keyword_counts.head(10)

        #     for idx, (kw, count) in enumerate(top_keywords.items(), 1):
        #         st.markdown(f"{idx}. **{kw}** — xuất hiện trong **{count} cụm**")

        #     # Optional: vẽ biểu đồ từ khóa nổi bật
        #     fig, ax = plt.subplots()
        #     sns.barplot(x=top_keywords.values, y=top_keywords.index, palette="viridis", ax=ax)
        #     ax.set_title("📈 Từ khóa nổi bật nhất trong các cụm")
        #     ax.set_xlabel("Số cụm xuất hiện")
        #     ax.set_ylabel("Từ khóa")
        #     st.pyplot(fig) 
                # Từ khóa nổi bật toàn công ty
        st.markdown("---")
        st.subheader("📌 Từ khóa nổi bật toàn công ty")
        top_keywords = get_top_keywords_company(df, n_keywords=10)
        st.write("Top 10 từ khóa phổ biến:")
        st.markdown(", ".join(top_keywords.index))

        wordcloud_all = WordCloud(width=1000, height=500, background_color='white',max_words=10).generate(" ".join(df['clean_text']))
        st.image(wordcloud_all.to_array(), caption=f"WordCloud toàn bộ review công ty {selected_company}", use_container_width=True)            
    except Exception as e:
        st.error(f"Lỗi đọc hoặc xử lý dữ liệu: {e}")


