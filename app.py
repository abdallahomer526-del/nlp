# -*- coding: utf-8 -*-
"""
نظام التنبؤ الافتراضي - بنك النيل الرقمي
تشغيل: streamlit run app.py
"""
import re
import joblib
import streamlit as st

st.set_page_config(page_title="بنك النيل الرقمي", page_icon="🏦", layout="centered")


def clean_text(text):
    text = str(text)
    text = re.sub(r"[\u064B-\u0652]", "", text)
    text = re.sub(r"[إأآا]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"ة", "ه", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@st.cache_resource
def load_model():
    model = joblib.load("model.joblib")
    vectorizer = joblib.load("vectorizer.joblib")
    return model, vectorizer


model, vectorizer = load_model()

# ---------- الواجهة ----------
st.markdown(
    "<h1 style='text-align:center;'>🏦 بنك النيل الرقمي</h1>"
    "<p style='text-align:center; color:gray;'>نظام تصنيف رسائل العملاء (نموذج افتراضي لأغراض تعليمية)</p>",
    unsafe_allow_html=True,
)
st.divider()

message = st.text_area("✍️ اكتب رسالة العميل هنا:", height=120,
                        placeholder="مثال: الخدمة كويسة بس التطبيق بطيء شوية")

if st.button("🔍 توقع", use_container_width=True):
    if not message.strip():
        st.warning("من فضلك اكتب رسالة أول.")
    else:
        cleaned = clean_text(message)
        X = vectorizer.transform([cleaned])
        pred = model.predict(X)[0]
        proba = model.predict_proba(X)[0]
        classes = model.classes_
        confidence = max(proba) * 100

        colors = {"إيجابي": "🟢", "سلبي": "🔴", "شكوى": "🟠"}
        st.markdown(f"### النتيجة: {colors.get(pred, '')} **{pred}**")
        st.progress(min(int(confidence), 100))
        st.write(f"نسبة الثقة: **{confidence:.1f}%**")

        with st.expander("تفاصيل الاحتمالات لكل فئة"):
            for cls, p in sorted(zip(classes, proba), key=lambda x: -x[1]):
                st.write(f"{cls}: {p * 100:.1f}%")

st.divider()
st.caption("هذا نظام افتراضي لمشروع أكاديمي — بنك النيل الرقمي شركة وهمية وليست كياناً حقيقياً.")
