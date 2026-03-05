import gradio as gr
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# =====================================
# LOAD DATA
# =====================================

df = pd.read_csv("cosmopedia_final.csv")

required_cols = [
    "asin",
    "title",
    "category_detected",
    "ultimate_score",
    "sentiment_score",
    "review_count"
]

for col in required_cols:
    if col not in df.columns:
        df[col] = 0

df["title"] = df["title"].fillna("").astype(str)
df["category_detected"] = df["category_detected"].fillna("Unknown")

# =====================================
# LOAD SEMANTIC SEARCH MODEL
# =====================================

search_model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = search_model.encode(
    df["title"].tolist(),
    show_progress_bar=False
)

# =====================================
# SEARCH FUNCTION
# =====================================

def search_product(query):

    if not query:
        return None

    query_embedding = search_model.encode([query])
    similarities = cosine_similarity(query_embedding, embeddings)[0]

    best_index = np.argmax(similarities)
    best_score = similarities[best_index]

    # Avoid random matches
    if best_score < 0.40:
        return None

    return df.iloc[best_index]

# =====================================
# SCORE COLOR FUNCTION
# =====================================

def score_color(score):

    if score >= 8:
        return "🟢 Excellent"
    elif score >= 6:
        return "🟡 Good"
    else:
        return "🔴 Needs Improvement"

# =====================================
# INGREDIENT SCORE CALCULATOR
# =====================================

def ingredient_score_calculator(title):

    title = title.lower()

    good_ingredients = [
        "aloe", "vitamin", "vitamin c", "vitamin e",
        "hyaluronic", "glycerin", "niacinamide",
        "ceramide", "zinc", "peptide",
        "retinol", "collagen", "salicylic",
        "tea tree", "jojoba", "shea butter"
    ]

    harmful_ingredients = [
        "paraben", "sulfate", "alcohol",
        "fragrance", "phthalate",
        "triclosan", "formaldehyde"
    ]

    score = 6

    for word in good_ingredients:
        if word in title:
            score += 1

    for word in harmful_ingredients:
        if word in title:
            score -= 2

    score = max(1, min(score, 10))

    return score

# =====================================
# COMPETITOR FUNCTION
# =====================================

def get_competitors(category, exclude_asin):

    competitors = df[
        (df["category_detected"] == category) &
        (df["asin"] != exclude_asin)
    ].sort_values(by="ultimate_score", ascending=False)

    return competitors.head(3)

# =====================================
# MAIN ANALYSIS FUNCTION
# =====================================

def analyze(product_name):

    if not product_name.strip():
        return "Please enter a product name.", None

    product = search_product(product_name)

    if product is None:
        return "❌ Product not found. Try simpler name.", None

    asin = product["asin"]
    title = product["title"]
    category = product["category_detected"]

    ultimate = round(float(product["ultimate_score"]), 2)
    sentiment = round(float(product.get("sentiment_score", 0)), 2)

    # Dynamic ingredient scoring
    ingredient = ingredient_score_calculator(title)

    reviews = int(product.get("review_count", 0))

    competitors = get_competitors(category, asin)
    competitor_table = competitors[["title", "ultimate_score"]]

    # =====================================
    # SUMMARY LOGIC
    # =====================================

    if sentiment > 8:
        customer_feel = "strongly positive"
    elif sentiment > 6:
        customer_feel = "generally positive"
    else:
        customer_feel = "mixed"

    if ingredient > 7:
        safety_level = "safe"
    elif ingredient > 5:
        safety_level = "moderate"
    else:
        safety_level = "concerning"

    report = f"""
## 🌿 AI Cosmopedia Report

### 📦 Product: {title}

**Category:** {category}  
**Total Reviews:** {reviews}

---

### ⭐ Overall Performance  
Ultimate Score: {ultimate}/10 — {score_color(ultimate)}

### 💬 Customer Sentiment  
Sentiment Score: {sentiment}/10 — {score_color(sentiment)}

### 🧪 Ingredient Safety  
Safety Score: {ingredient}/10 — {score_color(ingredient)}

---

### 📝 Summary
• This product belongs to the **{category}** category.  
• Customer response is **{customer_feel}**.  
• Ingredient safety level is **{safety_level}**.  
• Ranked among strong performers in its category based on AI evaluation.

---

### 🏆 Top Competitors
See comparison table below.
"""

    return report, competitor_table

# =====================================
# UI DESIGN
# =====================================

with gr.Blocks(theme=gr.themes.Soft(primary_hue="green")) as demo:

    gr.Markdown("""
# 🌿 AI Cosmopedia  
### Intelligent Cosmetic Product Evaluation System  
Analyze, Compare & Rank Beauty Products Using AI
""")

    product_input = gr.Textbox(
        label="🔍 Enter Product Name",
        placeholder="Example: Sunscreen SPF 50"
    )

    analyze_btn = gr.Button("✨ Analyze Product")

    report_output = gr.Markdown()
    competitor_output = gr.Dataframe(label="🏆 Competitor Comparison")

    analyze_btn.click(
        analyze,
        inputs=product_input,
        outputs=[report_output, competitor_output]
    )

demo.launch()