import re
import io
import base64
import ast
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from collections import Counter
from nltk.corpus import stopwords
import nltk
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

# Ensure stopwords are downloaded
nltk.download('stopwords', quiet=True)

analyzer = SentimentIntensityAnalyzer()
STOPWORDS = set(stopwords.words('english'))

# Enhanced complaint categories & keywords
CATEGORY_KEYWORDS = {
    "service": ["service", "wait", "waiter", "staff", "server", "attitude", "rude", "slow", 
                "unfriendly", "impolite", "ignored", "attention", "waiting"],
    "food_quality": ["cold", "burnt", "undercooked", "bland", "taste", "flavour", "flavor", 
                     "spoiled", "stale", "overcooked", "raw", "soggy", "hard", "dry", "greasy"],
    "hygiene": ["dirty", "hygiene", "clean", "unclean", "flies", "smell", "smelly", 
                "sanitation", "filthy", "unhygienic", "cockroach", "insects"],
    "price": ["expensive", "price", "cost", "overpriced", "costly", "pricey", "value", "money"],
    "delivery": ["delivery", "late", "packaging", "missing", "driver", "delayed", "order", 
                 "arrived", "cold food", "damaged"],
    "portion": ["small", "portion", "quantity", "size", "less", "tiny", "inadequate"],
    "ambience": ["ambience", "music", "noisy", "crowded", "lighting", "atmosphere", 
                 "decor", "seating", "comfort", "space"],
    "variety": ["menu", "options", "variety", "limited", "choices", "selection"],
}

DATASET_FOLDER = "datasets"

def extract_keywords(text, top_k=8):
    """Extract meaningful keywords from text, excluding stopwords"""
    text = text.lower()
    words = re.findall(r"\b[a-z]{3,}\b", text)
    words = [w for w in words if w not in STOPWORDS and len(w) > 3]
    c = Counter(words)
    return [w for w, _ in c.most_common(top_k)]

def analyze_text_and_keywords(text):
    """Analyze sentiment and extract keywords from review text"""
    vs = analyzer.polarity_scores(text)
    compound = vs.get("compound", 0.0)
    
    if compound >= 0.05:
        label = "Positive"
    elif compound <= -0.05:
        label = "Negative"
    else:
        label = "Neutral"
    
    keywords = extract_keywords(text, top_k=8)
    return label, compound, keywords

def categorize_complaints(text):
    """Categorize review into complaint categories"""
    text_l = text.lower()
    cats = []
    for cat, kws in CATEGORY_KEYWORDS.items():
        for kw in kws:
            if kw in text_l:
                cats.append(cat)
                break
    return list(dict.fromkeys(cats))

def plot_to_base64(fig):
    """Convert matplotlib figure to base64 string for HTML embedding"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return f"data:image/png;base64,{img_base64}"

def generate_visualizations(reviews):
    """
    Generate comprehensive visualizations and return as base64 encoded images
    Returns dict with image keys
    """
    images = {}
    
    counts = {}
    sentiments = {"Positive": 0, "Negative": 0, "Neutral": 0}
    sentiment_scores = []
    source_distribution = {}
    rating_distribution = []
    
    # Collect data
    for r in reviews:
        # Categories
        for cat in (r.categories or "").split(","):
            if cat.strip():
                counts[cat.strip()] = counts.get(cat.strip(), 0) + 1
        
        # Sentiments
        if hasattr(r, "sentiment") and r.sentiment:
            sentiments[r.sentiment] = sentiments.get(r.sentiment, 0) + 1
        
        # Sentiment scores
        if hasattr(r, "score") and r.score is not None:
            sentiment_scores.append(r.score)
        
        # Source distribution
        if hasattr(r, "source_file") and r.source_file:
            source = r.source_file
            source_distribution[source] = source_distribution.get(source, 0) + 1
        
        # Ratings
        if hasattr(r, "rating") and r.rating is not None:
            try:
                rating_distribution.append(float(r.rating))
            except:
                pass
    
    # 1. Complaint Category Bar Chart
    if counts:
        df_counts = pd.DataFrame(list(counts.items()), columns=["Category", "Count"])
        df_counts = df_counts.sort_values("Count", ascending=False)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = sns.barplot(data=df_counts, x="Category", y="Count", palette="viridis", ax=ax)
        ax.set_title("Complaint Frequency by Category", fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel("Category", fontsize=12, fontweight='bold')
        ax.set_ylabel("Count", fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels on bars
        for container in ax.containers:
            ax.bar_label(container, fontsize=10)
        
        plt.tight_layout()
        images['category_bar'] = plot_to_base64(fig)
    
    # 2. Sentiment Pie Chart
    if any(sentiments.values()):
        fig, ax = plt.subplots(figsize=(9, 7))
        colors = ['#66c2a5', '#fc8d62', '#8da0cb']
        explode = (0.05, 0.05, 0.05)
        
        wedges, texts, autotexts = ax.pie(
            sentiments.values(), 
            labels=sentiments.keys(), 
            autopct='%1.1f%%',
            startangle=90, 
            colors=colors,
            explode=explode,
            shadow=True
        )
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(12)
            autotext.set_fontweight('bold')
        
        ax.set_title("Sentiment Distribution", fontsize=16, fontweight='bold', pad=20)
        images['sentiment_pie'] = plot_to_base64(fig)
    
    # 3. Sentiment Bar Chart
    if any(sentiments.values()):
        df_sent = pd.DataFrame(list(sentiments.items()), columns=["Sentiment", "Count"])
        
        fig, ax = plt.subplots(figsize=(9, 6))
        bars = sns.barplot(data=df_sent, x="Sentiment", y="Count", palette="Set2", ax=ax)
        ax.set_title("Review Count by Sentiment", fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel("Sentiment", fontsize=12, fontweight='bold')
        ax.set_ylabel("Count", fontsize=12, fontweight='bold')
        
        # Add value labels
        for container in ax.containers:
            ax.bar_label(container, fontsize=11)
        
        plt.tight_layout()
        images['sentiment_bar'] = plot_to_base64(fig)
    
    # 4. Sentiment Score Distribution Histogram
    if sentiment_scores:
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.hist(sentiment_scores, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
        ax.axvline(x=0.05, color='green', linestyle='--', linewidth=2, label='Positive threshold')
        ax.axvline(x=-0.05, color='red', linestyle='--', linewidth=2, label='Negative threshold')
        ax.set_title("Distribution of Sentiment Scores", fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel("Sentiment Score", fontsize=12, fontweight='bold')
        ax.set_ylabel("Frequency", fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        images['score_hist'] = plot_to_base64(fig)
    
    # 5. Category vs Sentiment Heatmap
    if counts and any(sentiments.values()):
        category_sentiment = {}
        for r in reviews:
            cats = [c.strip() for c in (r.categories or "").split(",") if c.strip()]
            sent = r.sentiment if hasattr(r, "sentiment") and r.sentiment else "Neutral"
            for cat in cats:
                if cat not in category_sentiment:
                    category_sentiment[cat] = {"Positive": 0, "Negative": 0, "Neutral": 0}
                category_sentiment[cat][sent] += 1
        
        if category_sentiment:
            df_heatmap = pd.DataFrame(category_sentiment).T.fillna(0)
            
            fig, ax = plt.subplots(figsize=(11, 7))
            sns.heatmap(df_heatmap, annot=True, fmt='g', cmap='YlOrRd', 
                       linewidths=0.5, ax=ax, cbar_kws={'label': 'Count'})
            ax.set_title("Complaint Categories vs Sentiment", fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel("Category", fontsize=12, fontweight='bold')
            ax.set_ylabel("Sentiment", fontsize=12, fontweight='bold')
            plt.tight_layout()
            images['category_sentiment_heatmap'] = plot_to_base64(fig)
    
    # 6. Top Keywords Bar Chart
    all_keywords = []
    for r in reviews:
        if hasattr(r, 'keywords') and r.keywords:
            all_keywords.extend([k.strip() for k in r.keywords.split(',') if k.strip()])
    
    if all_keywords:
        keyword_counts = Counter(all_keywords)
        top_keywords = dict(keyword_counts.most_common(20))
        
        fig, ax = plt.subplots(figsize=(12, 8))
        keywords = list(top_keywords.keys())
        counts_list = list(top_keywords.values())
        
        ax.barh(keywords, counts_list, color='coral', edgecolor='black', alpha=0.8)
        ax.set_title("Top 20 Keywords in Reviews", fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel("Frequency", fontsize=12, fontweight='bold')
        ax.set_ylabel("Keywords", fontsize=12, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(counts_list):
            ax.text(v + 0.3, i, str(v), va='center', fontsize=9)
        
        plt.tight_layout()
        images['keywords_bar'] = plot_to_base64(fig)
    
    # 7. Data Source Distribution
    if source_distribution:
        fig, ax = plt.subplots(figsize=(10, 6))
        sources = list(source_distribution.keys())
        counts_list = list(source_distribution.values())
        
        colors_list = plt.cm.Set3(range(len(sources)))
        ax.bar(sources, counts_list, color=colors_list, edgecolor='black', alpha=0.8)
        ax.set_title("Reviews by Data Source", fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel("Data Source", fontsize=12, fontweight='bold')
        ax.set_ylabel("Number of Reviews", fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels
        for i, v in enumerate(counts_list):
            ax.text(i, v + 0.5, str(v), ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        images['source_distribution'] = plot_to_base64(fig)
    
    # 8. Rating Distribution
    if rating_distribution:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(rating_distribution, bins=20, color='mediumseagreen', 
                edgecolor='black', alpha=0.7)
        ax.axvline(x=sum(rating_distribution)/len(rating_distribution), 
                   color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {sum(rating_distribution)/len(rating_distribution):.2f}')
        ax.set_title("Rating Distribution", fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel("Rating", fontsize=12, fontweight='bold')
        ax.set_ylabel("Frequency", fontsize=12, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        images['rating_dist'] = plot_to_base64(fig)
    
    # 9. Sentiment Trend (if timestamp available)
    reviews_with_time = [r for r in reviews if hasattr(r, 'created_at') and r.created_at]
    if len(reviews_with_time) > 10:
        df_timeline = pd.DataFrame([
            {
                'date': r.created_at,
                'sentiment': r.sentiment if hasattr(r, 'sentiment') else 'Neutral',
                'score': r.score if hasattr(r, 'score') else 0
            }
            for r in reviews_with_time
        ])
        
        df_timeline['date'] = pd.to_datetime(df_timeline['date'])
        df_timeline = df_timeline.sort_values('date')
        df_timeline['week'] = df_timeline['date'].dt.to_period('W')
        
        weekly_sentiment = df_timeline.groupby(['week', 'sentiment']).size().unstack(fill_value=0)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        weekly_sentiment.plot(kind='line', ax=ax, marker='o', linewidth=2)
        ax.set_title("Sentiment Trend Over Time", fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel("Week", fontsize=12, fontweight='bold')
        ax.set_ylabel("Number of Reviews", fontsize=12, fontweight='bold')
        ax.legend(title='Sentiment', fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        images['sentiment_trend'] = plot_to_base64(fig)
    
    return images


def extract_reviews_from_zomato_list(reviews_list_str):
    """Extract review texts from Zomato's reviews_list column"""
    reviews = []
    if pd.isna(reviews_list_str) or not reviews_list_str:
        return reviews
    
    try:
        parsed = ast.literal_eval(reviews_list_str)
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, tuple) and len(item) >= 2:
                    review_text = item[0]
                    if review_text and isinstance(review_text, str) and len(review_text) > 10:
                        reviews.append(review_text.strip())
        return reviews
    except:
        try:
            cleaned = str(reviews_list_str).replace('[', '').replace(']', '')
            parts = cleaned.split('(')
            for part in parts:
                if ')' in part:
                    text = part.split(')')[0].strip()
                    if text and len(text) > 10:
                        reviews.append(text.strip('"').strip("'"))
        except:
            pass
    
    return reviews


def get_reviews_from_datasets(restaurant_name):
    """
    Fetch reviews from all dataset CSVs for a specific restaurant
    Returns list of review dictionaries with metadata
    """
    all_reviews = []
    
    dataset_files = {
        "mumbaires.csv": {
            "name_col": "Restaurant Name",
            "review_col": "Review Text",
            "rating_col": "Reviewer Rating"
        },
        "Resreviews.csv": {
            "name_col": "Restaurant",
            "review_col": "Review",
            "rating_col": "Rating"
        },
        "reviews.csv": {
            "name_col": "business_name",
            "review_col": "text",
            "rating_col": "rating"
        },
        "zomato.csv": {
            "name_col": "name",
            "review_col": "reviews_list",  # Special handling needed
            "rating_col": "rate"
        },
        "zomato2.csv": {
            "name_col": "Restaurant_Name",
            "review_col": None,  # No direct review column
            "rating_col": "Average_Rating"
        }
    }
    
    for fname, config in dataset_files.items():
        fpath = os.path.join(DATASET_FOLDER, fname)
        if not os.path.exists(fpath):
            continue
        
        try:
            df = pd.read_csv(fpath, encoding="utf-8", on_bad_lines="skip")
            df.columns = df.columns.str.strip()
            
            # Filter by restaurant name
            name_col = config["name_col"]
            if name_col not in df.columns:
                continue
            
            df_filtered = df[df[name_col].astype(str).str.contains(
                restaurant_name, case=False, na=False, regex=False
            )]
            
            if df_filtered.empty:
                continue
            
            # Extract reviews based on file type
            if fname == "zomato.csv":
                # Special handling for zomato reviews_list
                for _, row in df_filtered.iterrows():
                    reviews_list = row.get(config["review_col"], "")
                    extracted = extract_reviews_from_zomato_list(reviews_list)
                    for review_text in extracted:
                        all_reviews.append({
                            'text': review_text,
                            'rating': row.get(config["rating_col"]),
                            'source': fname
                        })
            
            elif fname == "zomato2.csv":
                # Create synthetic insights from items
                for _, row in df_filtered.iterrows():
                    item = row.get('Item_Name', '')
                    if item and str(item).lower() != 'nan':
                        review_text = f"Tried {item}"
                        if row.get('Best_Seller', False):
                            review_text += " - highly popular item"
                        all_reviews.append({
                            'text': review_text,
                            'rating': row.get(config["rating_col"]),
                            'source': fname
                        })
            
            else:
                # Standard review extraction
                review_col = config["review_col"]
                if review_col and review_col in df_filtered.columns:
                    for _, row in df_filtered.iterrows():
                        review_text = str(row.get(review_col, '')).strip()
                        if review_text and review_text.lower() != 'nan' and len(review_text) > 10:
                            all_reviews.append({
                                'text': review_text,
                                'rating': row.get(config["rating_col"]),
                                'source': fname
                            })
        
        except Exception as e:
            print(f"Error reading {fname}: {e}")
    
    return all_reviews


def summarize_reviews_for_recommendations(reviews, restaurant_name=None):
    """
    Generate actionable recommendations based on review analysis
    Includes both database reviews and dataset reviews
    """
    counts = {}
    sentiments = {"Positive": 0, "Negative": 0, "Neutral": 0}
    total_reviews = len(reviews)
    
    # Process database reviews
    for r in reviews:
        # Count categories
        for cat in (r.categories or "").split(","):
            if cat.strip():
                counts[cat.strip()] = counts.get(cat.strip(), 0) + 1
        
        # Count sentiments
        if hasattr(r, "sentiment") and r.sentiment:
            sentiments[r.sentiment] = sentiments.get(r.sentiment, 0) + 1
    
    # Include dataset reviews if restaurant name provided
    if restaurant_name:
        dataset_reviews = get_reviews_from_datasets(restaurant_name)
        total_reviews += len(dataset_reviews)
        
        for review_dict in dataset_reviews:
            text = review_dict['text']
            sent, score, _ = analyze_text_and_keywords(text)
            sentiments[sent] = sentiments.get(sent, 0) + 1
            
            for cat in categorize_complaints(text):
                counts[cat] = counts.get(cat, 0) + 1
    
    # Calculate percentages
    category_percentages = {}
    if total_reviews > 0:
        for cat, count in counts.items():
            category_percentages[cat] = round((count / total_reviews) * 100, 1)
    
    # Generate recommendations based on complaint frequency
    recs = []
    priority_recs = []
    standard_recs = []
    
    # High priority (>20% of reviews)
    if category_percentages.get("service", 0) > 20:
        priority_recs.append({
            "category": "service",
            "priority": "HIGH",
            "recommendation": "🚨 URGENT: Train staff on customer service and reduce wait times",
            "percentage": category_percentages["service"]
        })
    elif counts.get("service", 0) > 0:
        standard_recs.append({
            "category": "service",
            "priority": "MEDIUM",
            "recommendation": "🧑‍🍳 Improve staff training and response efficiency",
            "percentage": category_percentages.get("service", 0)
        })
    
    if category_percentages.get("food_quality", 0) > 20:
        priority_recs.append({
            "category": "food_quality",
            "priority": "HIGH",
            "recommendation": "🚨 CRITICAL: Implement strict quality control in kitchen operations",
            "percentage": category_percentages["food_quality"]
        })
    elif counts.get("food_quality", 0) > 0:
        standard_recs.append({
            "category": "food_quality",
            "priority": "MEDIUM",
            "recommendation": "🍽️ Enhance food quality with regular kitchen audits",
            "percentage": category_percentages.get("food_quality", 0)
        })
    
    if category_percentages.get("hygiene", 0) > 15:
        priority_recs.append({
            "category": "hygiene",
            "priority": "HIGH",
            "recommendation": "🚨 CRITICAL: Immediate deep cleaning and hygiene protocol enforcement",
            "percentage": category_percentages["hygiene"]
        })
    elif counts.get("hygiene", 0) > 0:
        standard_recs.append({
            "category": "hygiene",
            "priority": "MEDIUM",
            "recommendation": "🧼 Improve cleanliness standards and visible hygiene practices",
            "percentage": category_percentages.get("hygiene", 0)
        })
    
    if counts.get("price", 0) > 0:
        standard_recs.append({
            "category": "price",
            "priority": "MEDIUM",
            "recommendation": "💰 Review pricing strategy and introduce value deals/combos",
            "percentage": category_percentages.get("price", 0)
        })
    
    if counts.get("delivery", 0) > 0:
        standard_recs.append({
            "category": "delivery",
            "priority": "MEDIUM",
            "recommendation": "🚚 Optimize delivery logistics and packaging quality",
            "percentage": category_percentages.get("delivery", 0)
        })
    
    if counts.get("portion", 0) > 0:
        standard_recs.append({
            "category": "portion",
            "priority": "LOW",
            "recommendation": "🍛 Standardize portion sizes to meet customer expectations",
            "percentage": category_percentages.get("portion", 0)
        })
    
    if counts.get("ambience", 0) > 0:
        standard_recs.append({
            "category": "ambience",
            "priority": "LOW",
            "recommendation": "🎶 Enhance ambience with better decor, lighting, and seating",
            "percentage": category_percentages.get("ambience", 0)
        })
    
    if counts.get("variety", 0) > 0:
        standard_recs.append({
            "category": "variety",
            "priority": "LOW",
            "recommendation": "📋 Expand menu variety and introduce seasonal specials",
            "percentage": category_percentages.get("variety", 0)
        })
    
    # Combine recommendations
    recs = priority_recs + standard_recs
    
    # If no issues found
    if not recs:
        positive_percentage = (sentiments.get("Positive", 0) / total_reviews * 100) if total_reviews > 0 else 0
        recs.append({
            "category": "overall",
            "priority": "INFO",
            "recommendation": f"✅ Overall sentiment is positive ({positive_percentage:.1f}%); maintain quality consistency",
            "percentage": positive_percentage
        })
    
    # Add sentiment-based insights
    if total_reviews > 0:
        neg_percentage = (sentiments.get("Negative", 0) / total_reviews) * 100
        pos_percentage = (sentiments.get("Positive", 0) / total_reviews) * 100
        
        if neg_percentage > 40:
            recs.insert(0, {
                "category": "sentiment",
                "priority": "CRITICAL",
                "recommendation": f"⚠️ ALERT: {neg_percentage:.1f}% negative reviews - immediate action required!",
                "percentage": neg_percentage
            })
        elif pos_percentage > 70:
            recs.append({
                "category": "sentiment",
                "priority": "INFO",
                "recommendation": f"🌟 Excellent performance with {pos_percentage:.1f}% positive reviews!",
                "percentage": pos_percentage
            })
    
    return recs, counts, sentiments