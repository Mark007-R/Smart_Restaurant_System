import re
import io
import base64
import ast
import numpy as np
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

nltk.download('stopwords', quiet=True)

analyzer = SentimentIntensityAnalyzer()
STOPWORDS = set(stopwords.words('english'))

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
    text = text.lower()
    words = re.findall(r"\b[a-z]{3,}\b", text)
    words = [w for w in words if w not in STOPWORDS and len(w) > 3]
    c = Counter(words)
    return [w for w, _ in c.most_common(top_k)]

def analyze_text_and_keywords(text):
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
    text_l = text.lower()
    cats = []
    for cat, kws in CATEGORY_KEYWORDS.items():
        for kw in kws:
            if kw in text_l:
                cats.append(cat)
                break
    return list(dict.fromkeys(cats))

def plot_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=120, facecolor='white')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return f"data:image/png;base64,{img_base64}"

def get_restaurant_info(restaurant_name):
    info = {
        'name': restaurant_name,
        'rating': None,
        'address': None,
        'cuisines': None,
        'cost': None,
        'location': None,
        'description': None
    }
    
    dataset_files = {
        "mumbaires.csv": {
            "name_col": "Restaurant Name",
            "rating_col": "Rating",
            "address_col": "Address",
            "price_col": "Price Level"
        },
        "zomato.csv": {
            "name_col": "name",
            "rating_col": "rate",
            "address_col": "address",
            "location_col": "location",
            "cuisines_col": "cuisines",
            "cost_col": "approx_cost(for two people)"
        },
        "zomato2.csv": {
            "name_col": "Restaurant_Name",
            "rating_col": "Avg_Rating_Restaurant",
            "location_col": "Place_Name",
            "city_col": "City",
            "cuisine_col": "Cuisine"
        }
    }
    
    for fname, config in dataset_files.items():
        fpath = os.path.join(DATASET_FOLDER, fname)
        if not os.path.exists(fpath):
            continue
        
        try:
            df = pd.read_csv(fpath, encoding="utf-8", on_bad_lines="skip")
            df.columns = df.columns.str.strip()
            
            name_col = config["name_col"]
            if name_col not in df.columns:
                continue
            
            df_filtered = df[df[name_col].astype(str).str.contains(
                restaurant_name, case=False, na=False, regex=False
            )]
            
            if not df_filtered.empty:
                row = df_filtered.iloc[0]
                
                if 'rating_col' in config and config['rating_col'] in df.columns:
                    rating = row.get(config['rating_col'])
                    if pd.notna(rating):
                        info['rating'] = rating
                
                if 'address_col' in config and config['address_col'] in df.columns:
                    address = row.get(config['address_col'])
                    if pd.notna(address):
                        info['address'] = address
                
                if 'cuisines_col' in config and config['cuisines_col'] in df.columns:
                    cuisines = row.get(config['cuisines_col'])
                    if pd.notna(cuisines):
                        info['cuisines'] = cuisines
                
                if 'cost_col' in config and config['cost_col'] in df.columns:
                    cost = row.get(config['cost_col'])
                    if pd.notna(cost):
                        info['cost'] = cost
                
                if 'location_col' in config and config['location_col'] in df.columns:
                    location = row.get(config['location_col'])
                    if pd.notna(location):
                        info['location'] = location
                
                if 'cuisine_col' in config and config['cuisine_col'] in df.columns:
                    cuisine = row.get(config['cuisine_col'])
                    if pd.notna(cuisine):
                        info['cuisines'] = cuisine
                
                if info['rating'] or info['address']:
                    break
        
        except Exception as e:
            print(f"Error reading {fname}: {e}")
    
    return info

def generate_visualizations(reviews):
    images = {}
    plt.style.use('dark_background')
    
    counts = {}
    sentiments = {"Positive": 0, "Negative": 0, "Neutral": 0}
    sentiment_scores = []
    rating_distribution = []
    category_sentiment_map = {}
    
    for r in reviews:
        cats = [c.strip() for c in (r.categories or "").split(",") if c.strip()]
        sent = r.sentiment if hasattr(r, "sentiment") and r.sentiment else "Neutral"
        
        for cat in cats:
            if cat not in category_sentiment_map:
                category_sentiment_map[cat] = {"Positive": 0, "Negative": 0, "Neutral": 0}
            category_sentiment_map[cat][sent] += 1
            counts[cat] = counts.get(cat, 0) + 1
        
        if hasattr(r, "sentiment") and r.sentiment:
            sentiments[r.sentiment] = sentiments.get(r.sentiment, 0) + 1
        
        if hasattr(r, "score") and r.score is not None:
            sentiment_scores.append(r.score)
        
        if hasattr(r, "rating") and r.rating is not None:
            try:
                rating_distribution.append(float(r.rating))
            except:
                pass
    
    if reviews:
        restaurant_name = reviews[0].restaurant
        restaurant_info = get_restaurant_info(restaurant_name)
        images['restaurant_info'] = restaurant_info
    
    if counts:
        problem_severity = {}
        for cat, sentiments_dict in category_sentiment_map.items():
            negative_count = sentiments_dict.get("Negative", 0)
            total_count = sum(sentiments_dict.values())
            problem_severity[cat] = {
                'negative_count': negative_count,
                'total_count': total_count,
                'neg_percentage': (negative_count / total_count * 100) if total_count > 0 else 0
            }
        
        sorted_problems = sorted(problem_severity.items(), 
                                key=lambda x: x[1]['neg_percentage'], reverse=True)
        
        fig, ax = plt.subplots(figsize=(14, 7))
        categories = [cat for cat, _ in sorted_problems]
        neg_percentages = [data['neg_percentage'] for _, data in sorted_problems]
        colors_severity = ['#ef4444' if p > 50 else '#f97316' if p > 30 else '#fbbf24' 
                          for p in neg_percentages]
        
        bars = ax.barh(categories, neg_percentages, color=colors_severity, 
                       edgecolor='#14b8a6', linewidth=2, alpha=0.85)
        
        ax.set_title("🚨 Problem Areas: Negative Sentiment by Issue", 
                    fontsize=20, fontweight='bold', pad=25, color='#e2e8f0')
        ax.set_xlabel("% of Issues with Negative Reviews", fontsize=14, fontweight='bold', color='#e2e8f0')
        ax.set_ylabel("Issue Category", fontsize=14, fontweight='bold', color='#e2e8f0')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.set_facecolor('#0f172a')
        
        for i, (bar, p) in enumerate(zip(bars, neg_percentages)):
            ax.text(p + 1.5, i, f'{p:.1f}%', va='center', 
                   fontsize=11, fontweight='bold', color='#22c55e')
        
        plt.tight_layout()
        images['problem_areas'] = plot_to_base64(fig)
    
    if any(sentiments.values()):
        total = sum(sentiments.values())
        pos_pct = (sentiments.get("Positive", 0) / total * 100) if total > 0 else 0
        neg_pct = (sentiments.get("Negative", 0) / total * 100) if total > 0 else 0
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        categories = ['Positive', 'Negative', 'Neutral']
        values = [sentiments.get(c, 0) for c in categories]
        colors_sent = ['#22c55e', '#ef4444', '#64748b']
        
        bars = ax.bar(categories, values, color=colors_sent, edgecolor='#14b8a6', 
                     linewidth=2.5, alpha=0.8, width=0.6)
        
        ax.set_title("💯 Overall Sentiment Health Score", fontsize=20, fontweight='bold', 
                    pad=25, color='#e2e8f0')
        ax.set_ylabel("Number of Reviews", fontsize=14, fontweight='bold', color='#e2e8f0')
        ax.set_facecolor('#0f172a')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,
                   f'{int(val)}\n({val/total*100:.1f}%)',
                   ha='center', va='bottom', fontsize=12, fontweight='bold', 
                   color='#e2e8f0')
        
        plt.tight_layout()
        images['sentiment_gauge'] = plot_to_base64(fig)
    
    if category_sentiment_map:
        fig, ax = plt.subplots(figsize=(14, 8))
        
        category_names = list(category_sentiment_map.keys())
        pos_counts = [category_sentiment_map[cat].get("Positive", 0) for cat in category_names]
        neg_counts = [category_sentiment_map[cat].get("Negative", 0) for cat in category_names]
        neu_counts = [category_sentiment_map[cat].get("Neutral", 0) for cat in category_names]
        
        x = range(len(category_names))
        width = 0.6
        
        ax.bar(x, pos_counts, width, label='Positive', color='#22c55e', edgecolor='#14b8a6', linewidth=1.5, alpha=0.8)
        ax.bar(x, neg_counts, width, bottom=pos_counts, label='Negative', color='#ef4444', edgecolor='#14b8a6', linewidth=1.5, alpha=0.8)
        ax.bar(x, neu_counts, width, bottom=[p+n for p,n in zip(pos_counts, neg_counts)], 
               label='Neutral', color='#64748b', edgecolor='#14b8a6', linewidth=1.5, alpha=0.8)
        
        ax.set_title("📊 Issue Breakdown: Sentiment per Category", fontsize=20, fontweight='bold', 
                    pad=25, color='#e2e8f0')
        ax.set_ylabel("Number of Mentions", fontsize=14, fontweight='bold', color='#e2e8f0')
        ax.set_xlabel("Issue Type", fontsize=14, fontweight='bold', color='#e2e8f0')
        ax.set_xticks(x)
        ax.set_xticklabels(category_names, rotation=45, ha='right', fontsize=11)
        ax.legend(fontsize=12, loc='upper right', framealpha=0.95)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_facecolor('#0f172a')
        
        plt.tight_layout()
        images['issue_breakdown'] = plot_to_base64(fig)
    
    if sentiment_scores:
        fig, ax = plt.subplots(figsize=(13, 6))
        
        sorted_scores = sorted(sentiment_scores)
        window = max(3, len(sorted_scores) // 10)
        moving_avg = pd.Series(sorted_scores).rolling(window=window, center=True).mean()
        
        ax.scatter(range(len(sorted_scores)), sorted_scores, alpha=0.4, s=60, 
                  color='#14b8a6', edgecolor='#0ea5e9', linewidth=0.8, label='Individual Reviews')
        ax.plot(range(len(sorted_scores)), moving_avg, color='#f59e0b', linewidth=3, 
               label=f'Trend (MA-{window})', alpha=0.9)
        ax.axhline(y=0, color='#64748b', linestyle='--', linewidth=2, alpha=0.6, label='Neutral Threshold')
        
        ax.fill_between(range(len(sorted_scores)), 0, sorted_scores, 
                       where=(sorted_scores >= 0), alpha=0.2, color='#22c55e', label='Positive Zone')
        ax.fill_between(range(len(sorted_scores)), 0, sorted_scores, 
                       where=(sorted_scores < 0), alpha=0.2, color='#ef4444', label='Negative Zone')
        
        ax.set_title("📈 Sentiment Trend Analysis", fontsize=20, fontweight='bold', 
                    pad=25, color='#e2e8f0')
        ax.set_xlabel("Review Index (Chronological)", fontsize=14, fontweight='bold', color='#e2e8f0')
        ax.set_ylabel("Sentiment Score", fontsize=14, fontweight='bold', color='#e2e8f0')
        ax.legend(fontsize=11, loc='best', framealpha=0.95)
        ax.grid(alpha=0.3, linestyle='--')
        ax.set_facecolor('#0f172a')
        
        plt.tight_layout()
        images['sentiment_trend'] = plot_to_base64(fig)
    
    if rating_distribution:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bins = [0, 1, 2, 3, 4, 5, 6]
        n, bins_edges, patches = ax.hist(rating_distribution, bins=bins, 
                                         color='#14b8a6', edgecolor='#0ea5e9', 
                                         linewidth=2, alpha=0.8)
        
        colors_rating = ['#ef4444', '#f97316', '#fbbf24', '#a3e635', '#22c55e']
        for patch, color in zip(patches, colors_rating):
            patch.set_facecolor(color)
        
        mean_rating = sum(rating_distribution) / len(rating_distribution)
        ax.axvline(x=mean_rating, color='#f59e0b', linestyle='--', linewidth=3, 
                  label=f'Average: {mean_rating:.2f}', alpha=0.9)
        
        ax.set_title("⭐ Customer Rating Distribution", fontsize=20, fontweight='bold', 
                    pad=25, color='#e2e8f0')
        ax.set_xlabel("Rating (1-5 scale)", fontsize=14, fontweight='bold', color='#e2e8f0')
        ax.set_ylabel("Number of Customers", fontsize=14, fontweight='bold', color='#e2e8f0')
        ax.legend(fontsize=12, loc='upper right', framealpha=0.95)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_facecolor('#0f172a')
        
        for height, left in zip(n, bins_edges[:-1]):
            if height > 0:
                ax.text(left + 0.5, height + max(n)*0.01, f'{int(height)}', 
                       ha='center', va='bottom', fontsize=10, fontweight='bold', color='#e2e8f0')
        
        plt.tight_layout()
        images['ratings'] = plot_to_base64(fig)
    
    all_keywords = []
    for r in reviews:
        if hasattr(r, 'keywords') and r.keywords:
            all_keywords.extend([k.strip() for k in r.keywords.split(',') if k.strip()])
    
    if all_keywords:
        keyword_counts = Counter(all_keywords)
        top_keywords = dict(keyword_counts.most_common(12))
        
        fig, ax = plt.subplots(figsize=(13, 8))
        keywords_list = list(top_keywords.keys())
        counts_list = list(top_keywords.values())
        
        colors_gradient = plt.cm.RdYlGn_r(np.linspace(0.3, 0.9, len(keywords_list)))
        bars = ax.barh(keywords_list, counts_list, color=colors_gradient, 
                      edgecolor='#14b8a6', linewidth=2, alpha=0.85)
        
        ax.set_title("🎯 Top Customer Concerns & Keywords", fontsize=20, fontweight='bold', 
                    pad=25, color='#e2e8f0')
        ax.set_xlabel("Frequency", fontsize=14, fontweight='bold', color='#e2e8f0')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.set_facecolor('#0f172a')
        
        for i, (bar, count) in enumerate(zip(bars, counts_list)):
            ax.text(count + 0.2, i, str(count), va='center', fontsize=11, 
                   fontweight='bold', color='#22c55e')
        
        plt.tight_layout()
        images['keywords'] = plot_to_base64(fig)
    
    if counts:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        neg_issues = sorted([(cat, category_sentiment_map[cat].get("Negative", 0)) 
                            for cat in category_sentiment_map.keys()],
                           key=lambda x: x[1], reverse=True)[:8]
        neg_cats, neg_vals = zip(*neg_issues) if neg_issues else ([], [])
        
        bars1 = ax1.barh(neg_cats, neg_vals, color='#ef4444', edgecolor='#dc2626', 
                        linewidth=2, alpha=0.8)
        ax1.set_title("🔴 Top Negative Issues", fontsize=16, fontweight='bold', 
                     color='#e2e8f0')
        ax1.set_xlabel("Count", fontsize=12, fontweight='bold', color='#e2e8f0')
        ax1.invert_yaxis()
        ax1.grid(axis='x', alpha=0.3, linestyle='--')
        ax1.set_facecolor('#0f172a')
        
        for bar, val in zip(bars1, neg_vals):
            ax1.text(val + 0.2, bar.get_y() + bar.get_height()/2, str(int(val)), 
                    va='center', fontsize=10, fontweight='bold', color='#fca5a5')
        
        pos_issues = sorted([(cat, category_sentiment_map[cat].get("Positive", 0)) 
                            for cat in category_sentiment_map.keys()],
                           key=lambda x: x[1], reverse=True)[:8]
        pos_cats, pos_vals = zip(*pos_issues) if pos_issues else ([], [])
        
        bars2 = ax2.barh(pos_cats, pos_vals, color='#22c55e', edgecolor='#16a34a', 
                        linewidth=2, alpha=0.8)
        ax2.set_title("🟢 Top Praised Attributes", fontsize=16, fontweight='bold', 
                     color='#e2e8f0')
        ax2.set_xlabel("Count", fontsize=12, fontweight='bold', color='#e2e8f0')
        ax2.invert_yaxis()
        ax2.grid(axis='x', alpha=0.3, linestyle='--')
        ax2.set_facecolor('#0f172a')
        
        for bar, val in zip(bars2, pos_vals):
            ax2.text(val + 0.2, bar.get_y() + bar.get_height()/2, str(int(val)), 
                    va='center', fontsize=10, fontweight='bold', color='#bbf7d0')
        
        plt.tight_layout()
        images['positive_vs_negative'] = plot_to_base64(fig)
    
    return images

def extract_reviews_from_zomato_list(reviews_list_str):
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
            "review_col": "reviews_list",
            "rating_col": "rate"
        },
        "zomato2.csv": {
            "name_col": "Restaurant_Name",
            "review_col": None,
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
            
            name_col = config["name_col"]
            if name_col not in df.columns:
                continue
            
            df_filtered = df[df[name_col].astype(str).str.contains(
                restaurant_name, case=False, na=False, regex=False
            )]
            
            if df_filtered.empty:
                continue
            
            if fname == "zomato.csv":
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
    counts = {}
    sentiments = {"Positive": 0, "Negative": 0, "Neutral": 0}
    total_reviews = len(reviews)
    
    for r in reviews:
        for cat in (r.categories or "").split(","):
            if cat.strip():
                counts[cat.strip()] = counts.get(cat.strip(), 0) + 1
        
        if hasattr(r, "sentiment") and r.sentiment:
            sentiments[r.sentiment] = sentiments.get(r.sentiment, 0) + 1
    
    if restaurant_name:
        dataset_reviews = get_reviews_from_datasets(restaurant_name)
        total_reviews += len(dataset_reviews)
        
        for review_dict in dataset_reviews:
            text = review_dict['text']
            sent, score, _ = analyze_text_and_keywords(text)
            sentiments[sent] = sentiments.get(sent, 0) + 1
            
            for cat in categorize_complaints(text):
                counts[cat] = counts.get(cat, 0) + 1
    
    category_percentages = {}
    if total_reviews > 0:
        for cat, count in counts.items():
            category_percentages[cat] = round((count / total_reviews) * 100, 1)
    
    recs = []
    priority_recs = []
    standard_recs = []
    
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
    
    recs = priority_recs + standard_recs
    
    if not recs:
        positive_percentage = (sentiments.get("Positive", 0) / total_reviews * 100) if total_reviews > 0 else 0
        recs.append({
            "category": "overall",
            "priority": "INFO",
            "recommendation": f"✅ Overall sentiment is positive ({positive_percentage:.1f}%); maintain quality consistency",
            "percentage": positive_percentage
        })
    
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