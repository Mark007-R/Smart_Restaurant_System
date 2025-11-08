import os
import ast
import time
import random
import re
import requests
import pandas as pd
from urllib.parse import quote_plus, urlparse
from bs4 import BeautifulSoup
from datetime import datetime

# Headers for web scraping
HEADERS_LIST = [
    {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0 Safari/537.36"},
    {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) Safari/605.1.15"},
    {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Edge/118.0"},
    {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) Firefox/120.0"},
]

DATASET_FOLDER = "datasets"

def get_headers():
    """Return random headers for web scraping"""
    return random.choice(HEADERS_LIST)

def scramble_delay():
    """Add random delay to avoid rate limiting"""
    time.sleep(0.7 + random.random() * 1.2)

def clean_text(txt):
    """Remove excess whitespace and non-printable chars"""
    if not txt:
        return ""
    txt = re.sub(r'\s+', ' ', str(txt)).strip()
    txt = re.sub(r'[^\x20-\x7E\n]', '', txt)
    return txt

def is_valid_review(text, min_length=30):
    """Check if text is a valid review"""
    if not text or len(text) < min_length:
        return False
    
    # Filter out common non-review patterns
    invalid_patterns = [
        r'^(copyright|privacy|terms|conditions|policy)',
        r'^(login|sign up|register|subscribe)',
        r'^(home|about|contact|menu|reservation)',
        r'^\d+$',  # Just numbers
        r'^[^a-zA-Z]+$',  # No letters
    ]
    
    text_lower = text.lower()
    for pattern in invalid_patterns:
        if re.match(pattern, text_lower):
            return False
    
    return True


# ============================================================================
# LOCAL DATASET LOADERS (Priority 1)
# ============================================================================

def extract_zomato_reviews_list(reviews_list_str):
    """Extract review texts from Zomato's reviews_list column"""
    reviews = []
    if pd.isna(reviews_list_str) or not reviews_list_str:
        return reviews
    
    try:
        # Try parsing as Python literal
        parsed = ast.literal_eval(reviews_list_str)
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, tuple) and len(item) >= 2:
                    review_text = item[0]
                    if review_text and isinstance(review_text, str):
                        cleaned = clean_text(review_text)
                        if is_valid_review(cleaned):
                            reviews.append(cleaned)
        return reviews
    except:
        # Fallback: string parsing
        try:
            cleaned = str(reviews_list_str).replace('[', '').replace(']', '')
            parts = cleaned.split('(')
            for part in parts:
                if ')' in part:
                    text = part.split(')')[0].strip()
                    text = clean_text(text.strip('"').strip("'"))
                    if is_valid_review(text):
                        reviews.append(text)
        except:
            pass
    
    return reviews


def load_mumbaires_reviews(filepath, restaurant_name=None):
    """Load reviews from mumbaires.csv"""
    reviews = []
    try:
        df = pd.read_csv(filepath, encoding="utf-8", on_bad_lines="skip")
        df.columns = df.columns.str.strip()
        
        if restaurant_name:
            df = df[df['Restaurant Name'].astype(str).str.contains(
                restaurant_name, case=False, na=False, regex=False
            )]
        
        for _, row in df.iterrows():
            review_text = clean_text(row.get('Review Text', ''))
            if is_valid_review(review_text):
                reviews.append({
                    'review': review_text,
                    'restaurant': clean_text(row.get('Restaurant Name', '')),
                    'rating': row.get('Reviewer Rating'),
                    'source': 'mumbaires.csv',
                    'metadata': {
                        'address': row.get('Address'),
                        'station_area': row.get('Station Area')
                    }
                })
    except Exception as e:
        print(f"Error loading mumbaires.csv: {e}")
    
    return reviews


def load_resreviews_reviews(filepath, restaurant_name=None):
    """Load reviews from Resreviews.csv"""
    reviews = []
    try:
        df = pd.read_csv(filepath, encoding="utf-8", on_bad_lines="skip")
        df.columns = df.columns.str.strip()
        
        if restaurant_name:
            df = df[df['Restaurant'].astype(str).str.contains(
                restaurant_name, case=False, na=False, regex=False
            )]
        
        for _, row in df.iterrows():
            review_text = clean_text(row.get('Review', ''))
            if is_valid_review(review_text):
                reviews.append({
                    'review': review_text,
                    'restaurant': clean_text(row.get('Restaurant', '')),
                    'rating': row.get('Rating'),
                    'reviewer': clean_text(row.get('Reviewer', 'anonymous')),
                    'source': 'Resreviews.csv',
                    'metadata': {
                        'time': row.get('Time')
                    }
                })
    except Exception as e:
        print(f"Error loading Resreviews.csv: {e}")
    
    return reviews


def load_reviews_reviews(filepath, restaurant_name=None):
    """Load reviews from reviews.csv"""
    reviews = []
    try:
        df = pd.read_csv(filepath, encoding="utf-8", on_bad_lines="skip")
        df.columns = df.columns.str.strip()
        
        if restaurant_name:
            df = df[df['business_name'].astype(str).str.contains(
                restaurant_name, case=False, na=False, regex=False
            )]
        
        for _, row in df.iterrows():
            review_text = clean_text(row.get('text', ''))
            if is_valid_review(review_text):
                reviews.append({
                    'review': review_text,
                    'restaurant': clean_text(row.get('business_name', '')),
                    'rating': row.get('rating'),
                    'reviewer': clean_text(row.get('author_name', 'anonymous')),
                    'source': 'reviews.csv',
                    'metadata': {
                        'rating_category': row.get('rating_category')
                    }
                })
    except Exception as e:
        print(f"Error loading reviews.csv: {e}")
    
    return reviews


def load_zomato_reviews(filepath, restaurant_name=None):
    """Load reviews from zomato.csv"""
    reviews = []
    try:
        df = pd.read_csv(filepath, encoding="utf-8", on_bad_lines="skip")
        df.columns = df.columns.str.strip()
        
        if restaurant_name:
            df = df[df['name'].astype(str).str.contains(
                restaurant_name, case=False, na=False, regex=False
            )]
        
        for _, row in df.iterrows():
            restaurant = clean_text(row.get('name', ''))
            reviews_list = row.get('reviews_list', '')
            
            # Extract reviews from the tuple format
            extracted_reviews = extract_zomato_reviews_list(reviews_list)
            
            for review_text in extracted_reviews:
                reviews.append({
                    'review': review_text,
                    'restaurant': restaurant,
                    'rating': row.get('rate'),
                    'source': 'zomato.csv',
                    'metadata': {
                        'location': row.get('location'),
                        'cuisines': row.get('cuisines'),
                        'cost': row.get('approx_cost(for two people)'),
                        'online_order': row.get('online_order'),
                        'book_table': row.get('book_table')
                    }
                })
    except Exception as e:
        print(f"Error loading zomato.csv: {e}")
    
    return reviews


def load_zomato2_reviews(filepath, restaurant_name=None):
    """Load insights from zomato2.csv (creates contextual reviews from items)"""
    reviews = []
    try:
        df = pd.read_csv(filepath, encoding="utf-8", on_bad_lines="skip")
        df.columns = df.columns.str.strip()
        
        if restaurant_name:
            df = df[df['Restaurant_Name'].astype(str).str.contains(
                restaurant_name, case=False, na=False, regex=False
            )]
        
        # Group by restaurant to get aggregated insights
        restaurant_groups = df.groupby('Restaurant_Name')
        
        for restaurant, group in restaurant_groups:
            # Create summary reviews from popular items
            items = []
            for _, row in group.iterrows():
                item = clean_text(row.get('Item_Name', ''))
                if item and item.lower() != 'nan':
                    if row.get('Best_Seller', False):
                        items.append(f"{item} (Best Seller)")
                    else:
                        items.append(item)
            
            if items and len(items) > 0:
                # Create a synthetic but informative review
                review_text = f"Popular items at this restaurant include: {', '.join(items[:5])}"
                if len(items) > 5:
                    review_text += f" and {len(items) - 5} more items"
                
                avg_rating = group['Average_Rating'].mean()
                reviews.append({
                    'review': review_text,
                    'restaurant': restaurant,
                    'rating': avg_rating,
                    'source': 'zomato2.csv',
                    'metadata': {
                        'cuisine': group['Cuisine'].iloc[0] if 'Cuisine' in group.columns else None,
                        'city': group['City'].iloc[0] if 'City' in group.columns else None,
                        'place': group['Place_Name'].iloc[0] if 'Place_Name' in group.columns else None,
                        'item_count': len(items)
                    }
                })
    except Exception as e:
        print(f"Error loading zomato2.csv: {e}")
    
    return reviews


def load_local_reviews(restaurant_name=None, data_folder=DATASET_FOLDER, max_reviews=100):
    """
    Load reviews from all local CSV datasets with priority.
    Returns list of review dictionaries.
    """
    all_reviews = []
    
    if not os.path.exists(data_folder):
        print(f"Dataset folder '{data_folder}' not found.")
        return all_reviews
    
    # Define loaders with priority order
    dataset_loaders = {
        "mumbaires.csv": load_mumbaires_reviews,
        "Resreviews.csv": load_resreviews_reviews,
        "reviews.csv": load_reviews_reviews,
        "zomato.csv": load_zomato_reviews,
        "zomato2.csv": load_zomato2_reviews,
    }
    
    for filename, loader_func in dataset_loaders.items():
        filepath = os.path.join(data_folder, filename)
        if os.path.exists(filepath):
            try:
                reviews = loader_func(filepath, restaurant_name)
                all_reviews.extend(reviews)
                
                if restaurant_name and reviews:
                    print(f"✓ Loaded {len(reviews)} reviews from {filename}")
                
                # Stop if we have enough reviews
                if len(all_reviews) >= max_reviews:
                    break
                    
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    
    # Remove duplicates based on review text
    seen_texts = set()
    unique_reviews = []
    for review in all_reviews:
        text = review['review']
        if text not in seen_texts:
            seen_texts.add(text)
            unique_reviews.append(review)
    
    return unique_reviews[:max_reviews]


# ============================================================================
# WEB SCRAPING FUNCTIONS (Priority 2 - Fallback only)
# ============================================================================

def scrape_generic_reviews(restaurant_name, max_reviews=20):
    """
    Scrape general reviews using DuckDuckGo HTML fallback.
    Only used when local datasets don't have enough data.
    """
    results = []
    seen = set()
    
    try:
        query = f"{restaurant_name} restaurant customer reviews Mumbai"
        url = "https://html.duckduckgo.com/html/"
        
        print(f"🌐 Web scraping fallback: Searching for '{restaurant_name}'...")
        
        resp = requests.post(url, data={"q": query}, headers=get_headers(), timeout=10)
        if resp.status_code != 200:
            print(f"⚠️ Web search failed with status {resp.status_code}")
            return []

        soup = BeautifulSoup(resp.text, "html.parser")
        links = [a.get("href") for a in soup.find_all("a", href=True)]
        
        # Filter for review-related links
        review_links = [
            l for l in links 
            if any(x in l.lower() for x in ["review", "tripadvisor", "zomato", "yelp", "blog", "dineout"])
        ][:8]
        
        print(f"Found {len(review_links)} potential review sources")

        for i, link in enumerate(review_links, 1):
            if len(results) >= max_reviews:
                break
                
            scramble_delay()
            
            try:
                print(f"  Scraping source {i}/{len(review_links)}...")
                r = requests.get(link, headers=get_headers(), timeout=8)
                
                if r.status_code != 200:
                    continue
                
                s = BeautifulSoup(r.text, "html.parser")
                
                # Look for review-like paragraphs
                ptexts = [p.get_text(strip=True) for p in s.find_all("p") 
                         if len(p.get_text(strip=True)) > 40]

                for p in ptexts:
                    p = clean_text(p)
                    
                    if not is_valid_review(p, min_length=50) or p in seen:
                        continue
                    
                    # Check if it mentions restaurant/food keywords
                    if any(word in p.lower() for word in 
                          ['food', 'dish', 'menu', 'service', 'taste', 'delicious', 
                           'restaurant', 'meal', 'ordered', 'ambience']):
                        seen.add(p)
                        results.append({
                            "source": urlparse(link).netloc,
                            "review": p,
                            "restaurant": restaurant_name
                        })
                        
                        if len(results) >= max_reviews:
                            break
                            
            except Exception as e:
                print(f"  Error scraping {link}: {e}")
                continue
                
    except Exception as e:
        print(f"❌ Generic scraping error: {e}")
        return []
    
    print(f"✓ Scraped {len(results)} reviews from web")
    return results


def scrape_zomato_placeholder(restaurant_name, max_reviews=10):
    """
    Placeholder scraper for Zomato-like content.
    Limited functionality due to dynamic content.
    """
    results = []
    seen = set()
    
    try:
        print(f"🌐 Attempting Zomato scrape for '{restaurant_name}'...")
        
        base = "https://www.zomato.com"
        url = base + "/search?q=" + quote_plus(restaurant_name)
        
        r = requests.get(url, headers=get_headers(), timeout=10)
        if r.status_code != 200:
            print(f"⚠️ Zomato search failed with status {r.status_code}")
            return results

        s = BeautifulSoup(r.text, "html.parser")
        
        # Look for text blocks that might be reviews
        possible_blocks = s.find_all(["p", "div", "span"], text=True)
        
        for tag in possible_blocks:
            if len(results) >= max_reviews:
                break
                
            txt = clean_text(tag.get_text())
            
            if not is_valid_review(txt, min_length=50):
                continue
            
            # Check for restaurant/food-related keywords
            if re.search(r"\b(food|service|taste|ambience|price|delivery|dish|menu)\b", txt, re.I):
                if txt not in seen:
                    seen.add(txt)
                    results.append({
                        "source": "zomato.com",
                        "review": txt,
                        "restaurant": restaurant_name
                    })
        
        print(f"✓ Found {len(results)} potential reviews on Zomato")
        
    except Exception as e:
        print(f"❌ Zomato scraping error: {e}")
        pass
    
    return results


# ============================================================================
# UNIFIED FUNCTION (Smart Priority System)
# ============================================================================

def scrape_reviews_combined(restaurant_name, data_folder=DATASET_FOLDER, 
                           max_reviews=50, enable_web_scraping=False):
    """
    Unified review collection function with smart priority:
    
    Priority 1: Local CSV datasets (fastest, most reliable)
    Priority 2: Web scraping (fallback, if enabled and needed)
    
    Args:
        restaurant_name: Name of restaurant to search for
        data_folder: Path to datasets folder
        max_reviews: Maximum number of reviews to collect
        enable_web_scraping: Whether to enable web scraping fallback
    
    Returns:
        List of review dictionaries with metadata
    """
    print(f"\n{'='*60}")
    print(f"🔍 Searching for reviews: '{restaurant_name}'")
    print(f"{'='*60}\n")
    
    all_reviews = []
    
    # PRIORITY 1: Load from local datasets
    print("📂 Step 1: Checking local datasets...")
    local_reviews = load_local_reviews(restaurant_name, data_folder, max_reviews)
    
    if local_reviews:
        all_reviews.extend(local_reviews)
        print(f"✅ Found {len(local_reviews)} reviews in local datasets\n")
    else:
        print("⚠️ No reviews found in local datasets\n")
    
    # PRIORITY 2: Web scraping fallback (only if enabled and needed)
    if enable_web_scraping and len(all_reviews) < max_reviews:
        remaining = max_reviews - len(all_reviews)
        print(f"📡 Step 2: Web scraping fallback (need {remaining} more reviews)...")
        
        # Try Zomato first
        zomato_reviews = scrape_zomato_placeholder(restaurant_name, max_reviews=remaining // 2)
        all_reviews.extend(zomato_reviews)
        
        # Then try generic sources if still need more
        if len(all_reviews) < max_reviews:
            remaining = max_reviews - len(all_reviews)
            generic_reviews = scrape_generic_reviews(restaurant_name, max_reviews=remaining)
            all_reviews.extend(generic_reviews)
        
        web_count = len(all_reviews) - len(local_reviews)
        if web_count > 0:
            print(f"✅ Scraped {web_count} additional reviews from web\n")
    
    # Summary
    print(f"{'='*60}")
    print(f"📊 SUMMARY: Collected {len(all_reviews)} total reviews")
    
    # Show source breakdown
    source_counts = {}
    for review in all_reviews:
        source = review.get('source', 'unknown')
        source_counts[source] = source_counts.get(source, 0) + 1
    
    for source, count in sorted(source_counts.items()):
        print(f"   • {source}: {count} reviews")
    
    print(f"{'='*60}\n")
    
    return all_reviews[:max_reviews]


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_all_restaurants_from_datasets(data_folder=DATASET_FOLDER):
    """Get list of all unique restaurants from datasets"""
    restaurants = set()
    
    if not os.path.exists(data_folder):
        return []
    
    dataset_configs = {
        "mumbaires.csv": "Restaurant Name",
        "Resreviews.csv": "Restaurant",
        "reviews.csv": "business_name",
        "zomato.csv": "name",
        "zomato2.csv": "Restaurant_Name",
    }
    
    for filename, name_col in dataset_configs.items():
        filepath = os.path.join(data_folder, filename)
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath, encoding="utf-8", on_bad_lines="skip")
                df.columns = df.columns.str.strip()
                
                if name_col in df.columns:
                    names = df[name_col].dropna().astype(str).unique()
                    for name in names:
                        if name and name.lower() != 'nan':
                            restaurants.add(clean_text(name))
            except Exception as e:
                print(f"Error reading {filename}: {e}")
    
    return sorted(list(restaurants))


def search_restaurants(query, data_folder=DATASET_FOLDER, max_results=10):
    """Search for restaurants matching a query"""
    all_restaurants = get_all_restaurants_from_datasets(data_folder)
    query_lower = query.lower()
    
    matches = [r for r in all_restaurants if query_lower in r.lower()]
    return matches[:max_results]