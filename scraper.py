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
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# Headers for web scraping
HEADERS_LIST = [
    {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"},
    {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15"},
    {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0"},
    {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:120.0) Gecko/20100101 Firefox/120.0"},
    {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0"},
]

DATASET_FOLDER = "datasets"

# Quality thresholds
MIN_REVIEW_LENGTH = 30
MAX_REVIEW_LENGTH = 2000
MIN_WORD_COUNT = 5

def get_headers():
    """Return random headers for web scraping"""
    return random.choice(HEADERS_LIST)

def smart_delay(min_delay=0.5, max_delay=2.0):
    """Intelligent delay with randomization to avoid detection"""
    delay = min_delay + random.random() * (max_delay - min_delay)
    time.sleep(delay)

def clean_text(txt):
    """Enhanced text cleaning with better whitespace handling"""
    if not txt:
        return ""
    
    # Remove excess whitespace
    txt = re.sub(r'\s+', ' ', str(txt)).strip()
    
    # Remove non-printable characters but keep common punctuation
    txt = re.sub(r'[^\x20-\x7E\n\t]', '', txt)
    
    # Remove multiple consecutive punctuation
    txt = re.sub(r'([.!?])\1+', r'\1', txt)
    
    # Fix spacing around punctuation
    txt = re.sub(r'\s+([.,!?;:])', r'\1', txt)
    txt = re.sub(r'([.,!?;:])\s*([.,!?;:])', r'\1 \2', txt)
    
    return txt.strip()

def is_valid_review(text, min_length=MIN_REVIEW_LENGTH, max_length=MAX_REVIEW_LENGTH):
    """Enhanced review validation with multiple quality checks"""
    if not text or not isinstance(text, str):
        return False
    
    text = text.strip()
    
    # Length checks
    if len(text) < min_length or len(text) > max_length:
        return False
    
    # Word count check
    words = text.split()
    if len(words) < MIN_WORD_COUNT:
        return False
    
    # Must contain letters
    if not re.search(r'[a-zA-Z]', text):
        return False
    
    # Filter out common non-review patterns
    invalid_patterns = [
        r'^(copyright|privacy|terms|conditions|policy|disclaimer)',
        r'^(login|sign up|register|subscribe|newsletter)',
        r'^(home|about|contact|menu|reservation|book|order)',
        r'^\d+$',  # Just numbers
        r'^[^a-zA-Z]+$',  # No letters
        r'^(photo|image|video|gallery)',
        r'^(facebook|twitter|instagram|social)',
        r'(cookie|gdpr|accept|decline)\s*(policy|notice)',
        r'^\s*[\d\W]+\s*$',  # Only digits and special chars
    ]
    
    text_lower = text.lower()
    for pattern in invalid_patterns:
        if re.search(pattern, text_lower):
            return False
    
    # Check for minimum sentence structure
    if not re.search(r'[.!?]', text):
        # If no sentence endings, check for reasonable content
        if len(words) < 10:
            return False
    
    # Calculate letter to total character ratio
    letter_count = sum(c.isalpha() for c in text)
    if letter_count / len(text) < 0.5:  # Less than 50% letters
        return False
    
    return True

def calculate_review_quality_score(text):
    """Calculate a quality score for a review (0-100)"""
    if not text:
        return 0
    
    score = 50  # Base score
    
    # Length bonus (sweet spot is 100-500 chars)
    length = len(text)
    if 100 <= length <= 500:
        score += 20
    elif 50 <= length < 100:
        score += 10
    elif length > 500:
        score += 10
    
    # Sentence structure bonus
    sentences = len(re.findall(r'[.!?]+', text))
    if sentences >= 2:
        score += 15
    
    # Vocabulary diversity
    words = text.lower().split()
    unique_ratio = len(set(words)) / len(words) if words else 0
    score += int(unique_ratio * 15)
    
    # Food/restaurant keywords bonus
    food_keywords = ['food', 'dish', 'menu', 'taste', 'delicious', 'flavor', 
                     'service', 'staff', 'ambience', 'restaurant', 'meal', 
                     'ordered', 'tried', 'recommend', 'experience']
    keyword_count = sum(1 for word in food_keywords if word in text.lower())
    score += min(keyword_count * 2, 10)
    
    return min(score, 100)


# ============================================================================
# LOCAL DATASET LOADERS (Priority 1) - Enhanced
# ============================================================================

def extract_zomato_reviews_list(reviews_list_str):
    """Extract review texts from Zomato's reviews_list column with better parsing"""
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
                elif isinstance(item, str):
                    cleaned = clean_text(item)
                    if is_valid_review(cleaned):
                        reviews.append(cleaned)
        return reviews
    except:
        # Enhanced fallback parsing
        try:
            # Remove brackets and quotes
            cleaned = str(reviews_list_str).replace('[', '').replace(']', '')
            cleaned = re.sub(r'[\'"]+', '', cleaned)
            
            # Split by common delimiters
            parts = re.split(r'[(),]+', cleaned)
            
            for part in parts:
                text = clean_text(part)
                if is_valid_review(text):
                    reviews.append(text)
        except:
            pass
    
    return reviews


def load_mumbaires_reviews(filepath, restaurant_name=None):
    """Enhanced loader for mumbaires.csv with better error handling"""
    reviews = []
    try:
        df = pd.read_csv(filepath, encoding="utf-8", on_bad_lines="skip")
        df.columns = df.columns.str.strip()
        
        if restaurant_name:
            # Case-insensitive fuzzy matching
            mask = df['Restaurant Name'].astype(str).str.contains(
                restaurant_name, case=False, na=False, regex=False
            )
            df = df[mask]
        
        for _, row in df.iterrows():
            review_text = clean_text(row.get('Review Text', ''))
            if is_valid_review(review_text):
                quality_score = calculate_review_quality_score(review_text)
                
                reviews.append({
                    'review': review_text,
                    'restaurant': clean_text(row.get('Restaurant Name', '')),
                    'rating': row.get('Reviewer Rating'),
                    'source': 'mumbaires.csv',
                    'quality_score': quality_score,
                    'metadata': {
                        'address': clean_text(str(row.get('Address', ''))),
                        'station_area': clean_text(str(row.get('Station Area', '')))
                    }
                })
    except Exception as e:
        print(f"⚠️  Error loading mumbaires.csv: {e}")
    
    return reviews


def load_resreviews_reviews(filepath, restaurant_name=None):
    """Enhanced loader for Resreviews.csv"""
    reviews = []
    try:
        df = pd.read_csv(filepath, encoding="utf-8", on_bad_lines="skip")
        df.columns = df.columns.str.strip()
        
        if restaurant_name:
            mask = df['Restaurant'].astype(str).str.contains(
                restaurant_name, case=False, na=False, regex=False
            )
            df = df[mask]
        
        for _, row in df.iterrows():
            review_text = clean_text(row.get('Review', ''))
            if is_valid_review(review_text):
                quality_score = calculate_review_quality_score(review_text)
                
                reviews.append({
                    'review': review_text,
                    'restaurant': clean_text(row.get('Restaurant', '')),
                    'rating': row.get('Rating'),
                    'reviewer': clean_text(row.get('Reviewer', 'anonymous')),
                    'source': 'Resreviews.csv',
                    'quality_score': quality_score,
                    'metadata': {
                        'time': row.get('Time')
                    }
                })
    except Exception as e:
        print(f"⚠️  Error loading Resreviews.csv: {e}")
    
    return reviews


def load_reviews_reviews(filepath, restaurant_name=None):
    """Enhanced loader for reviews.csv"""
    reviews = []
    try:
        df = pd.read_csv(filepath, encoding="utf-8", on_bad_lines="skip")
        df.columns = df.columns.str.strip()
        
        if restaurant_name:
            mask = df['business_name'].astype(str).str.contains(
                restaurant_name, case=False, na=False, regex=False
            )
            df = df[mask]
        
        for _, row in df.iterrows():
            review_text = clean_text(row.get('text', ''))
            if is_valid_review(review_text):
                quality_score = calculate_review_quality_score(review_text)
                
                reviews.append({
                    'review': review_text,
                    'restaurant': clean_text(row.get('business_name', '')),
                    'rating': row.get('rating'),
                    'reviewer': clean_text(row.get('author_name', 'anonymous')),
                    'source': 'reviews.csv',
                    'quality_score': quality_score,
                    'metadata': {
                        'rating_category': row.get('rating_category')
                    }
                })
    except Exception as e:
        print(f"⚠️  Error loading reviews.csv: {e}")
    
    return reviews


def load_zomato_reviews(filepath, restaurant_name=None):
    """Enhanced loader for zomato.csv with better review extraction"""
    reviews = []
    try:
        df = pd.read_csv(filepath, encoding="utf-8", on_bad_lines="skip")
        df.columns = df.columns.str.strip()
        
        if restaurant_name:
            mask = df['name'].astype(str).str.contains(
                restaurant_name, case=False, na=False, regex=False
            )
            df = df[mask]
        
        for _, row in df.iterrows():
            restaurant = clean_text(row.get('name', ''))
            reviews_list = row.get('reviews_list', '')
            
            # Extract reviews from the tuple format
            extracted_reviews = extract_zomato_reviews_list(reviews_list)
            
            for review_text in extracted_reviews:
                quality_score = calculate_review_quality_score(review_text)
                
                reviews.append({
                    'review': review_text,
                    'restaurant': restaurant,
                    'rating': row.get('rate'),
                    'source': 'zomato.csv',
                    'quality_score': quality_score,
                    'metadata': {
                        'location': clean_text(str(row.get('location', ''))),
                        'cuisines': clean_text(str(row.get('cuisines', ''))),
                        'cost': row.get('approx_cost(for two people)'),
                        'online_order': row.get('online_order'),
                        'book_table': row.get('book_table')
                    }
                })
    except Exception as e:
        print(f"⚠️  Error loading zomato.csv: {e}")
    
    return reviews


def load_zomato2_reviews(filepath, restaurant_name=None):
    """Enhanced loader for zomato2.csv with better item aggregation"""
    reviews = []
    try:
        df = pd.read_csv(filepath, encoding="utf-8", on_bad_lines="skip")
        df.columns = df.columns.str.strip()
        
        if restaurant_name:
            mask = df['Restaurant_Name'].astype(str).str.contains(
                restaurant_name, case=False, na=False, regex=False
            )
            df = df[mask]
        
        # Group by restaurant for aggregated insights
        restaurant_groups = df.groupby('Restaurant_Name')
        
        for restaurant, group in restaurant_groups:
            # Separate best sellers and regular items
            best_sellers = []
            regular_items = []
            
            for _, row in group.iterrows():
                item = clean_text(row.get('Item_Name', ''))
                if item and item.lower() != 'nan':
                    if row.get('Best_Seller', False):
                        best_sellers.append(item)
                    else:
                        regular_items.append(item)
            
            # Create meaningful reviews from items
            if best_sellers:
                review_text = f"Highly recommended items: {', '.join(best_sellers[:5])}"
                if len(best_sellers) > 5:
                    review_text += f" and {len(best_sellers) - 5} more bestsellers"
                
                avg_rating = group['Average_Rating'].mean()
                reviews.append({
                    'review': review_text,
                    'restaurant': restaurant,
                    'rating': avg_rating,
                    'source': 'zomato2.csv',
                    'quality_score': 70,  # Synthetic but useful
                    'metadata': {
                        'cuisine': group['Cuisine'].iloc[0] if 'Cuisine' in group.columns else None,
                        'city': group['City'].iloc[0] if 'City' in group.columns else None,
                        'place': group['Place_Name'].iloc[0] if 'Place_Name' in group.columns else None,
                        'best_seller_count': len(best_sellers),
                        'total_items': len(best_sellers) + len(regular_items)
                    }
                })
            
            # Add a general menu review
            if regular_items:
                review_text = f"Menu includes: {', '.join(regular_items[:8])}"
                if len(regular_items) > 8:
                    review_text += f" and more"
                
                reviews.append({
                    'review': review_text,
                    'restaurant': restaurant,
                    'rating': group['Average_Rating'].mean(),
                    'source': 'zomato2.csv',
                    'quality_score': 60,
                    'metadata': {
                        'cuisine': group['Cuisine'].iloc[0] if 'Cuisine' in group.columns else None,
                        'item_count': len(regular_items)
                    }
                })
                
    except Exception as e:
        print(f"⚠️  Error loading zomato2.csv: {e}")
    
    return reviews


def load_local_reviews(restaurant_name=None, data_folder=DATASET_FOLDER, max_reviews=100):
    """
    Enhanced local review loader with quality scoring and deduplication
    """
    all_reviews = []
    
    if not os.path.exists(data_folder):
        print(f"❌ Dataset folder '{data_folder}' not found.")
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
                    print(f"✅ Loaded {len(reviews)} reviews from {filename}")
                    
            except Exception as e:
                print(f"❌ Error processing {filename}: {e}")
    
    # Advanced deduplication using text similarity
    unique_reviews = []
    seen_texts = set()
    
    for review in all_reviews:
        text = review['review']
        # Create a normalized version for comparison
        normalized = re.sub(r'\W+', '', text.lower())
        
        if normalized not in seen_texts:
            seen_texts.add(normalized)
            unique_reviews.append(review)
    
    # Sort by quality score (if available)
    unique_reviews.sort(key=lambda x: x.get('quality_score', 0), reverse=True)
    
    return unique_reviews[:max_reviews]


# ============================================================================
# WEB SCRAPING FUNCTIONS (Priority 2 - Enhanced)
# ============================================================================

def scrape_single_url(url, restaurant_name, timeout=10):
    """
    Scrape a single URL for reviews - used for parallel scraping
    """
    results = []
    
    try:
        smart_delay(0.3, 0.8)
        response = requests.get(url, headers=get_headers(), timeout=timeout)
        
        if response.status_code != 200:
            return results
        
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Look for review-like content in multiple tag types
        potential_reviews = []
        
        # Paragraphs
        for p in soup.find_all("p"):
            text = clean_text(p.get_text())
            if is_valid_review(text, min_length=50):
                potential_reviews.append(text)
        
        # Divs with review-like classes
        review_divs = soup.find_all("div", class_=re.compile(r'review|comment|feedback|testimonial', re.I))
        for div in review_divs:
            text = clean_text(div.get_text())
            if is_valid_review(text, min_length=50):
                potential_reviews.append(text)
        
        # Filter and score reviews
        for text in potential_reviews:
            if any(word in text.lower() for word in 
                  ['food', 'dish', 'menu', 'service', 'taste', 'delicious', 
                   'restaurant', 'meal', 'ordered', 'ambience', 'staff']):
                
                quality_score = calculate_review_quality_score(text)
                
                if quality_score >= 50:  # Only keep decent quality reviews
                    results.append({
                        "source": urlparse(url).netloc,
                        "review": text,
                        "restaurant": restaurant_name,
                        "quality_score": quality_score,
                        "url": url
                    })
        
    except Exception as e:
        print(f"  ⚠️  Error scraping {urlparse(url).netloc}: {str(e)[:50]}")
    
    return results


def scrape_generic_reviews(restaurant_name, max_reviews=20):
    """
    Enhanced generic review scraper with parallel processing
    """
    results = []
    seen = set()
    
    try:
        query = f"{restaurant_name} restaurant customer reviews Mumbai"
        url = "https://html.duckduckgo.com/html/"
        
        print(f"🌐 Web scraping: Searching for '{restaurant_name}'...")
        
        response = requests.post(url, data={"q": query}, headers=get_headers(), timeout=10)
        if response.status_code != 200:
            print(f"⚠️  Web search failed with status {response.status_code}")
            return []

        soup = BeautifulSoup(response.text, "html.parser")
        links = [a.get("href") for a in soup.find_all("a", href=True)]
        
        # Filter for quality review sources
        review_links = [
            l for l in links 
            if any(x in l.lower() for x in ["review", "tripadvisor", "zomato", "yelp", 
                                           "dineout", "blog", "food", "restaurant"])
            and not any(x in l.lower() for x in ["facebook", "twitter", "instagram", "ad"])
        ][:12]
        
        print(f"📍 Found {len(review_links)} potential sources")

        # Parallel scraping for speed
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_url = {
                executor.submit(scrape_single_url, link, restaurant_name): link 
                for link in review_links
            }
            
            for i, future in enumerate(as_completed(future_to_url), 1):
                try:
                    url_results = future.result()
                    
                    for review_dict in url_results:
                        text = review_dict['review']
                        normalized = re.sub(r'\W+', '', text.lower())
                        
                        if normalized not in seen:
                            seen.add(normalized)
                            results.append(review_dict)
                            
                            if len(results) >= max_reviews:
                                break
                    
                    print(f"  ✓ Processed {i}/{len(review_links)} sources ({len(results)} reviews found)")
                    
                except Exception as e:
                    print(f"  ⚠️  Error: {str(e)[:50]}")
                
                if len(results) >= max_reviews:
                    break
                    
    except Exception as e:
        print(f"❌ Generic scraping error: {e}")
        return []
    
    # Sort by quality
    results.sort(key=lambda x: x.get('quality_score', 0), reverse=True)
    
    print(f"✅ Scraped {len(results)} quality reviews from web")
    return results[:max_reviews]


def scrape_zomato_placeholder(restaurant_name, max_reviews=10):
    """
    Enhanced Zomato placeholder scraper (limited due to dynamic content)
    """
    results = []
    seen = set()
    
    try:
        print(f"🔍 Attempting Zomato search for '{restaurant_name}'...")
        
        base = "https://www.zomato.com"
        search_url = base + "/mumbai/search?q=" + quote_plus(restaurant_name)
        
        response = requests.get(search_url, headers=get_headers(), timeout=10)
        
        if response.status_code != 200:
            print(f"⚠️  Zomato search returned status {response.status_code}")
            return results

        soup = BeautifulSoup(response.text, "html.parser")
        
        # Look for text blocks that might contain reviews
        possible_blocks = soup.find_all(["p", "div", "span"], text=True)
        
        for tag in possible_blocks:
            if len(results) >= max_reviews:
                break
                
            text = clean_text(tag.get_text())
            
            if not is_valid_review(text, min_length=50):
                continue
            
            # Check for restaurant/food-related keywords
            if re.search(r"\b(food|service|taste|ambience|price|delivery|dish|menu|recommend)\b", text, re.I):
                normalized = re.sub(r'\W+', '', text.lower())
                
                if normalized not in seen:
                    seen.add(normalized)
                    quality_score = calculate_review_quality_score(text)
                    
                    results.append({
                        "source": "zomato.com",
                        "review": text,
                        "restaurant": restaurant_name,
                        "quality_score": quality_score
                    })
        
        if results:
            print(f"✅ Found {len(results)} potential reviews on Zomato")
        else:
            print("⚠️  No reviews extracted from Zomato (likely dynamic content)")
        
    except Exception as e:
        print(f"⚠️  Zomato scraping error: {e}")
    
    return results


# ============================================================================
# UNIFIED FUNCTION (Enhanced with Better Reporting)
# ============================================================================

def scrape_reviews_combined(restaurant_name, data_folder=DATASET_FOLDER, 
                           max_reviews=50, enable_web_scraping=False,
                           quality_threshold=40):
    """
    Enhanced unified review collection with quality filtering
    
    Args:
        restaurant_name: Name of restaurant to search for
        data_folder: Path to datasets folder
        max_reviews: Maximum number of reviews to collect
        enable_web_scraping: Whether to enable web scraping fallback
        quality_threshold: Minimum quality score (0-100) for reviews
    
    Returns:
        List of high-quality review dictionaries
    """
    print(f"\n{'='*70}")
    print(f"🔍 RESTAURANT REVIEW COLLECTOR")
    print(f"{'='*70}")
    print(f"Restaurant: {restaurant_name}")
    print(f"Max Reviews: {max_reviews}")
    print(f"Quality Threshold: {quality_threshold}")
    print(f"Web Scraping: {'Enabled' if enable_web_scraping else 'Disabled'}")
    print(f"{'='*70}\n")
    
    all_reviews = []
    
    # PRIORITY 1: Load from local datasets
    print("📂 STEP 1: Searching local datasets...")
    local_reviews = load_local_reviews(restaurant_name, data_folder, max_reviews)
    
    if local_reviews:
        # Filter by quality
        quality_reviews = [r for r in local_reviews if r.get('quality_score', 0) >= quality_threshold]
        all_reviews.extend(quality_reviews)
        print(f"✅ Found {len(quality_reviews)} quality reviews in local datasets")
        print(f"   (Filtered from {len(local_reviews)} total reviews)\n")
    else:
        print("⚠️  No reviews found in local datasets\n")
    
    # PRIORITY 2: Web scraping fallback
    if enable_web_scraping and len(all_reviews) < max_reviews:
        remaining = max_reviews - len(all_reviews)
        print(f"🌐 STEP 2: Web scraping fallback (need {remaining} more reviews)...")
        
        # Try generic sources first (usually better quality)
        generic_reviews = scrape_generic_reviews(restaurant_name, max_reviews=remaining)
        quality_web_reviews = [r for r in generic_reviews if r.get('quality_score', 0) >= quality_threshold]
        all_reviews.extend(quality_web_reviews)
        
        # Try Zomato if still need more
        if len(all_reviews) < max_reviews:
            remaining = max_reviews - len(all_reviews)
            zomato_reviews = scrape_zomato_placeholder(restaurant_name, max_reviews=remaining)
            quality_zomato = [r for r in zomato_reviews if r.get('quality_score', 0) >= quality_threshold]
            all_reviews.extend(quality_zomato)
        
        web_count = len(all_reviews) - len([r for r in local_reviews if r.get('quality_score', 0) >= quality_threshold])
        if web_count > 0:
            print(f"✅ Scraped {web_count} quality reviews from web\n")
        else:
            print("⚠️  No additional quality reviews found on web\n")
    
    # Calculate statistics
    print(f"{'='*70}")
    print(f"📊 COLLECTION SUMMARY")
    print(f"{'='*70}")
    print(f"Total Reviews Collected: {len(all_reviews)}")
    
    if all_reviews:
        avg_quality = sum(r.get('quality_score', 0) for r in all_reviews) / len(all_reviews)
        print(f"Average Quality Score: {avg_quality:.1f}/100")
        
        # Show source breakdown
        source_counts = {}
        for review in all_reviews:
            source = review.get('source', 'unknown')
            source_counts[source] = source_counts.get(source, 0) + 1
        
        print(f"\nSource Breakdown:")
        for source, count in sorted(source_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(all_reviews)) * 100
            print(f"  • {source}: {count} reviews ({percentage:.1f}%)")
    
    print(f"{'='*70}\n")
    
    return all_reviews[:max_reviews]


# ============================================================================
# UTILITY FUNCTIONS (Enhanced)
# ============================================================================

def get_all_restaurants_from_datasets(data_folder=DATASET_FOLDER):
    """Get list of all unique restaurants with metadata"""
    restaurants = []
    seen_names = set()
    
    if not os.path.exists(data_folder):
        return []
    
    dataset_configs = {
        "mumbaires.csv": ("Restaurant Name", ["Rating", "Address"]),
        "Resreviews.csv": ("Restaurant", ["Rating"]),
        "reviews.csv": ("business_name", ["rating"]),
        "zomato.csv": ("name", ["rate", "location", "cuisines"]),
        "zomato2.csv": ("Restaurant_Name", ["Avg_Rating_Restaurant", "City"]),
    }
    
    for filename, (name_col, metadata_cols) in dataset_configs.items():
        filepath = os.path.join(data_folder, filename)
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath, encoding="utf-8", on_bad_lines="skip")
                df.columns = df.columns.str.strip()
                
                if name_col in df.columns:
                    # Get unique restaurants with metadata
                    for _, row in df.iterrows():
                        name = clean_text(str(row.get(name_col, '')))
                        
                        if name and name.lower() != 'nan' and name not in seen_names:
                            seen_names.add(name)
                            
                            restaurant_info = {
                                'name': name,
                                'source': filename
                            }
                            
                            # Add available metadata
                            for meta_col in metadata_cols:
                                if meta_col in df.columns:
                                    value = row.get(meta_col)
                                    if pd.notna(value):
                                        restaurant_info[meta_col.lower()] = value
                            
                            restaurants.append(restaurant_info)
                            
            except Exception as e:
                print(f"⚠️  Error reading {filename}: {e}")
    
    return sorted(restaurants, key=lambda x: x['name'])


def search_restaurants(query, data_folder=DATASET_FOLDER, max_results=10):
    """
    Enhanced restaurant search with fuzzy matching and ranking
    """
    all_restaurants = get_all_restaurants_from_datasets(data_folder)
    query_lower = query.lower()
    
    # Score each restaurant by relevance
    scored_matches = []
    
    for restaurant in all_restaurants:
        name = restaurant['name']
        name_lower = name.lower()
        
        # Calculate relevance score
        score = 0
        
        # Exact match (highest score)
        if query_lower == name_lower:
            score = 100
        # Starts with query
        elif name_lower.startswith(query_lower):
            score = 80
        # Contains query
        elif query_lower in name_lower:
            score = 60
        # Word-level match
        else:
            query_words = set(query_lower.split())
            name_words = set(name_lower.split())
            matching_words = query_words.intersection(name_words)
            
            if matching_words:
                score = (len(matching_words) / len(query_words)) * 50
        
        if score > 0:
            scored_matches.append((restaurant, score))
    
    # Sort by score (descending)
    scored_matches.sort(key=lambda x: x[1], reverse=True)
    
    # Return top matches
    return [match[0] for match in scored_matches[:max_results]]


def get_review_statistics(reviews):
    """
    Generate comprehensive statistics from a list of reviews
    """
    if not reviews:
        return {"error": "No reviews provided"}
    
    stats = {
        "total_reviews": len(reviews),
        "sources": {},
        "quality_metrics": {},
        "ratings": {},
        "content_analysis": {}
    }
    
    # Source distribution
    for review in reviews:
        source = review.get('source', 'unknown')
        stats["sources"][source] = stats["sources"].get(source, 0) + 1
    
    # Quality metrics
    quality_scores = [r.get('quality_score', 0) for r in reviews if 'quality_score' in r]
    if quality_scores:
        stats["quality_metrics"] = {
            "average": sum(quality_scores) / len(quality_scores),
            "min": min(quality_scores),
            "max": max(quality_scores),
            "high_quality_count": sum(1 for s in quality_scores if s >= 70)
        }
    
    # Rating analysis
    ratings = [r.get('rating') for r in reviews if r.get('rating') is not None]
    if ratings:
        # Convert to float and filter valid ratings
        valid_ratings = []
        for r in ratings:
            try:
                rating_val = float(str(r).split('/')[0])  # Handle "4/5" format
                if 0 <= rating_val <= 5:
                    valid_ratings.append(rating_val)
            except:
                pass
        
        if valid_ratings:
            stats["ratings"] = {
                "average": sum(valid_ratings) / len(valid_ratings),
                "count": len(valid_ratings),
                "min": min(valid_ratings),
                "max": max(valid_ratings)
            }
    
    # Content analysis
    all_text = " ".join([r.get('review', '') for r in reviews])
    words = all_text.lower().split()
    
    stats["content_analysis"] = {
        "total_words": len(words),
        "unique_words": len(set(words)),
        "avg_review_length": len(all_text) / len(reviews) if reviews else 0
    }
    
    return stats


def export_reviews_to_csv(reviews, output_path="exported_reviews.csv"):
    """
    Export reviews to CSV format for further analysis
    """
    if not reviews:
        print("❌ No reviews to export")
        return False
    
    try:
        # Flatten the data structure
        flattened_reviews = []
        
        for review in reviews:
            flat_review = {
                'restaurant': review.get('restaurant', ''),
                'review_text': review.get('review', ''),
                'rating': review.get('rating', ''),
                'source': review.get('source', ''),
                'quality_score': review.get('quality_score', ''),
                'reviewer': review.get('reviewer', 'anonymous')
            }
            
            # Add metadata if exists
            if 'metadata' in review and review['metadata']:
                for key, value in review['metadata'].items():
                    flat_review[f'meta_{key}'] = value
            
            flattened_reviews.append(flat_review)
        
        # Create DataFrame and export
        df = pd.DataFrame(flattened_reviews)
        df.to_csv(output_path, index=False, encoding='utf-8')
        
        print(f"✅ Exported {len(reviews)} reviews to {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error exporting reviews: {e}")
        return False


def deduplicate_reviews(reviews, similarity_threshold=0.85):
    """
    Advanced deduplication using text similarity
    Removes near-duplicate reviews
    """
    if len(reviews) <= 1:
        return reviews
    
    unique_reviews = []
    seen_normalized = set()
    
    for review in reviews:
        text = review.get('review', '')
        
        # Create normalized version for comparison
        normalized = re.sub(r'\W+', '', text.lower())
        
        # Check for exact duplicates
        if normalized in seen_normalized:
            continue
        
        # Check for near-duplicates (simple approach)
        is_duplicate = False
        for seen_text in seen_normalized:
            # Calculate simple similarity (character overlap)
            overlap = len(set(normalized) & set(seen_text))
            total_chars = len(set(normalized) | set(seen_text))
            
            if total_chars > 0:
                similarity = overlap / total_chars
                if similarity >= similarity_threshold:
                    is_duplicate = True
                    break
        
        if not is_duplicate:
            seen_normalized.add(normalized)
            unique_reviews.append(review)
    
    removed_count = len(reviews) - len(unique_reviews)
    if removed_count > 0:
        print(f"🔄 Removed {removed_count} duplicate/similar reviews")
    
    return unique_reviews


def filter_reviews_by_keywords(reviews, keywords, mode='include'):
    """
    Filter reviews based on keywords
    
    Args:
        reviews: List of review dictionaries
        keywords: List of keywords to filter by
        mode: 'include' (keep matching) or 'exclude' (remove matching)
    
    Returns:
        Filtered list of reviews
    """
    filtered = []
    
    for review in reviews:
        text = review.get('review', '').lower()
        has_keyword = any(keyword.lower() in text for keyword in keywords)
        
        if mode == 'include' and has_keyword:
            filtered.append(review)
        elif mode == 'exclude' and not has_keyword:
            filtered.append(review)
    
    return filtered


def get_sentiment_keywords():
    """
    Return comprehensive sentiment keyword dictionaries
    """
    return {
        'positive': [
            'excellent', 'amazing', 'delicious', 'fantastic', 'wonderful',
            'great', 'best', 'perfect', 'awesome', 'outstanding', 'superb',
            'love', 'loved', 'incredible', 'exceptional', 'brilliant',
            'tasty', 'fresh', 'flavorful', 'recommend', 'recommended'
        ],
        'negative': [
            'terrible', 'horrible', 'awful', 'disgusting', 'worst',
            'bad', 'poor', 'disappointing', 'waste', 'pathetic',
            'avoid', 'never', 'rude', 'slow', 'cold', 'burnt',
            'overpriced', 'stale', 'dirty', 'unhygienic', 'bland'
        ],
        'neutral': [
            'okay', 'average', 'decent', 'fine', 'normal',
            'standard', 'typical', 'ordinary', 'acceptable'
        ]
    }


# ============================================================================
# BATCH PROCESSING UTILITIES
# ============================================================================

def batch_load_reviews(restaurant_names, data_folder=DATASET_FOLDER, 
                      max_reviews_per_restaurant=50):
    """
    Load reviews for multiple restaurants in batch
    """
    print(f"\n🔄 Batch loading reviews for {len(restaurant_names)} restaurants...")
    
    all_results = {}
    
    for i, restaurant in enumerate(restaurant_names, 1):
        print(f"\n[{i}/{len(restaurant_names)}] Processing: {restaurant}")
        
        reviews = load_local_reviews(restaurant, data_folder, max_reviews_per_restaurant)
        all_results[restaurant] = reviews
        
        print(f"  ✅ Loaded {len(reviews)} reviews")
    
    total_reviews = sum(len(reviews) for reviews in all_results.values())
    print(f"\n✅ Batch complete: {total_reviews} total reviews loaded")
    
    return all_results


def compare_restaurants(restaurant_names, data_folder=DATASET_FOLDER):
    """
    Compare multiple restaurants based on their reviews
    """
    print(f"\n📊 Comparing {len(restaurant_names)} restaurants...\n")
    
    comparison = {}
    
    for restaurant in restaurant_names:
        reviews = load_local_reviews(restaurant, data_folder, max_reviews=100)
        
        if reviews:
            stats = get_review_statistics(reviews)
            comparison[restaurant] = stats
        else:
            comparison[restaurant] = {"error": "No reviews found"}
    
    return comparison


# ============================================================================
# MAIN EXECUTION EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Example usage
    print("🍽️  Restaurant Review Scraper - Enhanced Edition\n")
    
    # Test with a restaurant
    test_restaurant = "Leopold Cafe"
    
    # Load reviews
    reviews = scrape_reviews_combined(
        restaurant_name=test_restaurant,
        max_reviews=30,
        enable_web_scraping=False,  # Set to True to enable web scraping
        quality_threshold=50
    )
    
    # Show statistics
    if reviews:
        print("\n📈 Review Statistics:")
        stats = get_review_statistics(reviews)
        print(f"  Total Reviews: {stats['total_reviews']}")
        
        if 'quality_metrics' in stats:
            print(f"  Average Quality: {stats['quality_metrics']['average']:.1f}/100")
        
        if 'ratings' in stats:
            print(f"  Average Rating: {stats['ratings']['average']:.2f}/5")
        
        # Export to CSV
        export_reviews_to_csv(reviews, f"{test_restaurant.replace(' ', '_')}_reviews.csv")
    else:
        print("❌ No reviews found")