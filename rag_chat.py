import os
import ast
import pandas as pd
import numpy as np
import textwrap
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from collections import Counter
import re

class RAGChat:
    def __init__(self, data_folder="datasets"):
        self.vectorizer = None
        self.doc_texts = []
        self.doc_metadata = []  # Store metadata about each document
        self.tfidf = None
        self.data_folder = data_folder
        self.loaded = False
        self.current_restaurant = None

    def extract_reviews_from_zomato_list(self, reviews_list_str):
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

    def load_mumbaires_csv(self, filepath, restaurant_name=None):
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
                review_text = str(row.get('Review Text', '')).strip()
                restaurant = str(row.get('Restaurant Name', '')).strip()
                
                if review_text and review_text.lower() != 'nan' and len(review_text) > 20:
                    reviews.append({
                        'text': review_text,
                        'restaurant': restaurant,
                        'rating': row.get('Reviewer Rating'),
                        'source': 'mumbaires.csv'
                    })
        except Exception as e:
            print(f"Error loading mumbaires.csv: {e}")
        
        return reviews

    def load_resreviews_csv(self, filepath, restaurant_name=None):
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
                review_text = str(row.get('Review', '')).strip()
                restaurant = str(row.get('Restaurant', '')).strip()
                
                if review_text and review_text.lower() != 'nan' and len(review_text) > 20:
                    reviews.append({
                        'text': review_text,
                        'restaurant': restaurant,
                        'rating': row.get('Rating'),
                        'reviewer': row.get('Reviewer', 'anonymous'),
                        'source': 'Resreviews.csv'
                    })
        except Exception as e:
            print(f"Error loading Resreviews.csv: {e}")
        
        return reviews

    def load_reviews_csv(self, filepath, restaurant_name=None):
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
                review_text = str(row.get('text', '')).strip()
                restaurant = str(row.get('business_name', '')).strip()
                
                if review_text and review_text.lower() != 'nan' and len(review_text) > 20:
                    reviews.append({
                        'text': review_text,
                        'restaurant': restaurant,
                        'rating': row.get('rating'),
                        'reviewer': row.get('author_name', 'anonymous'),
                        'source': 'reviews.csv'
                    })
        except Exception as e:
            print(f"Error loading reviews.csv: {e}")
        
        return reviews

    def load_zomato_csv(self, filepath, restaurant_name=None):
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
                restaurant = str(row.get('name', '')).strip()
                reviews_list = row.get('reviews_list', '')
                
                # Extract reviews from the reviews_list tuple format
                extracted_reviews = self.extract_reviews_from_zomato_list(reviews_list)
                
                for review_text in extracted_reviews:
                    reviews.append({
                        'text': review_text,
                        'restaurant': restaurant,
                        'rating': row.get('rate'),
                        'location': row.get('location'),
                        'cuisines': row.get('cuisines'),
                        'source': 'zomato.csv'
                    })
        except Exception as e:
            print(f"Error loading zomato.csv: {e}")
        
        return reviews

    def load_zomato2_csv(self, filepath, restaurant_name=None):
        """Load reviews from zomato2.csv (creates synthetic reviews from items)"""
        reviews = []
        try:
            df = pd.read_csv(filepath, encoding="utf-8", on_bad_lines="skip")
            df.columns = df.columns.str.strip()
            
            if restaurant_name:
                df = df[df['Restaurant_Name'].astype(str).str.contains(
                    restaurant_name, case=False, na=False, regex=False
                )]
            
            # Group by restaurant
            restaurant_groups = df.groupby('Restaurant_Name')
            
            for restaurant, group in restaurant_groups:
                for _, row in group.iterrows():
                    item = str(row.get('Item_Name', '')).strip()
                    if item and item.lower() != 'nan':
                        # Create contextual review from item data
                        review_text = f"Tried {item}"
                        if row.get('Best_Seller', False):
                            review_text += " - This is a best seller"
                        if row.get('Votes', 0) > 0:
                            review_text += f" with {row.get('Votes')} votes"
                        
                        reviews.append({
                            'text': review_text,
                            'restaurant': restaurant,
                            'rating': row.get('Average_Rating'),
                            'cuisine': row.get('Cuisine'),
                            'city': row.get('City'),
                            'source': 'zomato2.csv'
                        })
        except Exception as e:
            print(f"Error loading zomato2.csv: {e}")
        
        return reviews

    def load_csv_data(self, restaurant_name=None):
        """
        Loads all review-related CSV files from the datasets folder.
        If restaurant_name is provided, filters reviews belonging to that restaurant.
        """
        all_reviews = []
        
        if not os.path.exists(self.data_folder):
            print(f"Dataset folder '{self.data_folder}' not found.")
            return []

        dataset_loaders = {
            "mumbaires.csv": self.load_mumbaires_csv,
            "Resreviews.csv": self.load_resreviews_csv,
            "reviews.csv": self.load_reviews_csv,
            "zomato.csv": self.load_zomato_csv,
            "zomato2.csv": self.load_zomato2_csv,
        }
        
        for filename, loader_func in dataset_loaders.items():
            filepath = os.path.join(self.data_folder, filename)
            if os.path.exists(filepath):
                try:
                    reviews = loader_func(filepath, restaurant_name)
                    all_reviews.extend(reviews)
                    if restaurant_name and reviews:
                        print(f"✓ Loaded {len(reviews)} reviews from {filename}")
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
        
        self.loaded = True
        self.current_restaurant = restaurant_name
        return all_reviews

    def index_documents(self, texts, metadata=None):
        """
        Index documents using TF-IDF.
        texts: list of review text strings
        metadata: optional list of metadata dicts for each text
        """
        # Filter valid texts
        valid_indices = [i for i, t in enumerate(texts) if t and len(str(t)) > 20]
        self.doc_texts = [texts[i] for i in valid_indices]
        
        # Store metadata if provided
        if metadata:
            self.doc_metadata = [metadata[i] for i in valid_indices]
        else:
            self.doc_metadata = [{}] * len(self.doc_texts)
        
        if not self.doc_texts:
            self.tfidf = None
            print("No valid documents to index.")
            return
        
        # Create TF-IDF vectors
        self.vectorizer = TfidfVectorizer(
            stop_words='english', 
            max_features=5000,
            ngram_range=(1, 2),  # Include bigrams for better context
            min_df=1,
            max_df=0.95
        )
        self.tfidf = self.vectorizer.fit_transform(self.doc_texts)
        print(f"✓ Indexed {len(self.doc_texts)} documents")

    def answer_query(self, query, restaurant_name=None, top_k=3):
        """
        Answers a query using indexed documents.
        Falls back to loading CSV data if not already loaded.
        """
        # Load data if not already loaded
        if not self.loaded or (restaurant_name and restaurant_name != self.current_restaurant):
            print(f"Loading reviews for '{restaurant_name}'...")
            reviews = self.load_csv_data(restaurant_name)
            
            if reviews:
                texts = [r['text'] for r in reviews]
                self.index_documents(texts, metadata=reviews)
            else:
                return "No reviews found for this restaurant in the datasets.", []

        # Check if we have indexed documents
        if self.tfidf is None or self.vectorizer is None:
            return "No indexed reviews available. Please analyze the restaurant first.", []

        # Transform query and compute similarities
        try:
            qv = self.vectorizer.transform([query])
            sims = linear_kernel(qv, self.tfidf).flatten()
            
            # Get top-k most similar documents
            top_indices = np.argsort(sims)[::-1][:top_k * 2]  # Get more initially
            top_texts = []
            top_meta = []
            
            for idx in top_indices:
                if sims[idx] > 0.05:  # Minimum similarity threshold
                    top_texts.append(self.doc_texts[idx])
                    top_meta.append(self.doc_metadata[idx])
                if len(top_texts) >= top_k:
                    break
            
            # If no good matches found, try web fallback
            if not top_texts:
                web_answer = self._search_google_fallback(query, restaurant_name)
                return web_answer, []
            
            # Format the answer with context
            answer = self._format_answer(query, top_texts, top_meta, restaurant_name)
            return answer, top_texts
            
        except Exception as e:
            print(f"Error during query: {e}")
            return f"Error processing your question: {str(e)}", []

    def _format_answer(self, query, top_texts, top_meta, restaurant_name):
        """Format a comprehensive answer from retrieved documents"""
        # Analyze query intent
        query_lower = query.lower()
        
        # Detect query type
        is_quality_query = any(word in query_lower for word in ['quality', 'taste', 'food', 'delicious'])
        is_service_query = any(word in query_lower for word in ['service', 'staff', 'waiter', 'wait'])
        is_price_query = any(word in query_lower for word in ['price', 'cost', 'expensive', 'cheap'])
        is_recommend_query = any(word in query_lower for word in ['recommend', 'suggest', 'best', 'good'])
        
        # Build answer header
        restaurant_phrase = f"for {restaurant_name}" if restaurant_name else "from the reviews"
        answer = f"Based on customer reviews {restaurant_phrase}:\n\n"
        
        # Add relevant snippets
        answer += "**Relevant Reviews:**\n"
        for i, (text, meta) in enumerate(zip(top_texts, top_meta), 1):
            snippet = textwrap.shorten(text, width=200, placeholder="...")
            rating = meta.get('rating', 'N/A')
            source = meta.get('source', 'unknown')
            
            answer += f"\n{i}. {snippet}"
            if rating and str(rating).lower() != 'nan':
                answer += f" [Rating: {rating}]"
            answer += "\n"
        
        # Add synthesis based on query type
        answer += "\n**Summary:**\n"
        summary_points = self._synthesize_answer(query, top_texts, is_quality_query, 
                                                 is_service_query, is_price_query, 
                                                 is_recommend_query)
        answer += summary_points
        
        return answer

    def _synthesize_answer(self, query, texts, is_quality, is_service, is_price, is_recommend):
        """Synthesize an answer based on query type and retrieved texts"""
        # Extract key information
        all_text = " ".join(texts).lower()
        
        # Common sentiment words
        positive_words = ['good', 'great', 'excellent', 'amazing', 'delicious', 'perfect', 
                         'wonderful', 'fantastic', 'love', 'best', 'awesome']
        negative_words = ['bad', 'poor', 'terrible', 'horrible', 'awful', 'worst', 
                         'disappointing', 'waste', 'avoid', 'never']
        
        pos_count = sum(all_text.count(word) for word in positive_words)
        neg_count = sum(all_text.count(word) for word in negative_words)
        
        # Extract frequent meaningful terms
        key_terms = self._extract_key_terms(texts)
        
        # Build synthesized answer
        synthesis = ""
        
        if is_quality:
            if pos_count > neg_count:
                synthesis = f"The food quality is generally praised. "
            else:
                synthesis = f"There are mixed reviews about food quality. "
            synthesis += f"Commonly mentioned aspects: {', '.join(key_terms[:5])}."
        
        elif is_service:
            if 'slow' in all_text or 'wait' in all_text:
                synthesis = "Some customers mention slow service or waiting times. "
            if pos_count > neg_count:
                synthesis += "However, staff attitude is generally appreciated."
            else:
                synthesis += "Service quality could be improved according to reviews."
        
        elif is_price:
            if 'expensive' in all_text or 'overpriced' in all_text:
                synthesis = "Customers find the prices on the higher side. "
            else:
                synthesis = "The pricing is considered reasonable by most customers. "
            if 'value' in all_text:
                synthesis += "Value for money is a topic of discussion."
        
        elif is_recommend:
            if pos_count > neg_count * 1.5:
                synthesis = f"✅ Highly recommended! "
            elif pos_count > neg_count:
                synthesis = f"Generally recommended with some caveats. "
            else:
                synthesis = f"Mixed reviews - read carefully before visiting. "
            synthesis += f"Popular mentions: {', '.join(key_terms[:4])}."
        
        else:
            # General query
            if pos_count > neg_count:
                synthesis = f"Overall positive feedback. "
            else:
                synthesis = f"Reviews show some concerns. "
            synthesis += f"Key points mentioned: {', '.join(key_terms[:6])}."
        
        return synthesis

    def _extract_key_terms(self, texts):
        """Extract the most important terms from texts"""
        words = []
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                    'of', 'with', 'by', 'from', 'is', 'was', 'are', 'were', 'been', 'be',
                    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                    'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'}
        
        for text in texts:
            tokens = re.findall(r"\b[a-z]{4,}\b", text.lower())
            words.extend([t for t in tokens if t not in stopwords])
        
        # Count and return most common
        counter = Counter(words)
        return [word for word, _ in counter.most_common(10)]

    def _search_google_fallback(self, query, restaurant_name=None):
        """
        Lightweight web fallback when local data doesn't match.
        Can be replaced with proper API (SerpAPI, Tavily, etc.)
        """
        try:
            search_query = f"{restaurant_name} {query}" if restaurant_name else query
            url = f"https://www.google.com/search?q={search_query.replace(' ', '+')}"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            
            response = requests.get(url, headers=headers, timeout=5)
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Extract search result snippets
            snippets = []
            for div in soup.select("div.BNeawe.s3v9rd.AP7Wnd")[:5]:
                text = div.get_text().strip()
                if text and len(text) > 30:
                    snippets.append(text)
            
            if not snippets:
                return (f"I couldn't find relevant information in the reviews or online "
                       f"about '{query}' for {restaurant_name or 'this restaurant'}.")
            
            result = "I couldn't find this in the reviews, but here's what I found online:\n\n"
            for i, snippet in enumerate(snippets, 1):
                shortened = textwrap.shorten(snippet, width=200, placeholder="...")
                result += f"{i}. {shortened}\n\n"
            
            return result
            
        except Exception as e:
            return (f"I couldn't find information in the local reviews. "
                   f"Error searching online: {str(e)}")

    def get_statistics(self):
        """Get statistics about indexed documents"""
        if not self.doc_texts:
            return "No documents indexed."
        
        stats = {
            'total_reviews': len(self.doc_texts),
            'restaurant': self.current_restaurant or 'All',
            'sources': {}
        }
        
        for meta in self.doc_metadata:
            source = meta.get('source', 'unknown')
            stats['sources'][source] = stats['sources'].get(source, 0) + 1
        
        return stats