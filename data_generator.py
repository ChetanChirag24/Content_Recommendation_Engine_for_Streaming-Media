"""
Data Generator for Content Recommendation System
Generates synthetic user-content interaction data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import argparse
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InteractionDataGenerator:
    """Generate synthetic streaming interaction data"""
    
    def __init__(self, n_users=10000, n_content=50000, n_interactions=1000000, random_state=42):
        self.n_users = n_users
        self.n_content = n_content
        self.n_interactions = n_interactions
        self.random_state = random_state
        np.random.seed(random_state)
        
        # Content metadata
        self.genres = ['Action', 'Comedy', 'Drama', 'Thriller', 'Sci-Fi', 
                       'Romance', 'Horror', 'Documentary', 'Animation', 'Musical']
        self.devices = ['mobile', 'desktop', 'tablet', 'smart_tv', 'console']
        
    def generate_user_profiles(self):
        """Generate user profiles with preferences"""
        logger.info(f"Generating {self.n_users} user profiles...")
        
        users = []
        for i in range(self.n_users):
            # User demographics
            age = np.random.choice([18, 25, 35, 45, 55, 65], p=[0.15, 0.25, 0.25, 0.20, 0.10, 0.05])
            
            # Genre preferences (each user has 2-3 preferred genres)
            n_pref_genres = np.random.randint(2, 4)
            pref_genres = np.random.choice(self.genres, size=n_pref_genres, replace=False)
            
            # Activity level
            activity_level = np.random.choice(['low', 'medium', 'high'], p=[0.3, 0.5, 0.2])
            
            # Preferred device
            pref_device = np.random.choice(self.devices, p=[0.35, 0.25, 0.15, 0.20, 0.05])
            
            # Engagement score (latent quality)
            engagement_score = np.random.beta(5, 2)
            
            users.append({
                'user_id': i,
                'age_group': age,
                'preferred_genres': ','.join(pref_genres),
                'activity_level': activity_level,
                'preferred_device': pref_device,
                'engagement_score': engagement_score
            })
        
        return pd.DataFrame(users)
    
    def generate_content_catalog(self):
        """Generate content catalog"""
        logger.info(f"Generating {self.n_content} content items...")
        
        content = []
        for i in range(self.n_content):
            # Content metadata
            content_type = np.random.choice(['movie', 'series', 'documentary', 'short'], 
                                          p=[0.40, 0.35, 0.15, 0.10])
            
            genre = np.random.choice(self.genres)
            
            # Duration in minutes
            if content_type == 'movie':
                duration = np.random.choice([90, 100, 110, 120, 130, 140, 150])
            elif content_type == 'series':
                duration = np.random.choice([40, 45, 50, 55, 60])
            elif content_type == 'documentary':
                duration = np.random.choice([45, 60, 90, 120])
            else:  # short
                duration = np.random.choice([5, 10, 15, 20])
            
            # Quality score (affects engagement)
            quality_score = np.random.beta(6, 2)
            
            # Release year
            release_year = np.random.choice(range(2015, 2025), p=[0.05, 0.05, 0.08, 0.10, 0.12, 
                                                                    0.12, 0.15, 0.15, 0.15, 0.03])
            
            # Popularity score
            popularity = np.random.beta(3, 5)  # Most content is not super popular
            
            content.append({
                'content_id': i,
                'content_type': content_type,
                'genre': genre,
                'duration_min': duration,
                'quality_score': quality_score,
                'release_year': release_year,
                'popularity': popularity
            })
        
        return pd.DataFrame(content)
    
    def generate_interactions(self, users_df, content_df):
        """Generate user-content interactions"""
        logger.info(f"Generating {self.n_interactions} interactions...")
        
        interactions = []
        
        # Pre-compute user preferences
        user_genre_prefs = {}
        for _, user in users_df.iterrows():
            user_genre_prefs[user['user_id']] = user['preferred_genres'].split(',')
        
        for _ in range(self.n_interactions):
            # Sample user (power law distribution - some users more active)
            user_id = np.random.choice(
                users_df['user_id'].values,
                p=np.random.dirichlet(np.ones(len(users_df)) * 0.5)
            )
            user = users_df[users_df['user_id'] == user_id].iloc[0]
            
            # Sample content (biased toward popular and matching genres)
            content_id = np.random.choice(
                content_df['content_id'].values,
                p=np.random.dirichlet(np.ones(len(content_df)) * 0.3)
            )
            content = content_df[content_df['content_id'] == content_id].iloc[0]
            
            # Generate interaction features
            timestamp = datetime.now() - timedelta(days=np.random.randint(0, 365))
            hour = timestamp.hour
            day_of_week = timestamp.weekday()
            is_weekend = 1 if day_of_week >= 5 else 0
            
            # Device for this session
            device = np.random.choice(self.devices, p=[0.40, 0.25, 0.15, 0.15, 0.05])
            is_preferred_device = 1 if device == user['preferred_device'] else 0
            
            # Genre match
            user_prefs = user_genre_prefs[user_id]
            genre_match = 1 if content['genre'] in user_prefs else 0
            
            # Calculate engagement probability
            base_prob = 0.15
            
            # Factors that increase engagement
            prob = base_prob
            prob += 0.20 * genre_match
            prob += 0.15 * content['quality_score']
            prob += 0.10 * content['popularity']
            prob += 0.10 * user['engagement_score']
            prob += 0.05 * is_preferred_device
            prob += 0.05 * (1 if content['release_year'] >= 2022 else 0)
            
            # Peak hours boost
            if hour in [19, 20, 21, 22]:
                prob += 0.05
            
            # Weekend boost
            if is_weekend:
                prob += 0.05
            
            prob = np.clip(prob, 0, 1)
            
            # Generate outcomes
            clicked = 1 if np.random.random() < prob else 0
            
            if clicked:
                # Watch time (correlated with quality and interest)
                watch_ratio = np.random.beta(5, 2) * (0.7 + 0.3 * genre_match)
                watch_ratio = np.clip(watch_ratio, 0, 1)
                watch_time = content['duration_min'] * watch_ratio
                
                # Rating (if watched enough)
                if watch_ratio > 0.3:
                    rating = np.random.choice([1, 2, 3, 4, 5], 
                                             p=[0.05, 0.10, 0.25, 0.35, 0.25])
                else:
                    rating = 0
                
                completed = 1 if watch_ratio > 0.9 else 0
            else:
                watch_time = 0
                watch_ratio = 0
                rating = 0
                completed = 0
            
            # Create cross features for wide component
            user_content_cross = f"{user_id}_{content_id}"
            user_genre_cross = f"{user_id}_{content['genre']}"
            device_genre_cross = f"{device}_{content['genre']}"
            
            interaction = {
                'user_id': user_id,
                'content_id': content_id,
                'timestamp': timestamp,
                'hour': hour,
                'day_of_week': day_of_week,
                'is_weekend': is_weekend,
                'device': device,
                'is_preferred_device': is_preferred_device,
                'genre': content['genre'],
                'genre_match': genre_match,
                'content_type': content['content_type'],
                'duration_min': content['duration_min'],
                'release_year': content['release_year'],
                'content_quality': content['quality_score'],
                'content_popularity': content['popularity'],
                'user_engagement': user['engagement_score'],
                'user_age_group': user['age_group'],
                'clicked': clicked,
                'watch_time_min': watch_time,
                'watch_ratio': watch_ratio,
                'rating': rating,
                'completed': completed,
                'user_content_cross': user_content_cross,
                'user_genre_cross': user_genre_cross,
                'device_genre_cross': device_genre_cross
            }
            
            interactions.append(interaction)
        
        df = pd.DataFrame(interactions)
        logger.info(f"Generated {len(df)} interactions")
        logger.info(f"Click-through rate: {df['clicked'].mean():.2%}")
        logger.info(f"Completion rate: {df['completed'].mean():.2%}")
        
        return df


def main():
    parser = argparse.ArgumentParser(description='Generate recommendation data')
    parser.add_argument('--users', type=int, default=10000, help='Number of users')
    parser.add_argument('--content', type=int, default=50000, help='Number of content items')
    parser.add_argument('--samples', type=int, default=1000000, help='Number of interactions')
    parser.add_argument('--output-dir', type=str, default='data/raw', help='Output directory')
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate data
    generator = InteractionDataGenerator(
        n_users=args.users,
        n_content=args.content,
        n_interactions=args.samples
    )
    
    # Generate users and content
    users_df = generator.generate_user_profiles()
    content_df = generator.generate_content_catalog()
    interactions_df = generator.generate_interactions(users_df, content_df)
    
    # Save data
    users_df.to_csv(f'{args.output_dir}/users.csv', index=False)
    content_df.to_csv(f'{args.output_dir}/content.csv', index=False)
    interactions_df.to_csv(f'{args.output_dir}/interactions.csv', index=False)
    
    logger.info(f"\nData saved to {args.output_dir}/")
    logger.info(f"Users: {len(users_df)}")
    logger.info(f"Content: {len(content_df)}")
    logger.info(f"Interactions: {len(interactions_df)}")
    
    # Print sample
    print("\nSample interactions:")
    print(interactions_df.head())


if __name__ == "__main__":
    main()