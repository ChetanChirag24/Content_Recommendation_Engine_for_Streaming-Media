"""
ONNX Inference for Content Recommendation
Optimized real-time serving with <50ms latency
"""

import numpy as np
import onnxruntime as ort
import time
import torch
from pathlib import Path
import logging
from typing import Dict, List
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ONNXRecommendationService:
    """Real-time recommendation service using ONNX"""
    
    def __init__(self, model_path: str):
        """
        Initialize ONNX inference session
        
        Args:
            model_path: Path to ONNX model file
        """
        self.model_path = model_path
        
        # Create ONNX Runtime session with optimization
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4
        
        self.session = ort.InferenceSession(
            model_path,
            sess_options,
            providers=['CPUExecutionProvider']  # Use 'CUDAExecutionProvider' for GPU
        )
        
        # Get input/output names
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]
        
        logger.info(f"ONNX model loaded from {model_path}")
        logger.info(f"Input names: {self.input_names}")
        logger.info(f"Output names: {self.output_names}")
    
    def predict_batch(self, 
                     wide_features: np.ndarray,
                     deep_cont_features: np.ndarray,
                     deep_cat_features: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Generate predictions for a batch
        
        Args:
            wide_features: Wide features (batch_size, wide_dim)
            deep_cont_features: Deep continuous features (batch_size, deep_dim)
            deep_cat_features: Dict of categorical features
        
        Returns:
            predictions: Array of probabilities (batch_size,)
        """
        # Prepare inputs
        inputs = {
            'wide_features': wide_features.astype(np.float32),
            'deep_cont_features': deep_cont_features.astype(np.float32),
        }
        
        # Add categorical features
        for feat_name, feat_values in deep_cat_features.items():
            inputs[feat_name] = feat_values.astype(np.int64)
        
        # Run inference
        outputs = self.session.run(self.output_names, inputs)
        logits = outputs[0]
        
        # Apply sigmoid
        probs = 1 / (1 + np.exp(-logits))
        
        return probs.flatten()
    
    def predict_single(self,
                       user_features: Dict,
                       content_features: Dict,
                       context: Dict) -> float:
        """
        Generate prediction for single user-content pair
        
        Args:
            user_features: User feature dict
            content_features: Content feature dict
            context: Context features (device, time, etc.)
        
        Returns:
            probability: Click probability
        """
        # Convert to batch format
        wide = np.array([self._create_wide_features(user_features, content_features, context)])
        deep_cont = np.array([self._create_deep_cont_features(user_features, content_features, context)])
        deep_cat = self._create_deep_cat_features(user_features, content_features, context)
        
        # Predict
        prob = self.predict_batch(wide, deep_cont, deep_cat)[0]
        
        return float(prob)
    
    def _create_wide_features(self, user_feat, content_feat, context):
        """Create wide (cross-product) features"""
        # This is a simplified version - in production, use actual feature engineering
        features = np.zeros(100)
        
        # Example cross features
        features[0] = hash(f"{user_feat.get('user_id', 0)}_{content_feat.get('content_id', 0)}") % 100
        features[1] = hash(f"{user_feat.get('user_id', 0)}_{content_feat.get('genre', '')}") % 100
        features[2] = hash(f"{context.get('device', '')}_{content_feat.get('genre', '')}") % 100
        
        return features
    
    def _create_deep_cont_features(self, user_feat, content_feat, context):
        """Create deep continuous features"""
        features = np.zeros(50)
        
        # User features
        features[0] = user_feat.get('age_group', 25) / 100.0
        features[1] = user_feat.get('engagement_score', 0.5)
        
        # Content features
        features[2] = content_feat.get('duration_min', 60) / 180.0
        features[3] = content_feat.get('quality_score', 0.5)
        features[4] = content_feat.get('popularity', 0.3)
        features[5] = (content_feat.get('release_year', 2020) - 2015) / 10.0
        
        # Context features
        features[6] = context.get('hour', 12) / 24.0
        features[7] = context.get('is_weekend', 0)
        features[8] = context.get('is_preferred_device', 0)
        features[9] = context.get('genre_match', 0)
        
        return features
    
    def _create_deep_cat_features(self, user_feat, content_feat, context):
        """Create deep categorical features"""
        features = {
            'user_id': np.array([user_feat.get('user_id', 0)]),
            'content_id': np.array([content_feat.get('content_id', 0)]),
            'genre': np.array([self._encode_genre(content_feat.get('genre', 'Drama'))]),
            'device': np.array([self._encode_device(context.get('device', 'mobile'))]),
            'time_bucket': np.array([context.get('hour', 12)])
        }
        return features
    
    def _encode_genre(self, genre):
        """Encode genre to integer"""
        genre_map = {
            'Action': 0, 'Comedy': 1, 'Drama': 2, 'Thriller': 3, 'Sci-Fi': 4,
            'Romance': 5, 'Horror': 6, 'Documentary': 7, 'Animation': 8, 'Musical': 9
        }
        return genre_map.get(genre, 2)  # Default to Drama
    
    def _encode_device(self, device):
        """Encode device to integer"""
        device_map = {'mobile': 0, 'desktop': 1, 'tablet': 2, 'smart_tv': 3, 'console': 4}
        return device_map.get(device, 0)  # Default to mobile
    
    def recommend_top_k(self,
                        user_id: int,
                        candidate_content: List[Dict],
                        context: Dict,
                        k: int = 10) -> List[Dict]:
        """
        Recommend top-k content items for a user
        
        Args:
            user_id: User ID
            candidate_content: List of candidate content items
            context: Context features
            k: Number of recommendations
        
        Returns:
            List of top-k recommendations with scores
        """
        if len(candidate_content) == 0:
            return []
        
        # Prepare batch features for all candidates
        batch_size = len(candidate_content)
        
        wide_batch = np.zeros((batch_size, 100))
        deep_cont_batch = np.zeros((batch_size, 50))
        deep_cat_batch = {
            'user_id': np.zeros(batch_size, dtype=np.int64),
            'content_id': np.zeros(batch_size, dtype=np.int64),
            'genre': np.zeros(batch_size, dtype=np.int64),
            'device': np.zeros(batch_size, dtype=np.int64),
            'time_bucket': np.zeros(batch_size, dtype=np.int64)
        }
        
        # Placeholder user features
        user_features = {'user_id': user_id, 'age_group': 25, 'engagement_score': 0.7}
        
        for idx, content in enumerate(candidate_content):
            wide_batch[idx] = self._create_wide_features(user_features, content, context)
            deep_cont_batch[idx] = self._create_deep_cont_features(user_features, content, context)
            
            cat_feats = self._create_deep_cat_features(user_features, content, context)
            for feat_name, feat_val in cat_feats.items():
                deep_cat_batch[feat_name][idx] = feat_val[0]
        
        # Get predictions
        scores = self.predict_batch(wide_batch, deep_cont_batch, deep_cat_batch)
        
        # Get top-k
        top_indices = np.argsort(scores)[::-1][:k]
        
        recommendations = []
        for idx in top_indices:
            rec = candidate_content[idx].copy()
            rec['score'] = float(scores[idx])
            rec['rank'] = len(recommendations) + 1
            recommendations.append(rec)
        
        return recommendations


class LatencyBenchmark:
    """Benchmark inference latency"""
    
    def __init__(self, service: ONNXRecommendationService):
        self.service = service
    
    def run_benchmark(self, n_requests=10000, batch_size=1):
        """
        Run latency benchmark
        
        Args:
            n_requests: Number of requests to simulate
            batch_size: Batch size for each request
        
        Returns:
            Dict with latency statistics
        """
        logger.info(f"Running benchmark with {n_requests} requests, batch_size={batch_size}")
        
        latencies = []
        
        for _ in range(n_requests):
            # Generate random features
            wide = np.random.randn(batch_size, 100).astype(np.float32)
            deep_cont = np.random.randn(batch_size, 50).astype(np.float32)
            deep_cat = {
                'user_id': np.random.randint(0, 10000, batch_size, dtype=np.int64),
                'content_id': np.random.randint(0, 50000, batch_size, dtype=np.int64),
                'genre': np.random.randint(0, 10, batch_size, dtype=np.int64),
                'device': np.random.randint(0, 5, batch_size, dtype=np.int64),
                'time_bucket': np.random.randint(0, 24, batch_size, dtype=np.int64)
            }
            
            # Measure latency
            start_time = time.perf_counter()
            _ = self.service.predict_batch(wide, deep_cont, deep_cat)
            end_time = time.perf_counter()
            
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
        
        # Calculate statistics
        latencies = np.array(latencies)
        stats = {
            'mean_latency_ms': np.mean(latencies),
            'median_latency_ms': np.median(latencies),
            'p50_latency_ms': np.percentile(latencies, 50),
            'p95_latency_ms': np.percentile(latencies, 95),
            'p99_latency_ms': np.percentile(latencies, 99),
            'min_latency_ms': np.min(latencies),
            'max_latency_ms': np.max(latencies),
            'throughput_rps': 1000 / np.mean(latencies)
        }
        
        return stats
    
    def print_results(self, stats):
        """Print benchmark results"""
        print("\n" + "="*60)
        print("INFERENCE LATENCY BENCHMARK RESULTS")
        print("="*60)
        print(f"Mean Latency:     {stats['mean_latency_ms']:.2f} ms")
        print(f"Median Latency:   {stats['median_latency_ms']:.2f} ms")
        print(f"P50 Latency:      {stats['p50_latency_ms']:.2f} ms")
        print(f"P95 Latency:      {stats['p95_latency_ms']:.2f} ms")
        print(f"P99 Latency:      {stats['p99_latency_ms']:.2f} ms")
        print(f"Min Latency:      {stats['min_latency_ms']:.2f} ms")
        print(f"Max Latency:      {stats['max_latency_ms']:.2f} ms")
        print(f"Throughput:       {stats['throughput_rps']:.0f} requests/second")
        print("="*60)


def convert_pytorch_to_onnx(pytorch_model_path: str, output_path: str):
    """
    Convert PyTorch model to ONNX format
    
    Args:
        pytorch_model_path: Path to PyTorch model (.pt file)
        output_path: Path to save ONNX model
    """
    from wide_deep import WideAndDeepModel, create_sample_model
    
    logger.info("Loading PyTorch model...")
    model = create_sample_model()
    model.load_state_dict(torch.load(pytorch_model_path))
    model.eval()
    
    # Create dummy inputs
    batch_size = 1
    wide_dummy = torch.randn(batch_size, 100)
    deep_cont_dummy = torch.randn(batch_size, 50)
    deep_cat_dummy = {
        'user_id': torch.randint(0, 10000, (batch_size,)),
        'content_id': torch.randint(0, 50000, (batch_size,)),
        'genre': torch.randint(0, 10, (batch_size,)),
        'device': torch.randint(0, 5, (batch_size,)),
        'time_bucket': torch.randint(0, 24, (batch_size,))
    }
    
    logger.info("Converting to ONNX...")
    
    # Export to ONNX
    torch.onnx.export(
        model,
        (wide_dummy, deep_cont_dummy, deep_cat_dummy),
        output_path,
        input_names=['wide_features', 'deep_cont_features', 'user_id', 'content_id', 
                    'genre', 'device', 'time_bucket'],
        output_names=['logits'],
        dynamic_axes={
            'wide_features': {0: 'batch_size'},
            'deep_cont_features': {0: 'batch_size'},
            'user_id': {0: 'batch_size'},
            'content_id': {0: 'batch_size'},
            'genre': {0: 'batch_size'},
            'device': {0: 'batch_size'},
            'time_bucket': {0: 'batch_size'},
            'logits': {0: 'batch_size'}
        },
        opset_version=13
    )
    
    logger.info(f"ONNX model saved to {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='ONNX Inference Service')
    parser.add_argument('--model-path', type=str, default='models/wide_deep.onnx',
                       help='Path to ONNX model')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run latency benchmark')
    parser.add_argument('--requests', type=int, default=10000,
                       help='Number of benchmark requests')
    parser.add_argument('--convert', action='store_true',
                       help='Convert PyTorch model to ONNX')
    parser.add_argument('--pytorch-model', type=str, default='models/best_model.pt',
                       help='Path to PyTorch model for conversion')
    
    args = parser.parse_args()
    
    if args.convert:
        # Convert PyTorch to ONNX
        convert_pytorch_to_onnx(args.pytorch_model, args.model_path)
        return
    
    # Load service
    service = ONNXRecommendationService(args.model_path)
    
    if args.benchmark:
        # Run benchmark
        benchmark = LatencyBenchmark(service)
        stats = benchmark.run_benchmark(n_requests=args.requests, batch_size=1)
        benchmark.print_results(stats)
        
        # Save results
        Path('results').mkdir(exist_ok=True)
        with open('results/latency_benchmark.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info("Benchmark results saved to results/latency_benchmark.json")
    
    else:
        # Example prediction
        logger.info("Running example prediction...")
        
        user_features = {
            'user_id': 123,
            'age_group': 25,
            'engagement_score': 0.75
        }
        
        content_features = {
            'content_id': 456,
            'genre': 'Action',
            'duration_min': 120,
            'quality_score': 0.85,
            'popularity': 0.7,
            'release_year': 2024
        }
        
        context = {
            'device': 'mobile',
            'hour': 20,
            'is_weekend': 1,
            'is_preferred_device': 1,
            'genre_match': 1
        }
        
        prob = service.predict_single(user_features, content_features, context)
        logger.info(f"Prediction probability: {prob:.4f}")
        
        # Top-K recommendations
        candidates = [
            {'content_id': i, 'genre': 'Action', 'quality_score': 0.8, 
             'duration_min': 120, 'popularity': 0.6, 'release_year': 2024}
            for i in range(100, 120)
        ]
        
        recommendations = service.recommend_top_k(
            user_id=123,
            candidate_content=candidates,
            context=context,
            k=10
        )
        
        print("\nTop 10 Recommendations:")
        for rec in recommendations:
            print(f"Rank {rec['rank']}: Content {rec['content_id']} - Score: {rec['score']:.4f}")


if __name__ == "__main__":
    main()