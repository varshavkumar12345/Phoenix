import os
import json
import logging
from datetime import datetime
import torch
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DDoSDataProcessor:
    """Processes and manages DDoS attack training data"""
    
    def __init__(self, central_data_path=None):
        self.central_data_path = central_data_path or os.getenv("CENTRAL_DATA_PATH")
        self._ensure_central_store_exists()
    
    def _ensure_central_store_exists(self):
        """Create central dataset file if it doesn't exist"""
        if not os.path.exists(self.central_data_path):
            initial_data = []
            with open(self.central_data_path, 'w', encoding='utf-8') as f:
                json.dump(initial_data, f, indent=2)
            logger.info(f"Created central dataset: {self.central_data_path}")
    
    def load_from_central_store(self):
        """Load all training data from central store"""
        try:
            with open(self.central_data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Loaded {len(data)} samples from central store")
            return data
        except Exception as e:
            logger.error(f"Failed to load central store: {e}")
            return []
    
    def append_unique_samples(self, new_samples):
        """Add new unique samples to central store"""
        try:
            existing_data = self.load_from_central_store()
            
            # Create lookup of existing samples by features
            existing_lookup = {
                self._sample_key(s): s for s in existing_data
            }
            
            added_count = 0
            for sample in new_samples:
                key = self._sample_key(sample)
                if key not in existing_lookup:
                    existing_lookup[key] = sample
                    added_count += 1
            
            # Save back to central store
            updated_data = list(existing_lookup.values())
            with open(self.central_data_path, 'w', encoding='utf-8') as f:
                json.dump(updated_data, f, indent=2)
            
            logger.info(f"Added {added_count} unique samples to central store (total: {len(updated_data)})")
            return added_count
        except Exception as e:
            logger.error(f"Failed to append samples: {e}")
            return 0
    
    def _sample_key(self, sample):
        """Create unique key for a sample"""
        # Use sequence history and current features as key
        history = tuple(tuple(x) for x in sample.get('history', []))
        current = tuple(sample.get('current', []))
        return (history, current)
    
    def save_for_training(self, samples, output_path="training_data_temp.json"):
        """Save samples in format expected by training scripts"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(samples, f, indent=2)
            logger.info(f"Saved {len(samples)} samples to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Failed to save training data: {e}")
            return None


class ContinuousLearningManager:
    """Manages continuous learning for DDoS attack prediction"""
    
    def __init__(self, feedback_path=None, threshold=20, reload_callback=None):
        self.feedback_path = feedback_path or os.getenv(
            "FEEDBACK_DATA_PATH_DDOS", 
            "ddos_feedback.json"
        )
        self.threshold = threshold
        self.reload_callback = reload_callback
        self.processor = DDoSDataProcessor()
    
    def log_interaction(self, history, current, ip, predicted_action, 
                       predicted_suspicious, actual_action, timestamp=None):
        """Log a DDoS prediction interaction for continuous learning"""
        entry = {
            "history": history,
            "current": current,
            "ip": ip,
            "predicted_action": int(predicted_action),
            "predicted_suspicious": float(predicted_suspicious),
            "actual_action": int(actual_action),
            "action": int(actual_action),  # for training
            "suspicious_score": float(predicted_suspicious),
            "timestamp": timestamp or datetime.utcnow().isoformat()
        }
        
        # Save to feedback file
        if not os.path.exists(self.feedback_path):
            with open(self.feedback_path, 'w', encoding='utf-8') as f:
                json.dump([entry], f, indent=2)
        else:
            with open(self.feedback_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            data.append(entry)
            with open(self.feedback_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        
        logger.info(f"Logged interaction: IP={ip}, Predicted={predicted_action}, Actual={actual_action}")
    
    def incorporate_feedback(self):
        """Process feedback and retrain models if threshold is reached"""
        if not os.path.exists(self.feedback_path):
            logger.warning("No feedback file found")
            return
        
        # Load feedback
        with open(self.feedback_path, 'r', encoding='utf-8') as f:
            feedback = json.load(f)
        
        if not feedback:
            logger.info("No feedback to process")
            return
        
        # Remove duplicates
        unique_samples = {}
        for item in feedback:
            key = (tuple(tuple(x) for x in item['history']), tuple(item['current']))
            unique_samples[key] = item
        
        new_samples = list(unique_samples.values())
        logger.info(f"Processing {len(new_samples)} unique feedback samples")
        
        # Add to central store
        added = self.processor.append_unique_samples(new_samples)
        
        # Check if retraining threshold is met
        if len(new_samples) >= self.threshold:
            logger.info(f"Threshold {self.threshold} reached - initiating retraining...")
            
            # Load full dataset
            raw_dataset = self.processor.load_from_central_store()
            
            if len(raw_dataset) < 20:
                logger.warning(f"Dataset too small ({len(raw_dataset)} samples), skipping retraining")
                return
            
            # Import and execute training scripts
            try:
                logger.info("=" * 80)
                logger.info("STEP 1: Training LSTM Model")
                logger.info("=" * 80)
                
                # Import train_lstm module and execute
                import train_lstm
                if hasattr(train_lstm, 'train'):
                    train_lstm.train()
                else:
                    logger.error("train_lstm.py does not have a train() function")
                    return
                
                logger.info("=" * 80)
                logger.info("STEP 2: Training XGBoost Model")
                logger.info("=" * 80)
                
                # Import train_xgb module and execute
                import train_xgb
                if hasattr(train_xgb, 'main'):
                    train_xgb.main()
                else:
                    logger.error("train_xgb.py does not have a main() function")
                    return
                
                logger.info("=" * 80)
                logger.info("Training Pipeline Completed Successfully")
                logger.info("=" * 80)
                
            except ImportError as e:
                logger.error(f"Failed to import training modules: {e}")
                logger.error("Make sure train_lstm.py and train_xgb.py are in the same directory")
                return
            except Exception as e:
                logger.error(f"Training failed: {e}")
                return
            
            # Clear feedback file after successful training
            os.remove(self.feedback_path)
            logger.info("Feedback file cleared after retraining")
            
            # Trigger reload callback if provided
            if self.reload_callback:
                logger.info("Triggering model reload in server...")
                self.reload_callback()
        else:
            logger.info(f"Threshold not met: {len(new_samples)}/{self.threshold} samples")
    
    def get_feedback_count(self):
        """Get current feedback count"""
        if not os.path.exists(self.feedback_path):
            return 0
        try:
            with open(self.feedback_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return len(data)
        except:
            return 0


# Example usage
if __name__ == "__main__":
    # Initialize manager
    manager = ContinuousLearningManager(threshold=20)
    
    # Example: Log an interaction
    history = [[100, 10, 0.1]] * 10  # 10 timesteps of [pps, unique_ips, syn_ratio]
    current = [150, 15, 0.15]
    
    manager.log_interaction(
        history=history,
        current=current,
        ip="192.168.1.100",
        predicted_action=0,
        predicted_suspicious=0.3,
        actual_action=1  # Human corrected to rate_limit
    )
    
    # Check and incorporate feedback
    manager.incorporate_feedback()