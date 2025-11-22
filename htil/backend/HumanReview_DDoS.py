import os
import json
import logging
import requests
from datetime import datetime
from typing import List, Dict
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from ContinuousLearning_DDoS import ContinuousLearningManager
import threading

load_dotenv()

# Configuration
FEEDBACK_DATA_PATH_DDOS= os.getenv("")

class ReviewConfig:
    INTERMEDIATE_JSON_PATH= os.getenv("INTERMEDIATE_JSON_PATH")
    CENTRAL_DATA_PATH_DDOS= os.getenv("CENTRAL_DATA_PATH")
    REVIEW_HISTORY_PATH= os.getenv("REVIEW_HISTORY_PATH")
    RETRAIN_THRESHOLD = 20
    CORRECT_REWARD = 1.0
    INCORRECT_PENALTY = -0.5
    CORRECTION_REWARD = 0.8
    FIREWALL_API_URL = "http://localhost:8000"

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('ddos_human_review.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


class DDoSHumanReviewManager:
    """Manages human review and continuous learning for DDoS predictions"""
    
    def __init__(self):
        self.config = ReviewConfig()
        self.review_history = self._load_review_history()
        self.current_session_reviews = []
        self.retraining_in_progress = False
        
        # Initialize continuous learning manager
        self.learning_manager = ContinuousLearningManager(
            feedback_path=FEEDBACK_DATA_PATH_DDOS,
            threshold=self.config.RETRAIN_THRESHOLD,
            reload_callback=self._trigger_model_reload
        )
    
    def _trigger_model_reload(self):
        """Trigger model reload in the FastAPI server"""
        try:
            response = requests.post(f"{self.config.FIREWALL_API_URL}/reload_models")
            if response.status_code == 200:
                logger.info("Models reloaded in Firewall API successfully")
            else:
                logger.error(f"Failed to reload models in Firewall API: {response.text}")
        except Exception as e:
            logger.error(f"Error triggering model reload: {e}")
    
    def _load_review_history(self) -> Dict:
        """Load review history from disk"""
        if os.path.exists(self.config.REVIEW_HISTORY_PATH):
            try:
                with open(self.config.REVIEW_HISTORY_PATH, 'r', encoding='utf-8') as f:
                    history = json.load(f)
                logger.info(f"Loaded review history: {len(history.get('reviews', []))} reviews")
                return history
            except Exception as e:
                logger.warning(f"Could not load review history: {e}")
        
        return {
            "reviews": [],
            "total_reviewed": 0,
            "correct_predictions": 0,
            "incorrect_predictions": 0,
            "accuracy": 0.0,
            "action_breakdown": {
                "allow": {"correct": 0, "incorrect": 0},
                "rate_limit": {"correct": 0, "incorrect": 0},
                "block": {"correct": 0, "incorrect": 0}
            },
            "last_review_timestamp": None
        }
    
    def _save_review_history(self):
        """Save review history to disk"""
        try:
            with open(self.config.REVIEW_HISTORY_PATH, 'w', encoding='utf-8') as f:
                json.dump(self.review_history, f, ensure_ascii=False, indent=2)
            logger.info("Review history saved")
        except Exception as e:
            logger.error(f"Failed to save review history: {e}")
    
    def _extract_network_parameters(self, sample: Dict) -> Dict:
        """Extract and format network parameters from sample for user display"""
        try:
            # Get current traffic data
            current = sample.get('current', [])
            history = sample.get('history', [])
            
            # Feature indices based on typical DDoS detection features
            # Adjust these indices based on your actual feature order
            params = {}
            
            if current and len(current) > 0:
                features = current[0] if isinstance(current[0], list) else current
                
                # Extract key parameters (adjust indices as needed)
                params['packets_per_second'] = round(features[0], 2) if len(features) > 0 else 0
                params['bytes_per_second'] = round(features[1], 2) if len(features) > 1 else 0
                params['syn_ratio'] = round(features[2], 4) if len(features) > 2 else 0
                params['unique_ports'] = int(features[3]) if len(features) > 3 else 0
                params['packet_size_avg'] = round(features[4], 2) if len(features) > 4 else 0
                params['packet_size_std'] = round(features[5], 2) if len(features) > 5 else 0
                params['tcp_ratio'] = round(features[6], 4) if len(features) > 6 else 0
                params['udp_ratio'] = round(features[7], 4) if len(features) > 7 else 0
                params['icmp_ratio'] = round(features[8], 4) if len(features) > 8 else 0
                params['entropy'] = round(features[9], 4) if len(features) > 9 else 0
                
                # Calculate historical averages if available
                if history and len(history) > 0:
                    params['historical_avg_pps'] = round(sum(h[0] for h in history) / len(history), 2)
                    params['historical_avg_bps'] = round(sum(h[1] for h in history) / len(history), 2)
                    params['traffic_spike_ratio'] = round(params['packets_per_second'] / params['historical_avg_pps'], 2) if params['historical_avg_pps'] > 0 else 0
            
            return params
        except Exception as e:
            logger.error(f"Error extracting network parameters: {e}")
            return {}
    
    def _get_parameter_analysis(self, params: Dict) -> Dict:
        """Analyze parameters and provide indicators for review"""
        analysis = {
            'suspicious_indicators': [],
            'normal_indicators': [],
            'severity_score': 0
        }
        
        try:
            # Check for suspicious patterns
            if params.get('packets_per_second', 0) > 10000:
                analysis['suspicious_indicators'].append(f"High PPS: {params['packets_per_second']}")
                analysis['severity_score'] += 2
            
            if params.get('syn_ratio', 0) > 0.7:
                analysis['suspicious_indicators'].append(f"High SYN ratio: {params['syn_ratio']} (possible SYN flood)")
                analysis['severity_score'] += 3
            
            if params.get('traffic_spike_ratio', 0) > 5:
                analysis['suspicious_indicators'].append(f"Traffic spike: {params['traffic_spike_ratio']}x normal")
                analysis['severity_score'] += 2
            
            if params.get('entropy', 0) < 0.3:
                analysis['suspicious_indicators'].append(f"Low entropy: {params['entropy']} (repetitive pattern)")
                analysis['severity_score'] += 1
            
            if params.get('unique_ports', 0) < 5:
                analysis['suspicious_indicators'].append(f"Few unique ports: {params['unique_ports']}")
                analysis['severity_score'] += 1
            
            # Check for normal patterns
            if params.get('packets_per_second', 0) < 1000:
                analysis['normal_indicators'].append(f"Normal PPS: {params['packets_per_second']}")
            
            if 0.2 < params.get('syn_ratio', 0) < 0.5:
                analysis['normal_indicators'].append(f"Normal SYN ratio: {params['syn_ratio']}")
            
            if params.get('traffic_spike_ratio', 0) < 2:
                analysis['normal_indicators'].append("Traffic within normal range")
            
        except Exception as e:
            logger.error(f"Error analyzing parameters: {e}")
        
        return analysis
    
    def load_intermediate_samples(self) -> List[Dict]:
        """Load samples pending human review with enhanced parameter display"""
        if not os.path.exists(self.config.INTERMEDIATE_JSON_PATH):
            logger.warning(f"Intermediate file not found: {self.config.INTERMEDIATE_JSON_PATH}")
            return []
        
        try:
            with open(self.config.INTERMEDIATE_JSON_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, dict):
                samples = data.get('samples', [data])
            elif isinstance(data, list):
                samples = data
            else:
                logger.error("Invalid intermediate.json format")
                return []
            
            # Enhance samples with extracted parameters and analysis
            enhanced_samples = []
            for sample in samples:
                enhanced_sample = sample.copy()
                
                # Extract network parameters
                params = self._extract_network_parameters(sample)
                enhanced_sample['network_parameters'] = params
                
                # Add parameter analysis
                analysis = self._get_parameter_analysis(params)
                enhanced_sample['parameter_analysis'] = analysis
                
                # Add human-readable action
                action_names = {0: "ALLOW", 1: "RATE_LIMIT", 2: "BLOCK"}
                enhanced_sample['predicted_action_name'] = action_names.get(
                    sample.get('predicted_action', 0), 
                    "UNKNOWN"
                )
                
                enhanced_samples.append(enhanced_sample)
            
            logger.info(f"Loaded {len(enhanced_samples)} samples for review with network parameters")
            return enhanced_samples
        except Exception as e:
            logger.error(f"Failed to load intermediate samples: {e}")
            return []
    
    def process_review(self, sample: Dict, is_correct: bool, corrected_action: int = None) -> Dict:
        """Process a human review of a DDoS prediction"""
        try:
            action_names = {0: "ALLOW", 1: "RATE_LIMIT", 2: "BLOCK"}
            
            # Determine final action
            if is_correct:
                reward = self.config.CORRECT_REWARD
                final_action = sample.get('predicted_action', 0)
                result_message = f"Correct prediction - Model rewarded ({action_names[final_action]})"
            else:
                reward = self.config.INCORRECT_PENALTY
                final_action = corrected_action if corrected_action is not None else sample.get('predicted_action', 0)
                result_message = f"Incorrect prediction - Model penalized. Corrected to: {action_names[final_action]}"
            
            # Create reviewed sample record
            reviewed_sample = {
                "history": sample.get('history', []),
                "current": sample.get('current', []),
                "ip": sample.get('ip', 'unknown'),
                "predicted_action": sample.get('predicted_action'),
                "predicted_suspicious": sample.get('predicted_suspicious'),
                "network_parameters": sample.get('network_parameters', {}),
                "parameter_analysis": sample.get('parameter_analysis', {}),
                "human_action": final_action,
                "is_correct": is_correct,
                "reward": reward,
                "review_timestamp": datetime.utcnow().isoformat(),
                "reviewer": "human"
            }
            
            # Update session and history
            self.current_session_reviews.append(reviewed_sample)
            self.review_history["reviews"].append(reviewed_sample)
            self.review_history["total_reviewed"] += 1
            
            # Update accuracy stats
            if is_correct:
                self.review_history["correct_predictions"] += 1
            else:
                self.review_history["incorrect_predictions"] += 1
            
            # Update action breakdown
            predicted_action_name = action_names[sample.get('predicted_action', 0)].lower()
            if is_correct:
                self.review_history["action_breakdown"][predicted_action_name]["correct"] += 1
            else:
                self.review_history["action_breakdown"][predicted_action_name]["incorrect"] += 1
            
            # Calculate overall accuracy
            total = self.review_history["total_reviewed"]
            if total > 0:
                accuracy = (self.review_history["correct_predictions"] / total) * 100
                self.review_history["accuracy"] = round(accuracy, 2)
            
            self.review_history["last_review_timestamp"] = datetime.utcnow().isoformat()
            self._save_review_history()
            
            # Log to continuous learning system
            logger.info(f"Logging interaction to continuous learning system")
            self.learning_manager.log_interaction(
                history=sample.get('history', []),
                current=sample.get('current', []),
                ip=sample.get('ip', 'unknown'),
                predicted_action=sample.get('predicted_action', 0),
                predicted_suspicious=sample.get('predicted_suspicious', 0.0),
                actual_action=final_action
            )
            
            # Check if retraining should be triggered
            self._check_and_trigger_auto_retraining()
            
            return {
                "success": True,
                "message": result_message,
                "reward": reward,
                "reviewed_sample": reviewed_sample
            }
        
        except Exception as e:
            logger.error(f"Error in process_review: {e}", exc_info=True)
            raise
    
    def _check_and_trigger_auto_retraining(self):
        """Check if auto-retraining threshold is met"""
        if self.retraining_in_progress:
            logger.info("Retraining already in progress, skipping auto-trigger")
            return
        
        feedback_count = self.learning_manager.get_feedback_count()
        
        if feedback_count >= self.config.RETRAIN_THRESHOLD:
            logger.info(f"Threshold reached - {feedback_count} feedback samples")
            logger.info("Starting automatic retraining in background...")
            retrain_thread = threading.Thread(target=self._execute_retraining)
            retrain_thread.daemon = True
            retrain_thread.start()
    
    def _execute_retraining(self):
        """Execute model retraining in background"""
        try:
            self.retraining_in_progress = True
            logger.info("=" * 80)
            logger.info("DDoS MODEL RETRAINING STARTED")
            logger.info("=" * 80)
            
            # Incorporate feedback and retrain
            self.learning_manager.incorporate_feedback()
            
            # Clear intermediate file
            self.current_session_reviews = []
            self._clear_intermediate_file()
            
            logger.info("=" * 80)
            logger.info("DDoS MODEL RETRAINING COMPLETED SUCCESSFULLY")
            logger.info("=" * 80)
        
        except Exception as e:
            logger.error(f"Error during retraining: {e}", exc_info=True)
        finally:
            self.retraining_in_progress = False
    
    def trigger_retraining(self) -> Dict:
        """Manually trigger model retraining"""
        if self.retraining_in_progress:
            return {
                "success": False,
                "message": "Retraining already in progress. Please wait..."
            }
        
        logger.info("Manual trigger: Starting retraining in background...")
        retrain_thread = threading.Thread(target=self._execute_retraining)
        retrain_thread.daemon = True
        retrain_thread.start()
        
        return {
            "success": True,
            "message": "Model retraining started in background. Models will be updated automatically."
        }
    
    def _clear_intermediate_file(self):
        """Clear intermediate file after processing"""
        try:
            if os.path.exists(self.config.INTERMEDIATE_JSON_PATH):
                # Create backup
                backup_path = f"{self.config.INTERMEDIATE_JSON_PATH}.backup"
                with open(self.config.INTERMEDIATE_JSON_PATH, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                with open(backup_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                
                # Clear file
                with open(self.config.INTERMEDIATE_JSON_PATH, 'w', encoding='utf-8') as f:
                    json.dump([], f)
                
                logger.info("Intermediate file cleared (backup saved)")
        except Exception as e:
            logger.error(f"Failed to clear intermediate file: {e}")
    
    def get_statistics(self) -> Dict:
        """Get review and training statistics"""
        feedback_count = self.learning_manager.get_feedback_count()
        
        return {
            "success": True,
            "stats": {
                "total_reviewed": self.review_history.get('total_reviewed', 0),
                "correct_predictions": self.review_history.get('correct_predictions', 0),
                "incorrect_predictions": self.review_history.get('incorrect_predictions', 0),
                "accuracy": self.review_history.get('accuracy', 0.0),
                "action_breakdown": self.review_history.get('action_breakdown', {}),
                "last_review_timestamp": self.review_history.get('last_review_timestamp'),
                "current_session_reviews": len(self.current_session_reviews),
                "retraining_in_progress": self.retraining_in_progress,
                "pending_feedback": {
                    "count": feedback_count,
                    "threshold": self.config.RETRAIN_THRESHOLD
                }
            }
        }


# Flask API setup
app = Flask(__name__)
CORS(app)

review_manager = DDoSHumanReviewManager()


@app.route('/load_samples', methods=['GET'])
def load_samples():
    """Load samples pending human review"""
    try:
        samples = review_manager.load_intermediate_samples()
        
        if not samples:
            return jsonify({
                "success": False,
                "message": "No samples available in intermediate.json",
                "samples": []
            })
        
        return jsonify({
            "success": True,
            "samples": samples,
            "count": len(samples)
        })
    
    except Exception as e:
        logger.error(f"Error loading samples: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "message": f"Error loading samples: {str(e)}",
            "samples": []
        }), 500


@app.route('/review', methods=['POST'])
def review_sample():
    """Submit a human review for a DDoS prediction"""
    try:
        data = request.get_json()
        sample = data.get('sample')
        is_correct = data.get('is_correct')
        corrected_action = data.get('corrected_action')
        
        if sample is None or is_correct is None:
            return jsonify({
                "success": False,
                "message": "Missing sample or is_correct parameter"
            }), 400
        
        result = review_manager.process_review(sample, is_correct, corrected_action)
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error processing review: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "message": f"Error processing review: {str(e)}"
        }), 500


@app.route('/stats', methods=['GET'])
def get_stats():
    """Get review and training statistics"""
    try:
        return jsonify(review_manager.get_statistics())
    except Exception as e:
        logger.error(f"Error getting stats: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "message": f"Error getting stats: {str(e)}"
        }), 500


@app.route('/trigger_retraining', methods=['POST'])
def trigger_retraining():
    """Manually trigger model retraining"""
    try:
        result = review_manager.trigger_retraining()
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error triggering retraining: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "message": f"Error triggering retraining: {str(e)}"
        }), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "success": True,
        "message": "DDoS Human Review API is running",
        "timestamp": datetime.utcnow().isoformat(),
        "retraining_in_progress": review_manager.retraining_in_progress
    })


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("DDoS AUTONOMOUS FIREWALL - HUMAN REVIEW API")
    print("=" * 80)
    print("Continuous Learning through Human-in-the-Loop Feedback")
    print("LSTM + XGBoost Model Training Pipeline")
    print("=" * 80)
    print("\nServer starting on http://localhost:5002")
    print("CORS enabled for frontend")
    print("\nEndpoints:")
    print("   GET  /load_samples       - Load samples for review")
    print("   POST /review             - Submit review feedback")
    print("   GET  /stats              - Get review statistics")
    print("   POST /trigger_retraining - Trigger model retraining")
    print("   GET  /health             - Health check")
    print(f"\nAuto-retraining threshold: {ReviewConfig.RETRAIN_THRESHOLD} samples")
    print("=" * 80 + "\n")
    
    app.run(host="0.0.0.0", port=5002, debug=False)