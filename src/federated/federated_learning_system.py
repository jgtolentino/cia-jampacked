"""
Federated Learning Infrastructure for Privacy-Preserving Personalization
Implements secure multi-party computation and differential privacy
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import asyncio
from datetime import datetime


class PrivacyMechanism(Enum):
    DIFFERENTIAL_PRIVACY = "differential_privacy"
    HOMOMORPHIC_ENCRYPTION = "homomorphic_encryption"
    SECURE_AGGREGATION = "secure_aggregation"
    TRUSTED_EXECUTION = "trusted_execution"


class ModelType(Enum):
    LOGISTIC_REGRESSION = "logistic_regression"
    NEURAL_NETWORK = "neural_network"
    GRADIENT_BOOSTING = "gradient_boosting"
    COLLABORATIVE_FILTERING = "collaborative_filtering"


@dataclass
class FederatedClient:
    """Represents a client in federated learning"""
    client_id: str
    data_size: int
    model_version: str
    last_update: datetime
    privacy_budget: float = 1.0
    is_active: bool = True


@dataclass
class ModelUpdate:
    """Encrypted model update from client"""
    client_id: str
    encrypted_weights: bytes
    gradient_norm: float
    num_samples: int
    timestamp: datetime
    privacy_noise_added: bool = False


class DifferentialPrivacy:
    """Differential privacy implementation"""
    
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5):
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = 1.0
    
    def add_noise(self, data: np.ndarray, mechanism: str = "laplace") -> np.ndarray:
        """Add privacy-preserving noise to data"""
        if mechanism == "laplace":
            scale = self.sensitivity / self.epsilon
            noise = np.random.laplace(0, scale, data.shape)
        elif mechanism == "gaussian":
            sigma = self.sensitivity * np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
            noise = np.random.normal(0, sigma, data.shape)
        else:
            raise ValueError(f"Unknown mechanism: {mechanism}")
        
        return data + noise
    
    def clip_gradients(self, gradients: np.ndarray, clip_norm: float = 1.0) -> np.ndarray:
        """Clip gradients to bound sensitivity"""
        grad_norm = np.linalg.norm(gradients)
        if grad_norm > clip_norm:
            gradients = gradients * (clip_norm / grad_norm)
        return gradients
    
    def calculate_privacy_spent(self, num_iterations: int) -> float:
        """Calculate total privacy budget spent"""
        # Using advanced composition theorem
        return np.sqrt(2 * num_iterations * np.log(1 / self.delta)) * self.epsilon


class HomomorphicEncryption:
    """Simplified homomorphic encryption for secure aggregation"""
    
    def __init__(self, key_size: int = 2048):
        self.key_size = key_size
        self._generate_keys()
    
    def _generate_keys(self):
        """Generate encryption keys"""
        # Simplified key generation - in production use proper HE library
        self.public_key = np.random.randint(0, 2**self.key_size)
        self.private_key = np.random.randint(0, 2**self.key_size)
    
    def encrypt(self, plaintext: np.ndarray) -> np.ndarray:
        """Encrypt data allowing computation on ciphertext"""
        # Simplified encryption - in production use SEAL or TenSEAL
        noise = np.random.normal(0, 0.1, plaintext.shape)
        return (plaintext + noise) * self.public_key
    
    def decrypt(self, ciphertext: np.ndarray) -> np.ndarray:
        """Decrypt aggregated results"""
        return ciphertext / self.public_key
    
    def add_encrypted(self, cipher1: np.ndarray, cipher2: np.ndarray) -> np.ndarray:
        """Add encrypted values without decryption"""
        return cipher1 + cipher2
    
    def multiply_encrypted(self, ciphertext: np.ndarray, scalar: float) -> np.ndarray:
        """Multiply encrypted value by scalar"""
        return ciphertext * scalar


class SecureAggregator:
    """Secure aggregation protocol for federated learning"""
    
    def __init__(self, num_clients: int, threshold: int):
        self.num_clients = num_clients
        self.threshold = threshold  # Minimum clients for aggregation
        self.client_masks = {}
        self._generate_masks()
    
    def _generate_masks(self):
        """Generate pairwise masks for secure aggregation"""
        for i in range(self.num_clients):
            self.client_masks[i] = {}
            for j in range(self.num_clients):
                if i != j:
                    # Generate shared secret
                    shared_secret = np.random.random()
                    self.client_masks[i][j] = shared_secret
    
    def mask_update(self, client_id: int, update: np.ndarray) -> np.ndarray:
        """Apply secure mask to client update"""
        masked = update.copy()
        
        for other_id, mask in self.client_masks[client_id].items():
            if client_id < other_id:
                masked += mask
            else:
                masked -= mask
        
        return masked
    
    def aggregate_masked_updates(self, masked_updates: List[Tuple[int, np.ndarray]]) -> np.ndarray:
        """Aggregate masked updates securely"""
        if len(masked_updates) < self.threshold:
            raise ValueError(f"Need at least {self.threshold} clients for secure aggregation")
        
        # Sum masked updates - masks cancel out
        aggregated = np.zeros_like(masked_updates[0][1])
        for client_id, update in masked_updates:
            aggregated += update
        
        return aggregated / len(masked_updates)


class FederatedLearningServer:
    """Central server for federated learning coordination"""
    
    def __init__(self, 
                 model_type: ModelType,
                 privacy_mechanism: PrivacyMechanism,
                 privacy_budget: float = 10.0):
        self.model_type = model_type
        self.privacy_mechanism = privacy_mechanism
        self.privacy_budget = privacy_budget
        self.global_model = self._initialize_model()
        self.clients = {}
        self.round_number = 0
        
        # Initialize privacy mechanisms
        self.dp = DifferentialPrivacy(epsilon=privacy_budget)
        self.he = HomomorphicEncryption()
        self.aggregator = SecureAggregator(num_clients=100, threshold=30)
    
    def _initialize_model(self) -> Dict:
        """Initialize global model"""
        if self.model_type == ModelType.LOGISTIC_REGRESSION:
            return {
                "weights": np.random.randn(100),  # 100 features
                "bias": 0.0,
                "version": "1.0.0"
            }
        elif self.model_type == ModelType.NEURAL_NETWORK:
            return {
                "layers": [
                    {"weights": np.random.randn(100, 50), "bias": np.zeros(50)},
                    {"weights": np.random.randn(50, 10), "bias": np.zeros(10)},
                    {"weights": np.random.randn(10, 1), "bias": np.zeros(1)}
                ],
                "version": "1.0.0"
            }
        else:
            return {"version": "1.0.0"}
    
    def register_client(self, client: FederatedClient) -> bool:
        """Register new client for federated learning"""
        if client.client_id not in self.clients:
            self.clients[client.client_id] = client
            return True
        return False
    
    async def start_training_round(self) -> Dict:
        """Start new federated training round"""
        self.round_number += 1
        active_clients = [c for c in self.clients.values() if c.is_active]
        
        # Select subset of clients for this round
        selected_clients = self._select_clients(active_clients)
        
        # Broadcast model to selected clients
        broadcast_data = {
            "round": self.round_number,
            "model": self._encrypt_model(self.global_model),
            "selected_clients": [c.client_id for c in selected_clients]
        }
        
        return broadcast_data
    
    def _select_clients(self, active_clients: List[FederatedClient]) -> List[FederatedClient]:
        """Select clients for training round"""
        # Implement client selection strategy
        min_clients = max(10, int(len(active_clients) * 0.1))
        
        # Weighted selection based on data size
        weights = np.array([c.data_size for c in active_clients])
        probabilities = weights / weights.sum()
        
        selected_indices = np.random.choice(
            len(active_clients),
            size=min(min_clients, len(active_clients)),
            replace=False,
            p=probabilities
        )
        
        return [active_clients[i] for i in selected_indices]
    
    def _encrypt_model(self, model: Dict) -> Dict:
        """Encrypt model for transmission"""
        if self.privacy_mechanism == PrivacyMechanism.HOMOMORPHIC_ENCRYPTION:
            encrypted_model = {}
            for key, value in model.items():
                if isinstance(value, np.ndarray):
                    encrypted_model[key] = self.he.encrypt(value)
                else:
                    encrypted_model[key] = value
            return encrypted_model
        return model
    
    async def aggregate_updates(self, updates: List[ModelUpdate]) -> Dict:
        """Aggregate client updates with privacy preservation"""
        if len(updates) < self.aggregator.threshold:
            return {"status": "insufficient_clients", "required": self.aggregator.threshold}
        
        # Decrypt and validate updates
        validated_updates = []
        for update in updates:
            if self._validate_update(update):
                decrypted = self._decrypt_update(update)
                validated_updates.append(decrypted)
        
        # Apply privacy mechanism
        if self.privacy_mechanism == PrivacyMechanism.DIFFERENTIAL_PRIVACY:
            aggregated = self._dp_aggregate(validated_updates)
        elif self.privacy_mechanism == PrivacyMechanism.SECURE_AGGREGATION:
            aggregated = self._secure_aggregate(validated_updates)
        elif self.privacy_mechanism == PrivacyMechanism.HOMOMORPHIC_ENCRYPTION:
            aggregated = self._he_aggregate(validated_updates)
        else:
            aggregated = self._simple_aggregate(validated_updates)
        
        # Update global model
        self._update_global_model(aggregated)
        
        return {
            "status": "success",
            "round": self.round_number,
            "clients_participated": len(validated_updates),
            "model_version": self.global_model["version"],
            "privacy_spent": self.dp.calculate_privacy_spent(self.round_number)
        }
    
    def _validate_update(self, update: ModelUpdate) -> bool:
        """Validate client update"""
        # Check gradient norm bounds
        if update.gradient_norm > 100:
            return False
        
        # Check client is registered
        if update.client_id not in self.clients:
            return False
        
        # Check privacy budget
        client = self.clients[update.client_id]
        if client.privacy_budget <= 0:
            return False
        
        return True
    
    def _decrypt_update(self, update: ModelUpdate) -> Dict:
        """Decrypt client update"""
        # In production, use proper decryption
        return {
            "client_id": update.client_id,
            "weights": np.frombuffer(update.encrypted_weights),
            "num_samples": update.num_samples
        }
    
    def _dp_aggregate(self, updates: List[Dict]) -> np.ndarray:
        """Aggregate with differential privacy"""
        # Weighted average based on number of samples
        total_samples = sum(u["num_samples"] for u in updates)
        
        aggregated = np.zeros_like(updates[0]["weights"])
        for update in updates:
            weight = update["num_samples"] / total_samples
            # Clip gradients
            clipped = self.dp.clip_gradients(update["weights"])
            aggregated += weight * clipped
        
        # Add privacy noise
        private_aggregated = self.dp.add_noise(aggregated)
        
        return private_aggregated
    
    def _secure_aggregate(self, updates: List[Dict]) -> np.ndarray:
        """Aggregate using secure aggregation protocol"""
        masked_updates = []
        
        for i, update in enumerate(updates):
            masked = self.aggregator.mask_update(i, update["weights"])
            masked_updates.append((i, masked))
        
        return self.aggregator.aggregate_masked_updates(masked_updates)
    
    def _he_aggregate(self, updates: List[Dict]) -> np.ndarray:
        """Aggregate using homomorphic encryption"""
        encrypted_sum = None
        total_samples = 0
        
        for update in updates:
            encrypted_weights = self.he.encrypt(update["weights"])
            if encrypted_sum is None:
                encrypted_sum = encrypted_weights
            else:
                encrypted_sum = self.he.add_encrypted(encrypted_sum, encrypted_weights)
            total_samples += update["num_samples"]
        
        # Decrypt aggregated result
        aggregated = self.he.decrypt(encrypted_sum)
        return aggregated / len(updates)
    
    def _simple_aggregate(self, updates: List[Dict]) -> np.ndarray:
        """Simple federated averaging"""
        total_samples = sum(u["num_samples"] for u in updates)
        
        aggregated = np.zeros_like(updates[0]["weights"])
        for update in updates:
            weight = update["num_samples"] / total_samples
            aggregated += weight * update["weights"]
        
        return aggregated
    
    def _update_global_model(self, aggregated_weights: np.ndarray):
        """Update global model with aggregated weights"""
        if self.model_type == ModelType.LOGISTIC_REGRESSION:
            self.global_model["weights"] = aggregated_weights
        elif self.model_type == ModelType.NEURAL_NETWORK:
            # Update neural network layers
            # Simplified - in production handle layer-wise updates
            pass
        
        # Update version
        version_parts = self.global_model["version"].split(".")
        version_parts[2] = str(int(version_parts[2]) + 1)
        self.global_model["version"] = ".".join(version_parts)
    
    def get_model_metrics(self) -> Dict:
        """Get current model performance metrics"""
        return {
            "round": self.round_number,
            "active_clients": sum(1 for c in self.clients.values() if c.is_active),
            "total_clients": len(self.clients),
            "model_version": self.global_model["version"],
            "privacy_remaining": self.privacy_budget - self.dp.calculate_privacy_spent(self.round_number),
            "aggregation_method": self.privacy_mechanism.value
        }


class FederatedLearningClient:
    """Client-side federated learning implementation"""
    
    def __init__(self, 
                 client_id: str,
                 local_data_size: int,
                 privacy_sensitivity: float = 1.0):
        self.client_id = client_id
        self.local_data_size = local_data_size
        self.privacy_sensitivity = privacy_sensitivity
        self.local_model = None
        self.training_history = []
    
    async def participate_in_round(self, 
                                  global_model: Dict,
                                  local_data: np.ndarray,
                                  local_labels: np.ndarray) -> ModelUpdate:
        """Train on local data and return update"""
        # Update local model with global model
        self.local_model = self._copy_model(global_model)
        
        # Train on local data
        trained_weights = self._train_local_model(local_data, local_labels)
        
        # Calculate update (difference from global model)
        if "weights" in global_model:
            weight_update = trained_weights - global_model["weights"]
        else:
            weight_update = trained_weights
        
        # Calculate gradient norm for validation
        gradient_norm = np.linalg.norm(weight_update)
        
        # Encrypt update
        encrypted_update = self._encrypt_weights(weight_update)
        
        # Create model update
        update = ModelUpdate(
            client_id=self.client_id,
            encrypted_weights=encrypted_update,
            gradient_norm=gradient_norm,
            num_samples=self.local_data_size,
            timestamp=datetime.now(),
            privacy_noise_added=self.privacy_sensitivity > 0
        )
        
        # Record training history
        self.training_history.append({
            "timestamp": datetime.now(),
            "samples_used": self.local_data_size,
            "gradient_norm": gradient_norm
        })
        
        return update
    
    def _copy_model(self, model: Dict) -> Dict:
        """Create local copy of model"""
        import copy
        return copy.deepcopy(model)
    
    def _train_local_model(self, 
                          data: np.ndarray, 
                          labels: np.ndarray,
                          epochs: int = 5) -> np.ndarray:
        """Train model on local data"""
        # Simplified training - in production use proper ML framework
        weights = self.local_model.get("weights", np.random.randn(data.shape[1]))
        
        learning_rate = 0.01
        for epoch in range(epochs):
            # Mini-batch gradient descent
            predictions = data @ weights
            errors = predictions - labels
            gradients = data.T @ errors / len(data)
            
            # Add local privacy noise if needed
            if self.privacy_sensitivity > 0:
                noise = np.random.normal(0, self.privacy_sensitivity, gradients.shape)
                gradients += noise
            
            weights -= learning_rate * gradients
        
        return weights
    
    def _encrypt_weights(self, weights: np.ndarray) -> bytes:
        """Encrypt weights for transmission"""
        # Simple encryption - in production use proper encryption
        return weights.tobytes()
    
    def get_client_stats(self) -> Dict:
        """Get client statistics"""
        return {
            "client_id": self.client_id,
            "data_size": self.local_data_size,
            "rounds_participated": len(self.training_history),
            "average_gradient_norm": np.mean([h["gradient_norm"] for h in self.training_history]) if self.training_history else 0,
            "privacy_sensitivity": self.privacy_sensitivity
        }


class PrivacyPreservingPersonalization:
    """Main system for privacy-preserving personalization"""
    
    def __init__(self):
        self.fl_server = FederatedLearningServer(
            model_type=ModelType.LOGISTIC_REGRESSION,
            privacy_mechanism=PrivacyMechanism.DIFFERENTIAL_PRIVACY,
            privacy_budget=10.0
        )
        self.clients = {}
        self.personalization_models = {}
    
    async def create_personalized_model(self, 
                                      user_segments: List[str],
                                      base_features: Dict) -> Dict:
        """Create personalized model while preserving privacy"""
        # Initialize federated learning for segments
        segment_models = {}
        
        for segment in user_segments:
            # Get clients in segment
            segment_clients = self._get_segment_clients(segment)
            
            # Run federated learning
            if len(segment_clients) >= 10:
                model = await self._train_segment_model(segment, segment_clients)
                segment_models[segment] = model
        
        # Combine segment models for personalization
        personalized_model = self._combine_segment_models(segment_models, base_features)
        
        return {
            "model": personalized_model,
            "segments_used": list(segment_models.keys()),
            "privacy_guarantee": self.fl_server.dp.epsilon,
            "clients_contributed": sum(len(self._get_segment_clients(s)) for s in user_segments)
        }
    
    def _get_segment_clients(self, segment: str) -> List[FederatedClient]:
        """Get clients belonging to segment"""
        # In production, implement proper segmentation
        return [c for c in self.fl_server.clients.values() if c.is_active][:20]
    
    async def _train_segment_model(self, 
                                  segment: str,
                                  clients: List[FederatedClient]) -> Dict:
        """Train model for specific segment"""
        # Register clients
        for client in clients:
            self.fl_server.register_client(client)
        
        # Run training rounds
        for round_num in range(5):
            # Start round
            round_data = await self.fl_server.start_training_round()
            
            # Simulate client updates
            updates = []
            for client_id in round_data["selected_clients"]:
                # In production, clients would train locally
                update = ModelUpdate(
                    client_id=client_id,
                    encrypted_weights=np.random.randn(100).tobytes(),
                    gradient_norm=np.random.uniform(0.1, 1.0),
                    num_samples=np.random.randint(100, 1000),
                    timestamp=datetime.now()
                )
                updates.append(update)
            
            # Aggregate updates
            await self.fl_server.aggregate_updates(updates)
        
        return self.fl_server.global_model
    
    def _combine_segment_models(self, 
                               segment_models: Dict[str, Dict],
                               base_features: Dict) -> Dict:
        """Combine segment models for personalization"""
        if not segment_models:
            return base_features
        
        # Weighted combination based on segment importance
        combined_weights = np.zeros(100)  # Assuming 100 features
        total_weight = 0
        
        for segment, model in segment_models.items():
            if "weights" in model:
                weight = base_features.get(f"{segment}_importance", 1.0)
                combined_weights += weight * model["weights"]
                total_weight += weight
        
        if total_weight > 0:
            combined_weights /= total_weight
        
        return {
            "weights": combined_weights,
            "segments": list(segment_models.keys()),
            "combination_method": "weighted_average"
        }


# Example usage
async def main():
    """Example federated learning setup"""
    # Create privacy-preserving personalization system
    system = PrivacyPreservingPersonalization()
    
    # Define user segments
    user_segments = ["high_value", "frequent_buyers", "new_users"]
    
    # Create base features
    base_features = {
        "high_value_importance": 2.0,
        "frequent_buyers_importance": 1.5,
        "new_users_importance": 1.0
    }
    
    # Create personalized model
    result = await system.create_personalized_model(user_segments, base_features)
    
    print("Privacy-Preserving Personalization Result:")
    print(f"Segments used: {result['segments_used']}")
    print(f"Privacy guarantee (epsilon): {result['privacy_guarantee']}")
    print(f"Clients contributed: {result['clients_contributed']}")
    
    # Get server metrics
    metrics = system.fl_server.get_model_metrics()
    print("\nFederated Learning Metrics:")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    asyncio.run(main())