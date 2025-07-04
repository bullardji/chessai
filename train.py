# Install required packages
#!pip install pufferlib torch numpy python-chess gymnasium

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import chess
import gymnasium as gym
from gymnasium import spaces
import pufferlib
import pufferlib.utils
import pufferlib.vector

from typing import Dict, Optional, Tuple, Any
import time
from collections import deque

#---------------------------
# 1) Chess Environment for Gymnasium
#---------------------------

class ChessEnv(gym.Env):
    """Chess environment compatible with Gymnasium and PufferLib"""
    
    def __init__(self):
        super().__init__()
        self.board = None
        self.move_count = 0
        self._build_move_mappings()
        
        # Define observation and action spaces
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(18, 8, 8), dtype=np.float32
        )
        self.action_space = spaces.Discrete(len(self.all_moves))
        
    def _build_move_mappings(self):
        """Build all possible chess moves and their mappings"""
        self.all_moves = []
        for f in chess.SQUARES:
            for t in chess.SQUARES:
                self.all_moves.append(chess.Move(f, t))
                fr, tr = chess.square_rank(f), chess.square_rank(t)
                # Add promotion moves
                if (fr == 6 and tr == 7) or (fr == 1 and tr == 0):
                    for promo in (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT):
                        self.all_moves.append(chess.Move(f, t, promotion=promo))
        
        # Remove duplicates
        seen = set()
        unique_moves = []
        for m in self.all_moves:
            if m.uci() not in seen:
                seen.add(m.uci())
                unique_moves.append(m)
        self.all_moves = unique_moves
        self.move_to_idx = {m.uci(): i for i, m in enumerate(self.all_moves)}
        self.idx_to_move = {i: m for i, m in enumerate(self.all_moves)}
        
    def reset(self, seed=None, options=None):
        """Reset the chess board to initial position"""
        super().reset(seed=seed)
        self.board = chess.Board()
        self.move_count = 0
        self.episode_reward = 0
        return self._get_obs(), {}
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one move in the chess game"""
        # Get legal actions
        legal_actions = self._get_legal_actions()
        
        # If illegal action, pick a random legal move
        if action not in legal_actions:
            if legal_actions:
                action = np.random.choice(legal_actions)
            else:
                # Should never happen
                return self._get_obs(), -1.0, True, False, {"illegal_move": True}
        
        # Make the move
        move = self.all_moves[action]
        self.board.push(move)
        self.move_count += 1
        
        # Check if game is over
        terminated = self.board.is_game_over()
        truncated = False  # Chess games don't truncate
        reward = 0.0
        
        if terminated:
            result = self.board.result()
            if result == "1-0":
                # White wins
                reward = 1.0 if self.move_count % 2 == 1 else -1.0
            elif result == "0-1":
                # Black wins
                reward = -1.0 if self.move_count % 2 == 1 else 1.0
            else:
                # Draw
                reward = 0.0
            
            self.episode_reward = reward
        
        info = {
            "legal_actions": self._get_legal_actions(),
            "move_count": self.move_count,
            "board_fen": self.board.fen()
        }
        
        # Add episode info when done
        if terminated:
            info["episode"] = {
                "r": self.episode_reward,
                "l": self.move_count,
                "t": time.time()
            }
        
        return self._get_obs(), reward, terminated, truncated, info
    
    def _get_obs(self) -> np.ndarray:
        """Convert board state to observation tensor"""
        planes = np.zeros((18, 8, 8), dtype=np.float32)
        
        # Piece planes (12 planes: 6 piece types × 2 colors)
        for sq, pc in self.board.piece_map().items():
            r, c = 7 - chess.square_rank(sq), chess.square_file(sq)
            idx = (pc.piece_type - 1) + (0 if pc.color == chess.WHITE else 6)
            planes[idx, r, c] = 1.0
        
        # Castling rights (4 planes)
        planes[12, :, :] = float(self.board.has_kingside_castling_rights(chess.WHITE))
        planes[13, :, :] = float(self.board.has_queenside_castling_rights(chess.WHITE))
        planes[14, :, :] = float(self.board.has_kingside_castling_rights(chess.BLACK))
        planes[15, :, :] = float(self.board.has_queenside_castling_rights(chess.BLACK))
        
        # En passant square (1 plane)
        if self.board.ep_square is not None:
            r, c = 7 - chess.square_rank(self.board.ep_square), chess.square_file(self.board.ep_square)
            planes[16, r, c] = 1.0
        
        # Current player (1 plane)
        planes[17, :, :] = float(self.board.turn)
        
        return planes
    
    def _get_legal_actions(self) -> list:
        """Get list of legal action indices"""
        return [self.move_to_idx[m.uci()] for m in self.board.legal_moves]
    
    def render(self):
        """Render the chess board"""
        return str(self.board)

#---------------------------
# 2) Model Components
#---------------------------

class RelPosSelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float = 0.1):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.dh = dim // heads
        self.scale = self.dh ** -0.5
        
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
        # Relative position biases
        self.rel_bias = nn.Parameter(torch.zeros(15, 15, heads))
        nn.init.xavier_uniform_(self.rel_bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.size()
        
        # Generate Q, K, V
        qkv = self.to_qkv(x).view(B, L, 3, self.heads, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Compute attention scores
        scores = (q @ k.transpose(-2, -1)) * self.scale
        
        # Add relative position bias
        device = x.device
        rows = torch.arange(8, device=device)
        cols = torch.arange(8, device=device)
        ri = rows.unsqueeze(1).expand(8, 8).reshape(-1)
        ci = cols.expand(8, 8).reshape(-1)
        
        dr = (ri.unsqueeze(1) - ri.unsqueeze(0)).clamp(-7, 7) + 7
        dc = (ci.unsqueeze(1) - ci.unsqueeze(0)).clamp(-7, 7) + 7
        
        bias = self.rel_bias[dr.long(), dc.long()]
        bias = bias.permute(2, 0, 1).unsqueeze(0)
        scores = scores + bias
        
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = attn @ v
        out = out.transpose(1, 2).reshape(B, L, C)
        
        return self.to_out(out)

class EncoderLayer(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_mult: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = RelPosSelfAttention(dim, heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_mult)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_mult), dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

#---------------------------
# 3) Chessformer Policy
#---------------------------

class ChessformerPolicy(nn.Module):
    def __init__(self, observation_shape, action_space_size):
        super().__init__()
        
        # Model hyperparameters
        self.planes = 18
        self.dim = 256
        self.heads = 8
        self.layers = 6
        self.hidden = 256
        self.dropout = 0.1
        self.num_actions = action_space_size
        
        # Input embedding
        self.to_emb = nn.Linear(self.planes, self.dim)
        
        # Transformer encoder layers
        self.encoder = nn.ModuleList([
            EncoderLayer(self.dim, self.heads, dropout=self.dropout)
            for _ in range(self.layers)
        ])
        
        # JEPA predictor
        self.jepa_pred = nn.Sequential(
            nn.Linear(self.dim, self.hidden),
            nn.GELU(),
            nn.Linear(self.hidden, self.dim)
        )
        
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.GELU(),
            nn.Linear(self.dim, self.num_actions)
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(self.dim, self.dim // 2),
            nn.GELU(),
            nn.Linear(self.dim // 2, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning policy logits and value"""
        B = obs.shape[0]
        
        # Reshape observation to (B, 64, planes)
        x = obs.view(B, self.planes, 64).permute(0, 2, 1)
        
        # Embed input
        h = self.to_emb(x)
        
        # Pass through encoder layers
        for layer in self.encoder:
            h = layer(h)
        
        # Global average pooling
        z = h.mean(dim=1)
        
        # Get policy logits and value
        logits = self.policy_head(z)
        value = self.value_head(z).squeeze(-1)
        
        return logits, value
    
    def compute_jepa_loss(self, obs_batch: torch.Tensor, next_obs_batch: torch.Tensor) -> torch.Tensor:
        """Compute JEPA loss for consecutive observations"""
        B = obs_batch.shape[0]
        
        # Current observation encoding
        x = obs_batch.view(B, self.planes, 64).permute(0, 2, 1)
        h = self.to_emb(x)
        for layer in self.encoder:
            h = layer(h)
        z = h.mean(dim=1)
        
        # Predict next embedding
        z_pred = self.jepa_pred(z)
        
        # Next observation encoding (stop gradient)
        with torch.no_grad():
            x_next = next_obs_batch.view(B, self.planes, 64).permute(0, 2, 1)
            h_next = self.to_emb(x_next)
            for layer in self.encoder:
                h_next = layer(h_next)
            z_next = h_next.mean(dim=1)
        
        # JEPA loss
        jepa_loss = (1 - F.cosine_similarity(z_pred, z_next, dim=-1)).mean()
        
        return jepa_loss

#---------------------------
# 4) Dr. GRPO Training
#---------------------------

class DrGRPOTrainer:
    def __init__(self, 
                 env_creator,
                 num_envs=128,
                 device='cuda',
                 lr=3e-4,
                 clip_range=0.2,
                 value_coef=0.5,
                 entropy_coef=0.01,
                 jepa_coef=1.0,
                 max_grad_norm=0.5):
        
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.num_envs = num_envs
        self.lr = lr
        self.clip_range = clip_range
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.jepa_coef = jepa_coef
        self.max_grad_norm = max_grad_norm
        
        # Create environments
        print(f"Creating {num_envs} environments...")
        self.vec_env = pufferlib.vector.make(
            env_creator,
            num_envs=num_envs,
            backend=pufferlib.vector.Multiprocessing  # or Serial for debugging
        )
        
        # Get env specs from a single env
        single_env = env_creator()
        obs_shape = single_env.observation_space.shape
        act_shape = single_env.action_space.n
        single_env.close()
        
        # Create policy
        print("Creating policy network...")
        self.policy = ChessformerPolicy(obs_shape, act_shape).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        
        # Print model info
        total_params = sum(p.numel() for p in self.policy.parameters())
        print(f"Total parameters: {total_params:,}")
        
    def compute_dr_grpo_loss(self, 
                            obs_batch: torch.Tensor,
                            action_batch: torch.Tensor,
                            old_log_prob_batch: torch.Tensor,
                            return_batch: torch.Tensor,
                            advantage_batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute Dr. GRPO loss"""
        
        # Get current policy outputs
        logits, values = self.policy(obs_batch)
        
        # Compute log probabilities
        log_probs = F.log_softmax(logits, dim=-1)
        action_log_probs = log_probs.gather(1, action_batch.unsqueeze(-1)).squeeze(-1)
        
        # Compute probability ratio
        ratio = torch.exp(action_log_probs - old_log_prob_batch)
        
        # Dr. GRPO policy loss (clipped)
        pg_loss1 = -advantage_batch * ratio
        pg_loss2 = -advantage_batch * torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
        policy_loss = torch.max(pg_loss1, pg_loss2).mean()
        
        # Value loss
        value_loss = F.mse_loss(values, return_batch)
        
        # Entropy
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1).mean()
        
        # Total loss
        total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
        
        # Additional metrics
        with torch.no_grad():
            approx_kl = ((ratio - 1) - (action_log_probs - old_log_prob_batch)).mean()
            clipfrac = ((ratio - 1.0).abs() > self.clip_range).float().mean()
        
        return {
            'total_loss': total_loss,
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'entropy': entropy,
            'approx_kl': approx_kl,
            'clipfrac': clipfrac
        }
    
    def train(self, total_timesteps=10_000_000, 
              rollout_length=256,
              num_epochs=4,
              batch_size=512,
              log_interval=10):
        """Main training loop"""
        
        # Initialize
        obs = torch.tensor(self.vec_env.reset()[0], device=self.device, dtype=torch.float32)
        num_updates = total_timesteps // (rollout_length * self.num_envs)
        
        # Metrics
        episode_returns = deque(maxlen=100)
        episode_lengths = deque(maxlen=100)
        win_rate = deque(maxlen=100)
        
        print(f"Starting training for {total_timesteps:,} timesteps...")
        print(f"Updates: {num_updates:,}, Rollout length: {rollout_length}")
        
        start_time = time.time()
        
        for update in range(num_updates):
            # Collect rollout
            rollout_obs = []
            rollout_actions = []
            rollout_log_probs = []
            rollout_rewards = []
            rollout_dones = []
            rollout_values = []
            
            with torch.no_grad():
                for step in range(rollout_length):
                    # Store observation
                    rollout_obs.append(obs)
                    
                    # Get policy outputs
                    logits, values = self.policy(obs)
                    
                    # Sample actions
                    probs = F.softmax(logits, dim=-1)
                    dist = torch.distributions.Categorical(probs)
                    actions = dist.sample()
                    log_probs = dist.log_prob(actions)
                    
                    # Store data
                    rollout_actions.append(actions)
                    rollout_log_probs.append(log_probs)
                    rollout_values.append(values)
                    
                    # Step environment
                    obs_np, rewards, dones, truncs, infos = self.vec_env.step(actions.cpu().numpy())
                    
                    obs = torch.tensor(obs_np, device=self.device, dtype=torch.float32)
                    rewards = torch.tensor(rewards, device=self.device, dtype=torch.float32)
                    dones = torch.tensor(dones, device=self.device, dtype=torch.float32)
                    
                    rollout_rewards.append(rewards)
                    rollout_dones.append(dones)
                    
                    # Track episodes
                    for i, (done, info) in enumerate(zip(dones, infos)):
                        if done and 'episode' in info:
                            ep_return = info['episode']['r']
                            ep_length = info['episode']['l']
                            episode_returns.append(ep_return)
                            episode_lengths.append(ep_length)
                            win_rate.append(1.0 if ep_return > 0 else 0.0)
            
            # Convert to tensors
            rollout_obs = torch.stack(rollout_obs)
            rollout_actions = torch.stack(rollout_actions)
            rollout_log_probs = torch.stack(rollout_log_probs)
            rollout_rewards = torch.stack(rollout_rewards)
            rollout_dones = torch.stack(rollout_dones)
            rollout_values = torch.stack(rollout_values)
            
            # Compute returns and advantages (Dr. GRPO style)
            returns = torch.zeros_like(rollout_rewards)
            current_return = torch.zeros(self.num_envs, device=self.device)
            
            for t in reversed(range(rollout_length)):
                # Reset return when episode ends
                current_return = rollout_rewards[t] + current_return * (1 - rollout_dones[t])
                returns[t] = current_return
            
            # Dr. GRPO advantages: unbiased Monte-Carlo baseline
            advantages = returns - returns.mean()
            
            # Flatten for training
            obs_batch = rollout_obs.reshape(-1, *rollout_obs.shape[2:])
            action_batch = rollout_actions.reshape(-1)
            log_prob_batch = rollout_log_probs.reshape(-1)
            return_batch = returns.reshape(-1)
            advantage_batch = advantages.reshape(-1)
            
            # Training epochs
            for epoch in range(num_epochs):
                # Create random batches
                indices = torch.randperm(obs_batch.shape[0], device=self.device)
                
                total_policy_loss = 0
                total_value_loss = 0
                total_entropy = 0
                total_approx_kl = 0
                total_clipfrac = 0
                total_jepa_loss = 0
                num_batches = 0
                
                for start in range(0, obs_batch.shape[0], batch_size):
                    end = min(start + batch_size, obs_batch.shape[0])
                    batch_indices = indices[start:end]
                    
                    # Get batch
                    mb_obs = obs_batch[batch_indices]
                    mb_actions = action_batch[batch_indices]
                    mb_log_probs = log_prob_batch[batch_indices]
                    mb_returns = return_batch[batch_indices]
                    mb_advantages = advantage_batch[batch_indices]
                    
                    # Compute loss
                    losses = self.compute_dr_grpo_loss(
                        mb_obs, mb_actions, mb_log_probs, mb_returns, mb_advantages
                    )
                    
                    # JEPA loss (find consecutive observations)
                    jepa_loss = torch.tensor(0.0, device=self.device)
                    jepa_count = 0
                    
                    # Look for consecutive indices in the batch
                    for i in range(len(batch_indices) - 1):
                        idx1, idx2 = batch_indices[i], batch_indices[i+1]
                        # Check if they're consecutive and not across episode boundary
                        if idx2 == idx1 + 1 and idx1 % rollout_length < rollout_length - 1:
                            if not rollout_dones.reshape(-1)[idx1]:
                                if jepa_count == 0:
                                    jepa_obs1 = mb_obs[i:i+1]
                                    jepa_obs2 = mb_obs[i+1:i+2]
                                else:
                                    jepa_obs1 = torch.cat([jepa_obs1, mb_obs[i:i+1]], dim=0)
                                    jepa_obs2 = torch.cat([jepa_obs2, mb_obs[i+1:i+2]], dim=0)
                                jepa_count += 1
                    
                    if jepa_count > 0:
                        jepa_loss = self.policy.compute_jepa_loss(jepa_obs1, jepa_obs2)
                    
                    # Total loss
                    total_loss = losses['total_loss'] + self.jepa_coef * jepa_loss
                    
                    # Optimize
                    self.optimizer.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    
                    # Accumulate metrics
                    total_policy_loss += losses['policy_loss'].item()
                    total_value_loss += losses['value_loss'].item()
                    total_entropy += losses['entropy'].item()
                    total_approx_kl += losses['approx_kl'].item()
                    total_clipfrac += losses['clipfrac'].item()
                    total_jepa_loss += jepa_loss.item()
                    num_batches += 1
            
            # Logging
            if update % log_interval == 0 and update > 0:
                elapsed = time.time() - start_time
                fps = (update + 1) * rollout_length * self.num_envs / elapsed
                
                print(f"\n{'='*60}")
                print(f"Update {update}/{num_updates}")
                print(f"Timesteps: {(update + 1) * rollout_length * self.num_envs:,}")
                print(f"FPS: {fps:.0f}")
                print(f"{'='*60}")
                
                if episode_returns:
                    print(f"Episode Return: {np.mean(episode_returns):.2f} ± {np.std(episode_returns):.2f}")
                    print(f"Episode Length: {np.mean(episode_lengths):.1f}")
                    print(f"Win Rate: {np.mean(win_rate):.2%}")
                
                print(f"Policy Loss: {total_policy_loss/num_batches:.4f}")
                print(f"Value Loss: {total_value_loss/num_batches:.4f}")
                print(f"Entropy: {total_entropy/num_batches:.4f}")
                print(f"JEPA Loss: {total_jepa_loss/num_batches:.4f}")
                print(f"Approx KL: {total_approx_kl/num_batches:.4f}")
                print(f"Clip Fraction: {total_clipfrac/num_batches:.3f}")
                
        print("\nTraining complete!")
        self.vec_env.close()

#---------------------------
# 5) Main Entry Point
#---------------------------

def main():
    # Create trainer
    trainer = DrGRPOTrainer(
        env_creator=lambda: ChessEnv(),
        num_envs=64,  # Adjust based on your GPU memory
        device='cuda',
        lr=3e-4,
        clip_range=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        jepa_coef=1.0,
        max_grad_norm=0.5
    )
    
    # Train
    trainer.train(
        total_timesteps=10_000_000,
        rollout_length=256,
        num_epochs=4,
        batch_size=512,
        log_interval=10
    )

if __name__ == "__main__":
    main()
