#!/usr/bin/env python3
"""
V-JEPA 2 Model Implementation using project's actual loading logic
"""
import sys
import os
import cv2
import time
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import torch
import numpy as np
from typing import List, Dict, Any, Optional, Callable

from core.base import BaseModel
from utils.video import (
    get_device, load_vjepa_model, preprocess_video_frames,
    get_processor_for_model, get_transform_for_model, apply_temperature_scaling
)
from utils.sliding_window import SlidingWindowPredictor

class VJEPAModel(BaseModel):
    """V-JEPA 2 model implementation using actual project logic"""

    def __init__(self, model_name: str, checkpoint_path: Optional[str] = None,
                 device: Optional[str] = None, frame_count: int = 32, img_size: int = 224,
                 target_fps: Optional[float] = None, take_last_frames: bool = True,
                 use_sliding_window: bool = False, window_stride: int = 16,
                 save_preprocessed_tensors: bool = False, fill_value=None):
        self.model_name = model_name
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device) if device else get_device()
        self.frame_count = frame_count
        self.img_size = img_size
        self.target_fps = target_fps
        self.take_last_frames = take_last_frames
        self.use_sliding_window = use_sliding_window
        self.window_stride = window_stride
        self.save_preprocessed_tensors = save_preprocessed_tensors
        self.fill_value = fill_value
        
        self.model = None
        self.processor = None
        self.transform = None
        self.sliding_window_predictor = None

        self.attention_history = []
        self.times = []
        
        self.preprocessed_tensors: Dict[str, torch.Tensor] = {}
        self.tensor_save_callback: Optional[Callable[[str, torch.Tensor], None]] = None

    def load(self) -> None:
        """Load V-JEPA model using project's loading logic"""
        try:
            print(f"Loading V-JEPA model: {self.model_name}")
            if self.checkpoint_path:
                print(f"Using checkpoint: {self.checkpoint_path}")

            self.model = load_vjepa_model(
                model_name=self.model_name,
                checkpoint_path=self.checkpoint_path,
                device=self.device
            )

            self.processor = get_processor_for_model(self.model_name)
            self.transform = get_transform_for_model(self.model_name, self.img_size)

            if self.use_sliding_window:
                self.sliding_window_predictor = SlidingWindowPredictor(
                    window_size=self.frame_count,
                    stride=self.window_stride,
                    target_fps=self.target_fps,
                    fill_value=self.fill_value
                )
                print(f"Sliding window predictor initialized (window={self.frame_count}, stride={self.window_stride})")

        except FileNotFoundError as e:
            raise FileNotFoundError(f"Model file not found: {e}")
        except RuntimeError as e:
            raise RuntimeError(f"Failed to load V-JEPA model: {e}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error loading V-JEPA model: {e}")

    def set_tensor_save_callback(self, callback: Optional[Callable[[str, torch.Tensor], None]]) -> None:
        """Set callback function to handle saving of preprocessed tensors"""
        self.tensor_save_callback = callback

    def enable_tensor_saving(self, enable: bool = True) -> None:
        """Enable or disable tensor saving during prediction"""
        self.save_preprocessed_tensors = enable
        if not enable:
            self.preprocessed_tensors.clear()

    def get_saved_tensor(self, video_path: str) -> Optional[torch.Tensor]:
        """Get saved preprocessed tensor for a video path"""
        return self.preprocessed_tensors.get(video_path)

    def get_all_saved_tensors(self) -> Dict[str, torch.Tensor]:
        """Get all saved preprocessed tensors"""
        return self.preprocessed_tensors.copy()

    def clear_saved_tensors(self) -> None:
        """Clear all saved tensors from memory"""
        self.preprocessed_tensors.clear()

    def predict(self, video_path: str, real_time: bool = False, show_time: bool = False, attn_rollout: bool = False, grad_rollout: bool = False) -> np.ndarray:
        """Predict frame-level probabilities for single video"""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        try:
            if self.use_sliding_window and self.sliding_window_predictor is not None:
                return self._predict_sliding_window(video_path, real_time=real_time, show_time=show_time, attn_rollout=attn_rollout, grad_rollout=grad_rollout)
            else:
                return self._predict_regular(video_path)

        except FileNotFoundError as e:
            raise FileNotFoundError(f"Video file not found: {e}")
        except Exception as e:
            raise RuntimeError(f"Prediction failed for {video_path}: {e}")

    def _predict_regular(self, video_path: str) -> np.ndarray:
        """Regular prediction using fixed window size"""
        video_tensor = preprocess_video_frames(
            video_path=video_path,
            target_frames=self.frame_count,
            target_size=(self.img_size, self.img_size),
            processor=self.processor,
            transform=self.transform,
            model_name=self.model_name,
            target_fps=self.target_fps,
            take_last_frames=self.take_last_frames
        )

        if self.save_preprocessed_tensors:
            self.preprocessed_tensors[video_path] = video_tensor.clone().cpu()
            if self.tensor_save_callback:
                self.tensor_save_callback(video_path, video_tensor.clone().cpu())
        
        if video_tensor.dim() == 4:
            video_tensor = video_tensor.unsqueeze(0)
        video_tensor = video_tensor.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(video_tensor)
            outputs_scaled = apply_temperature_scaling(outputs, temperature=2.0)
            probs = torch.softmax(outputs_scaled, dim=1)[:, 1].cpu().numpy()
            frame_probs = np.repeat(probs[0], self.frame_count)
            
        return frame_probs

    def _predict_sliding_window(self, video_path: str, real_time: bool = False, show_time: bool = False, grad_rollout: bool = False, attn_rollout: bool = False) -> np.ndarray:
        """Sliding window prediction for frame-level results"""

        first_tensor_saved = False
        current_display_frame = None

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        num_windows = max(1, (total_frames - self.frame_count) // self.window_stride + 1)
        pbar = tqdm(desc="Inference Progress", unit="win")

        def preprocess_fn(frames_array):
            nonlocal first_tensor_saved, current_display_frame

            if real_time:
                raw_img = frames_array[-1].copy()
                current_display_frame = cv2.cvtColor(raw_img, cv2.COLOR_RGB2BGR)

            if self.processor:
                try:
                    if hasattr(self.processor, '__call__'):
                        inputs = self.processor(videos=frames_array, return_tensors="pt")
                        if 'pixel_values_videos' in inputs:
                            video_tensor = inputs['pixel_values_videos'].squeeze(0)
                        elif 'pixel_values' in inputs:
                            video_tensor = inputs['pixel_values'].squeeze(0)
                        else:
                            video_tensor = list(inputs.values())[0].squeeze(0)
                    else:
                        raise ValueError("Invalid processor")
                except Exception as e:
                    print(f"Warning: Processor failed ({e}), using manual transform")
                    video_tensor = self._manual_transform_frames(frames_array)
            else:
                video_tensor = self._manual_transform_frames(frames_array)

            if self.save_preprocessed_tensors and not first_tensor_saved:
                self.preprocessed_tensors[video_path] = video_tensor.clone().cpu()
                first_tensor_saved = True
                if self.tensor_save_callback:
                    self.tensor_save_callback(video_path, video_tensor.clone().cpu())

            return video_tensor

        def model_predict_fn(processed_frames):
            nonlocal current_display_frame

            if processed_frames.dim() == 4:
                processed_frames = processed_frames.unsqueeze(0)
            processed_frames = processed_frames.to(self.device)

            # ---------------------------------------------------------
            # OPTIMISATION MÉMOIRE MAXIMALE : BACKWARD INTERCEPTION
            # ---------------------------------------------------------
            nb_layer_backbone = 24
            
            # Stocke l'attention brute temporairement (sur RAM/CPU)
            raw_attentions_cpu = [None] * nb_layer_backbone
            
            # Stocke le produit final [Attention * Gradient] (sur RAM/CPU)
            processed_components_cpu = [None] * nb_layer_backbone
            
            layer_counter = 0
            
            # Booléen global pour savoir si on doit générer une carte
            do_heatmap = grad_rollout or attn_rollout
            
            # --- 1. BRANCHEMENT DE L'ESPION DYNAMIQUE ---
            if do_heatmap:
                import torch.nn.functional as F
                import math
                
                original_sdpa = F.scaled_dot_product_attention

                def hooked_sdpa(*args, **kwargs):
                    nonlocal layer_counter
                    
                    query = args[0]
                    key = args[1]
                    value = args[2]
                    attn_mask = kwargs.get('attn_mask', args[3] if len(args) > 3 else None)
                    scale = kwargs.get('scale', None)

                    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
                    attn_weight = query @ key.transpose(-2, -1) * scale_factor
                    
                    if attn_mask is not None:
                        if attn_mask.dtype == torch.bool:
                            attn_weight.masked_fill_(attn_mask.logical_not(), float("-inf"))
                        else:
                            attn_weight += attn_mask
                            
                    attn_weight = torch.softmax(attn_weight, dim=-1)
                    
                    if attn_weight.dim() == 4 and attn_weight.size(-1) > 100:
                        if layer_counter < nb_layer_backbone:
                            
                            # On écrase l'attention tout de suite pour sauver la RAM
                            A_cpu = attn_weight.mean(dim=1).detach().cpu()
                            
                            if grad_rollout:
                                # VÉRIFICATION : Est-ce que cette couche a droit au gradient ?
                                if attn_weight.requires_grad:
                                    attn_weight.retain_grad()
                                    raw_attentions_cpu[layer_counter] = A_cpu
                                    
                                    def get_grad_hook(idx):
                                        def hook(grad):
                                            with torch.no_grad():
                                                mean_grad = grad.mean(dim=1)
                                                A_gpu = raw_attentions_cpu[idx].to(grad.device)
                                                positive_gradients = torch.relu(mean_grad)
                                                weighted_A = A_gpu * positive_gradients
                                                processed_components_cpu[idx] = weighted_A.cpu()
                                                raw_attentions_cpu[idx] = None
                                        return hook
                                        
                                    attn_weight.register_hook(get_grad_hook(layer_counter))
                                else:
                                    # COUCHE GELÉE : Pas de gradient. L'attention pure devient le résultat final !
                                    processed_components_cpu[layer_counter] = A_cpu
                            elif attn_rollout:
                                # PURE ATTENTION : Pas de gradient, on range la matrice prête pour la fin
                                processed_components_cpu[layer_counter] = A_cpu

                            layer_counter += 1

                    return attn_weight @ value

                F.scaled_dot_product_attention = hooked_sdpa

            start_time = time.time()

            try:
                # --- 2. PRÉDICTION ---
                # On active les gradients SEULEMENT si Grad-Rollout est demandé
                with torch.set_grad_enabled(grad_rollout):
                    self.model.zero_grad()
                    
                    if grad_rollout:
                        # 1. LA CLÉ DE LA VRAM : On refuse le gradient sur les pixels
                        processed_frames.requires_grad_(False)
                        
                        # 2. On gèle tout le modèle par défaut
                        for param in self.model.parameters():
                            param.requires_grad = False
                            
                        # 3. On ne dégèle que la 2ème moitié (Matrices > 2D)
                        grosses_matrices = [p for p in self.model.parameters() if p.dim() >= 2]
                        moitie = len(grosses_matrices) // 2
                        
                        for param in grosses_matrices[moitie:]:
                            param.requires_grad = True

                        # 4. On s'assure que la sortie est bien dégelée
                        if hasattr(self.model, 'predictor'):
                            for param in self.model.predictor.parameters():
                                param.requires_grad = True
                        elif hasattr(self.model, 'head'):
                            for param in self.model.head.parameters():
                                param.requires_grad = True

                    # Lancement du Forward Pass allégé
                    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                        outputs = self.model(processed_frames)
                        outputs_scaled = apply_temperature_scaling(outputs, temperature=2.0)
                        
                        probs = torch.softmax(outputs_scaled, dim=1)
                        prediction_score = probs[0, 1].item()
                        
                        target_score = outputs_scaled[0, 1]

                    # --- 3. LE DÉCLENCHEUR BACKWARD ---
                    if grad_rollout and layer_counter > 0:
                        target_score.backward(retain_graph=False)

                    
                    # --- 4. CALCUL DU ROLLOUT (Maintenant qu'on a les matrices prêtes) ---
                    if do_heatmap and layer_counter > 0:
                        rollout = None
                        
                        with torch.no_grad(): # Pas besoin de gradient pour le Rollout matriciel
                            for i in range(nb_layer_backbone):
                                
                                final_A_cpu = processed_components_cpu[i]
                                
                                if final_A_cpu is not None:
                                    # On remonte la matrice finale sur GPU pour faire le matmul très vite
                                    A_gpu = final_A_cpu.to(self.device)

                                    I = torch.eye(A_gpu.size(-1), device=A_gpu.device)
                                    A_final = A_gpu + I
                                    A_final = A_final / A_final.sum(dim=-1, keepdim=True)
                                    
                                    if rollout is None:
                                        rollout = A_final
                                    else:
                                        rollout = torch.matmul(A_final, rollout)
                                        
                                    # Nettoyage instantané du composant
                                    processed_components_cpu[i] = None

                        if rollout is not None:
                            temporal_blocks = []
                            for t in range(8):
                                start_idx = t * 256
                                end_idx = (t + 1) * 256
                                
                                block_t = rollout[..., start_idx:end_idx, start_idx:end_idx]
                                importance_t = block_t.mean(dim=-1)
                                
                                while importance_t.dim() > 1:
                                    importance_t = importance_t.mean(dim=0)
                                    
                                temporal_blocks.append(importance_t)
                                
                            final_heatmap_1d = torch.cat(temporal_blocks, dim=0).cpu().numpy()
                            self.attention_history.append(final_heatmap_1d)

                    end_time = time.time()
                    pbar.update(1)
                    
                    if show_time:
                        inference_time = end_time - start_time
                        self.times.append(inference_time)

                # --- AFFICHAGE DE LA JAUGE OPENCV ---
                if real_time and current_display_frame is not None:
                    scale_factor = 3
                    h, w = current_display_frame.shape[:2]
                    new_dim = (w * scale_factor, h * scale_factor)
                    big_frame = cv2.resize(current_display_frame, new_dim, interpolation=cv2.INTER_LINEAR)

                    gauge_w = 30
                    gauge_h = int(new_dim[1] * 0.6)
                    margin_right = 50
                    margin_bottom = 50

                    x_start = new_dim[0] - margin_right - gauge_w
                    y_bottom = new_dim[1] - margin_bottom
                    y_top = y_bottom - gauge_h

                    cv2.rectangle(big_frame, (x_start, y_top), (x_start + gauge_w, y_bottom), (40, 40, 40), -1)

                    score = prediction_score
                    if score < 0.5:
                        ratio = score / 0.5; r = int(255 * ratio); g = 255; b = 0
                    else:
                        ratio = (score - 0.5) / 0.5; r = 255; g = int(255 * (1 - ratio)); b = 0

                    gauge_color = (b, g, r)
                    fill_height = int(gauge_h * score)
                    y_fill_start = y_bottom - fill_height

                    cv2.rectangle(big_frame, (x_start, y_fill_start), (x_start + gauge_w, y_bottom), gauge_color, -1)
                    cv2.rectangle(big_frame, (x_start, y_top), (x_start + gauge_w, y_bottom), (200, 200, 200), 2)

                    y_50 = y_bottom - int(gauge_h * 0.5)
                    cv2.line(big_frame, (x_start - 5, y_50), (x_start + gauge_w + 5, y_50), (100, 100, 100), 1)
                    y_80 = y_bottom - int(gauge_h * 0.8)
                    cv2.line(big_frame, (x_start - 5, y_80), (x_start + gauge_w + 5, y_80), (100, 100, 100), 1)

                    text_pct = f"{int(score * 100)}%"
                    cv2.putText(big_frame, text_pct, (x_start - 60, y_fill_start + 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, gauge_color, 2)

                    cv2.imshow('BADAS Real-Time Inference', big_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("User interrupted")

                return prediction_score

            finally:
                # --- 5. DÉBRANCHEMENT ET NETTOYAGE ---
                if do_heatmap:
                    F.scaled_dot_product_attention = original_sdpa
                    raw_attentions_cpu.clear()
                    processed_components_cpu.clear()
                    if grad_rollout:
                        self.model.zero_grad()
                    torch.cuda.empty_cache()

        # --- EXÉCUTION DE LA FENÊTRE GLISSANTE ---
        try:
            results = self.sliding_window_predictor.predict_sliding_windows(
                video_path=video_path,
                model_predict_fn=model_predict_fn,
                preprocess_fn=preprocess_fn,
                return_per_frame=True
            )
        finally:
            pbar.close()
            if real_time:
                cv2.destroyAllWindows()

        return results['per_frame']

    def _manual_transform_frames(self, frames_array: np.ndarray) -> torch.Tensor:
        """Manual transformation for frames array"""
        if self.transform:
            transformed_frames = [self.transform(image=f)["image"] for f in frames_array]
            return torch.stack(transformed_frames)
        else:
            frames_tensor = torch.from_numpy(frames_array.transpose(0, 3, 1, 2)).float() / 255.0
            return frames_tensor

    def predict_batch(self, video_paths: List[str]) -> List[np.ndarray]:
        """Predict frame-level probabilities for multiple videos"""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        results = []
        for video_path in video_paths:
            try:
                pred = self.predict(video_path)
                results.append(pred)
            except Exception as e:
                raise RuntimeError(f"Batch prediction failed at {video_path}: {e}")

        return results

    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata"""
        info = {
            "name": "V-JEPA 2",
            "model_name": self.model_name,
            "checkpoint_path": self.checkpoint_path,
            "device": str(self.device),
            "frame_count": self.frame_count,
            "img_size": self.img_size,
            "target_fps": self.target_fps,
            "take_last_frames": self.take_last_frames,
            "use_sliding_window": self.use_sliding_window,
            "window_stride": self.window_stride,
            "has_model": self.model is not None,
            "has_processor": self.processor is not None,
            "has_sliding_window_predictor": self.sliding_window_predictor is not None,
            "version": "2.1"
        }

        if self.sliding_window_predictor:
            info["sliding_window_config"] = self.sliding_window_predictor.get_config()

        return info
