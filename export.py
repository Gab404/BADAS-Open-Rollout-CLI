import math
import numpy as np
import cv2
from tqdm import tqdm

def apply_attention_heatmap(frame, rollout_1d, temporal_index, score, previous_heatmap=None, smoothing_factor=0.85):
    """
    Overlay d'une heatmap avec opacité FIXE.
    La couleur maximale de la tache évolue avec le score : 
    Faible = Vert, Moyen = Jaune, Élevé = Rouge.
    """
    try:
        # Configuration des dimensions
        num_tokens = rollout_1d.shape[0]
        time_steps = 8 
        spatial_tokens = num_tokens // time_steps
        grid_size = int(math.sqrt(spatial_tokens))
        
        # Reconstitution du cube 3D et sélection de la tranche
        patch_importance_3d = rollout_1d.reshape(time_steps, grid_size, grid_size)
        heatmap_2d = patch_importance_3d[temporal_index, :, :].copy()

        # 1. LISSAGE TEMPOREL (Anti-clignotement)
        if previous_heatmap is not None:
            heatmap_2d = (previous_heatmap * smoothing_factor) + (heatmap_2d * (1.0 - smoothing_factor))
        current_smoothed_heatmap = heatmap_2d.copy()

        # 2. NORMALISATION (de 0.0 à 1.0)
        heatmap_min, heatmap_max = np.min(heatmap_2d), np.max(heatmap_2d)
        if heatmap_max - heatmap_min == 0:
            return frame, current_smoothed_heatmap 
            
        heatmap_norm = (heatmap_2d - heatmap_min) / (heatmap_max - heatmap_min)
        
        # Accentuation des zones de forte attention
        heatmap_norm = heatmap_norm ** 4 

        # 3. LE SECRET : LE SEUIL DE COULEUR MAXIMAL BARRÉ PAR LE SCORE
        # Au lieu de multiplier jusqu'à 255, on calcule le plafond.
        # Score 0.0 -> max_color = 130 (Vert clair)
        # Score 0.5 -> max_color = 192 (Jaune/Orange)
        # Score 1.0 -> max_color = 255 (Rouge vif)
        max_color_val = 130 + (score * 125)
        
        # Application du plafond sur la heatmap normalisée
        heatmap_uint8 = np.uint8(heatmap_norm * max_color_val)

        # Redimensionnement et flou
        height, width = frame.shape[:2]
        heatmap_resized = cv2.resize(heatmap_uint8, (width, height), interpolation=cv2.INTER_CUBIC)
        k_size = int(width * 0.06) | 1 
        heatmap_resized = cv2.GaussianBlur(heatmap_resized, (k_size, k_size), 0)

        # 4. OPACITÉ FIXE
        # L'opacité ne baisse plus avec le score. On la maintient à 45% en permanence.
        constant_alpha = 0.45

        # Application de la palette de couleurs (JET)
        heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
        
        # Fusion des images
        blended_frame = cv2.addWeighted(heatmap_color, constant_alpha, frame, 1.0, 0)
        
        return blended_frame, current_smoothed_heatmap

    except Exception as e:
        print(f"Error during heatmap processing: {e}")
        return frame, previous_heatmap


def export_video_with_gauge(video_path, export_path, predictions, attn_history, heatmap_is_on):
    """Exports a new video with a dynamic risk overlay and score-correlated heatmap colors."""
    print(f"\n--- Exporting video with gauge and correlated heatmap to {export_path} ---")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(export_path, fourcc, fps, (width, height))
    
    frame_idx = 0
    window_size_8fps = 16 
    smoothed_heatmap = None 

    with tqdm(total=total_frames, desc="Exporting Video", unit="frame") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret: break

            timestamp = frame_idx / fps
            pred_idx = int(timestamp * 8.0) 

            # Récupération du score actuel
            score_idx = min(pred_idx, len(predictions) - 1)
            score = float(predictions[score_idx]) if not math.isnan(float(predictions[score_idx])) else 0.0

            # --- APPLICATION DE LA HEATMAP ---
            if heatmap_is_on and attn_history:
                if pred_idx < window_size_8fps:
                    attn_idx, temporal_index = 0, min(pred_idx // 2, 7)
                else:
                    attn_idx, temporal_index = pred_idx - window_size_8fps + 1, -1 
                
                attn_idx = min(attn_idx, len(attn_history) - 1)

                if attn_idx >= 0:
                    frame, smoothed_heatmap = apply_attention_heatmap(
                        frame=frame, 
                        rollout_1d=attn_history[attn_idx], 
                        temporal_index=temporal_index,
                        score=score, 
                        previous_heatmap=smoothed_heatmap,
                        smoothing_factor=0.85
                    )
            
            # --- DESSIN DE LA JAUGE ---
            gauge_w, gauge_h = 40, int(height * 0.6)
            margin_right, margin_bottom = 50, 50
            x_start, y_bottom = width - margin_right - gauge_w, height - margin_bottom
            y_top = y_bottom - gauge_h

            cv2.rectangle(frame, (x_start, y_top), (x_start + gauge_w, y_bottom), (40, 40, 40), -1)

            # Couleur BGR pour la jauge
            if score < 0.5:
                r, g, b = int(255 * (score/0.5)), 255, 0
            else:
                r, g, b = 255, int(255 * (1 - (score-0.5)/0.5)), 0

            gauge_color = (b, g, r) 
            y_fill_start = y_bottom - int(gauge_h * score)

            cv2.rectangle(frame, (x_start, y_fill_start), (x_start + gauge_w, y_bottom), gauge_color, -1)
            cv2.rectangle(frame, (x_start, y_top), (x_start + gauge_w, y_bottom), (200, 200, 200), 2)

            for threshold in [0.5, 0.8]:
                y_line = y_bottom - int(gauge_h * threshold)
                cv2.line(frame, (x_start - 5, y_line), (x_start + gauge_w + 5, y_line), (100, 100, 100), 2)

            text_pct = f"{int(score * 100)}%"
            font_scale = height / 800.0 
            cv2.putText(frame, text_pct, (x_start - int(90*font_scale), y_fill_start + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, gauge_color, max(1, int(2*font_scale)))

            out.write(frame)
            frame_idx += 1
            pbar.update(1)
            
    out.release()
    cap.release()
    print(f"Video successfully exported with correlated heatmap to {export_path}")