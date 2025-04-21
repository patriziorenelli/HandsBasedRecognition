import numpy as np
from sklearn.metrics import auc, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_accuracy(y_pred, y_true):
    return sum(y_pred == y_true) / len(y_true)

def calculate_confusion_matrix(y_pred, y_true):
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Plot confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()


def calculate_FAR_plot(predicted_scores: np.ndarray, true_labels: np.ndarray, type_feature_extractor: str, palm_dorsal: str):
    # Label normalization: 1 = genuine, 0 = impostor
    true_labels_binary = np.where(true_labels != -1, 1, 0)
    predicted_score = np.max(predicted_scores, axis=1)

    # Real number of impostor images
    num_impostor_images = np.sum(true_labels_binary == 0)
    
    thresholds = np.linspace(0, 1, 1000)  # Threshold between 0 and 1 
    far_values = []

    for threshold in thresholds:
        predicted_labels = predicted_score >= threshold  # Predicted labels
        false_positives = np.sum((predicted_labels == 1) & (true_labels_binary == 0))
        far = false_positives / num_impostor_images if num_impostor_images > 0 else 0
        far_values.append(far)

    # Graph creation
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, far_values, label=f"{type_feature_extractor} {palm_dorsal} FAR", color='blue')
    plt.title('False Alarm Rate (FAR) vs Threshold', fontsize=16)
    plt.xlabel('Threshold', fontsize=14)
    plt.ylabel('False Alarm Rate (FAR)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.show()

    return far_values

    
def calculate_FRR_plot(predicted_scores: np.ndarray, true_labels: np.ndarray, type_feature_extractor: str, palm_dorsal: str):
    # Label normalization: 1 = genuine, 0 = impostor
    true_labels_binary = np.where(true_labels != -1, 1, 0)
    predicted_score = np.max(predicted_scores, axis=1)

    # Real number of genuine images
    num_genuine_images = np.sum(true_labels_binary)

   
    thresholds = np.linspace(0, 1, 1000)  # Threshold between 0 and 1
    frr_values = []

    for threshold in thresholds:
        predicted_labels = predicted_score >= threshold  # Predicted labels
        false_negatives = np.sum((predicted_labels == 0) & (true_labels_binary == 1))
        frr = false_negatives / num_genuine_images if num_genuine_images > 0 else 0
        frr_values.append(frr)

    # Graph creation
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, frr_values, label=f"{type_feature_extractor} {palm_dorsal} FRR", color='red')
    plt.title('False Rejection Rate (FRR) vs Threshold', fontsize=16)
    plt.xlabel('Threshold', fontsize=14)
    plt.ylabel('False Rejection Rate (FRR)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.show()

    return frr_values

def plot_FAR_FRR(far_values: np.array, frr_values: np.array, type_feature_extractor: str, palm_dorsal: str):  
    thresholds = np.linspace(0, 1, 1000)  

    # EER calculation: point where FAR ≈ FRR
    eer_index = np.nanargmin(np.abs(np.array(far_values) - np.array(frr_values)))  
    eer_threshold = thresholds[eer_index]
    eer_value = far_values[eer_index]  # EER = FAR (o FRR) on the intersection

    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, far_values, label=f"{type_feature_extractor} {palm_dorsal} FAR (False Acceptance Rate)", color='blue')
    plt.plot(thresholds, frr_values, label=f"{type_feature_extractor} {palm_dorsal} FRR (False Rejection Rate)", color='red')
    
    # EER show how a point and a dashed line
    plt.axvline(eer_threshold, color='green', linestyle='--', label=f"EER = {eer_value:.3f} @ threshold {eer_threshold:.3f}")
    plt.scatter(eer_threshold, eer_value, color='green', s=100, label="Equal Error Rate (EER)")

    plt.xlabel("Threshold", fontsize=14)
    plt.ylabel("Rate", fontsize=14)
    plt.title("FAR vs FRR con evidenziazione EER", fontsize=16)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.show()

    print(f"Equal Error Rate (EER): {eer_value:.3f} a soglia {eer_threshold:.3f}")

def calculate_CMC(score_matrix: np.ndarray, true_labels: np.ndarray, gallery_labels: np.ndarray):
    num_tests, num_gallery = score_matrix.shape
    ranks = np.zeros(num_gallery)
    num_samples_skipped = 0  # Contatore per test senza corrispondenza

    for i in range(num_tests):

        # Estrai la riga i (punteggi per il test i)
        scores = score_matrix[i]  

        # Ordina i punteggi in ordine decrescente
        sorted_indices = np.argsort(scores)[::-1]

        # Order the gallery labels according to the sorted indices
        sorted_labels = gallery_labels[sorted_indices] 
        correct_rank = np.where(sorted_labels == true_labels[i])[0]

        # If the identity is not in the gallery
        if correct_rank.size == 0:  
            print(f"Errore: impossibile trovare {true_labels[i]} in sorted_labels!")
            num_samples_skipped += 1
            continue  

        # Get the first correct rank (the first occurrence of the true label in the sorted list)
        correct_rank = correct_rank[0]

        # Encrease the correct rank and all ranks after it
        ranks[correct_rank:] += 1

    # Normalize the CMC curve to percentage
    cmc_curve = ranks / (num_tests - num_samples_skipped)

    return cmc_curve


def calculate_CMC_plot(score_matrix: np.ndarray, true_labels: np.ndarray, gallery_labels: np.ndarray, type_feature_extractor: str, palm_dorsal: str):
    cmc_curve = calculate_CMC(score_matrix, true_labels, gallery_labels)
    ranks = np.arange(1, len(cmc_curve) + 1)

    plt.figure(figsize=(8, 6))
    plt.plot(ranks, cmc_curve, marker='o', linestyle='-', color='green', label=f"CMC {type_feature_extractor} {palm_dorsal}")
    plt.title("Cumulative Match Characteristic (CMC) Curve", fontsize=16)
    plt.xlabel("Rank", fontsize=14)
    plt.ylabel("Recognition Rate", fontsize=14)
    # Mostra solo alcuni tick
    plt.xticks(ranks[::max(len(ranks)//10, 1)])  
    # Da 0 a 1 con step di 0.1
    plt.yticks(np.linspace(0, 1, 11))  
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(fontsize=12)
    plt.show()


def plot_OSROC_curve(true_labels: np.ndarray, predicted_scores: np.ndarray, known_classes: np.ndarray, type_feature_extractor: str, palm_dorsal: str):
    # Create the binary labels for the open set
    is_known = np.isin(true_labels, known_classes)  # 1 if it is a known class, 0 if it is an impostor

    # Highest score among known classes for each sample
    max_scores = predicted_scores.max(axis=1)

    # ROC curve calculation
    fpr, tpr, _ = roc_curve(is_known, max_scores)
    
    # Area under the curve (AUC) for the ROC curve
    roc_auc = auc(fpr, tpr)  

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'{type_feature_extractor} {palm_dorsal} OSROC Curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Linea casuale
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)', fontsize=14)
    plt.ylabel('True Positive Rate (TPR)', fontsize=14)
    plt.title('Open-Set ROC Curve (OSROC)', fontsize=16)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()
