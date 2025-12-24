import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
from sklearn.metrics import confusion_matrix

from data.dataset import RML2016Dataset
from models.networks import resnet18_1d, Masker, Classifier

def factorization_loss(f_a, f_b):
    # f_a, f_b: (B, D)
    # Normalize
    f_a_norm = (f_a - f_a.mean(0)) / (f_a.std(0) + 1e-6)
    f_b_norm = (f_b - f_b.mean(0)) / (f_b.std(0) + 1e-6)
    
    # Cross-correlation matrix
    c = torch.mm(f_a_norm.T, f_b_norm) / f_a.shape[0]
    
    on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
    off_diag = c.flatten()[:-1].view(c.shape[0]-1, c.shape[0]+1)[:, 1:].flatten()
    off_diag = off_diag.pow_(2).sum()
    
    loss = on_diag + 0.005 * off_diag
    return loss

def save_snr_results(log_dir, snr_accs):
    # Ensure directory exists
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Prepare data
    snrs = sorted(list(snr_accs.keys()))
    accs = [snr_accs[s] for s in snrs]
    
    # Save to JSON
    data = {'snr': [float(s) for s in snrs], 'accuracy': [float(a) for a in accs]}
    with open(os.path.join(log_dir, 'snr_acc_best.json'), 'w') as f:
        json.dump(data, f, indent=4)
        
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(snrs, accs, 'o-', linewidth=2, label='CIRL')
    plt.title('Accuracy vs SNR')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(log_dir, 'snr_acc_curve.png'))
    plt.close()

def save_confusion_matrix(log_dir, preds, labels, snrs, classes, target_snrs=[0, 18]):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    preds = np.array(preds)
    labels = np.array(labels)
    snrs = np.array(snrs)
    
    for snr in target_snrs:
        # Filter data for specific SNR
        indices = np.where(snrs == snr)[0]
        if len(indices) == 0:
            print(f"Warning: No samples found for SNR {snr}dB")
            continue
            
        curr_preds = preds[indices]
        curr_labels = labels[indices]
        
        cm = confusion_matrix(curr_labels, curr_preds)
        # Normalize
        cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)
        
        plt.figure(figsize=(12, 10))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix (SNR={snr}dB)')
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], '.2f'),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
                         
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig(os.path.join(log_dir, f'confusion_matrix_{snr}dB.png'))
        plt.close()

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data
    print("Loading dataset...")
    train_dataset = RML2016Dataset(args.data_path, train=True)
    test_dataset = RML2016Dataset(args.data_path, train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    num_classes = len(train_dataset.classes)
    print(f"Classes: {train_dataset.classes}")
    
    # Models
    backbone = resnet18_1d(num_classes=num_classes).to(device)
    feature_dim = backbone.feature_dim
    
    masker = Masker(feature_dim).to(device)
    classifier = Classifier(feature_dim, num_classes).to(device)
    classifier_ad = Classifier(feature_dim, num_classes).to(device) # Adversarial classifier
    
    # Optimizers
    # Step 1: Encoder + Classifiers
    # Added weight_decay to combat overfitting
    optimizer_enc = optim.Adam(
        list(backbone.parameters()) + list(classifier.parameters()) + list(classifier_ad.parameters()),
        lr=args.lr,
        weight_decay=1e-4
    )
    
    # Step 2: Masker
    optimizer_mask = optim.Adam(masker.parameters(), lr=args.lr, weight_decay=1e-4)
    
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0.0
    
    for epoch in range(args.epochs):
        backbone.train()
        masker.train()
        classifier.train()
        classifier_ad.train()
        
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for x_orig, x_aug, labels, snrs in pbar:
            x_orig, x_aug, labels = x_orig.to(device), x_aug.to(device), labels.to(device)
            
            # --- Forward ---
            f_orig = backbone(x_orig)
            f_aug = backbone(x_aug)
            
            # Generate Masks
            mask = masker(f_orig)
            
            # Split features
            f_sup = f_orig * mask
            f_inf = f_orig * (1 - mask)
            
            # Predictions
            pred_sup = classifier(f_sup)
            pred_inf = classifier_ad(f_inf)
            
            # --- Step 1: Update Encoder & Classifiers ---
            # Minimize L_sup + L_inf + L_fac
            loss_sup = criterion(pred_sup, labels)
            loss_inf = criterion(pred_inf, labels)
            loss_fac = factorization_loss(f_orig, f_aug)
            
            loss_step1 = loss_sup + args.alpha * loss_inf + args.beta * loss_fac
            
            optimizer_enc.zero_grad()
            loss_step1.backward()
            optimizer_enc.step()
            
            # --- Step 2: Update Masker ---
            # Minimize L_sup - gamma * L_inf
            # We want to maximize L_inf (make f_inf uninformative)
            
            # Re-compute forward pass for Masker update
            # We use the updated backbone to get features (detached, as we don't update backbone here)
            with torch.no_grad():
                f_orig_2 = backbone(x_orig)
            
            # Generate mask (track gradients for Masker)
            mask_2 = masker(f_orig_2)
            
            # Split
            f_sup_2 = f_orig_2 * mask_2
            f_inf_2 = f_orig_2 * (1 - mask_2)
            
            # Predict (detach classifiers, as we don't update them)
            # We treat Classifiers as fixed functions here
            pred_sup_2 = classifier(f_sup_2)
            pred_inf_2 = classifier_ad(f_inf_2)
            
            loss_sup_2 = criterion(pred_sup_2, labels)
            loss_inf_2 = criterion(pred_inf_2, labels)
            
            loss_step2 = loss_sup_2 - args.gamma * loss_inf_2
            
            optimizer_mask.zero_grad()
            loss_step2.backward()
            optimizer_mask.step()
            
            total_loss += loss_step1.item()
            
            _, predicted = pred_sup.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'Loss': total_loss/(total/args.batch_size), 'Acc': 100.*correct/total})
            
        # Evaluation
        acc, snr_accs, preds, labels_gt, snrs_gt = evaluate(backbone, masker, classifier, test_loader, device)
        print(f"Test Accuracy: {acc:.2f}%")
        
        if acc > best_acc:
            best_acc = acc
            torch.save(backbone.state_dict(), os.path.join(args.log_dir, 'best_backbone.pth'))
            torch.save(masker.state_dict(), os.path.join(args.log_dir, 'best_masker.pth'))
            torch.save(classifier.state_dict(), os.path.join(args.log_dir, 'best_classifier.pth'))
            
            # Save SNR results for the best model
            save_snr_results(args.log_dir, snr_accs)
            # Save Confusion Matrix for the best model (0dB and 18dB)
            save_confusion_matrix(args.log_dir, preds, labels_gt, snrs_gt, train_dataset.classes, target_snrs=[0, 18])

def evaluate(backbone, masker, classifier, loader, device):
    backbone.eval()
    masker.eval()
    classifier.eval()
    
    correct = 0
    total = 0
    
    # Track accuracy per SNR
    snr_correct = {}
    snr_total = {}
    
    all_preds = []
    all_labels = []
    all_snrs = []
    
    with torch.no_grad():
        for x, _, labels, snrs in loader:
            x, labels = x.to(device), labels.to(device)
            snrs_np = snrs.numpy()
            
            f = backbone(x)
            mask = masker(f)
            f_sup = f * mask
            outputs = classifier(f_sup)
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Collect for CM
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_snrs.extend(snrs_np)
            
            # Per SNR stats
            c = predicted.eq(labels).cpu().numpy()
            for i, snr in enumerate(snrs_np):
                if snr not in snr_total:
                    snr_total[snr] = 0
                    snr_correct[snr] = 0
                snr_total[snr] += 1
                snr_correct[snr] += c[i]
            
    # Print SNR accuracy
    print("\nAccuracy per SNR:")
    snr_accs = {}
    sorted_snrs = sorted(snr_total.keys())
    for snr in sorted_snrs:
        acc = 100. * snr_correct[snr] / snr_total[snr]
        snr_accs[snr] = acc
        print(f"SNR {snr}dB: {acc:.2f}%")
        
    return 100. * correct / total, snr_accs, all_preds, all_labels, all_snrs

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/RML2016.10a_dict.pkl')
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--alpha', type=float, default=0.5, help='Weight for adversarial loss in Step 1')
    parser.add_argument('--beta', type=float, default=0.1, help='Weight for factorization loss')
    parser.add_argument('--gamma', type=float, default=0.5, help='Weight for adversarial loss in Step 2')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
        
    train(args)
