import json
import matplotlib.pyplot as plt
import os

def plot_comparison():
    # Paths based on the user's provided context
    # Early time (CIRL): 17:02:00
    cirl_path = r'logs/exp_20251224_170200/snr_acc_best.json'
    # Late time (CIRL+CL): 17:50:43
    cirl_cl_path = r'logs/exp_20251224_175043/snr_acc_best.json'

    # Handle running from root or utils folder
    if not os.path.exists(cirl_path):
        # Try going up one level if running from utils
        cirl_path_up = os.path.join('..', cirl_path)
        cirl_cl_path_up = os.path.join('..', cirl_cl_path)
        if os.path.exists(cirl_path_up):
            cirl_path = cirl_path_up
            cirl_cl_path = cirl_cl_path_up
        else:
            # Fallback to absolute paths if relative paths fail
            # Using the paths provided in the prompt context
            cirl_path = r'd:\NieJingke\1-MyPapers\1-CIRL_AMR\CIRL_AMR_Code\logs\exp_20251224_170200\snr_acc_best.json'
            cirl_cl_path = r'd:\NieJingke\1-MyPapers\1-CIRL_AMR\CIRL_AMR_Code\logs\exp_20251224_175043\snr_acc_best.json'

    print(f"Reading CIRL data from: {cirl_path}")
    print(f"Reading CIRL+CL data from: {cirl_cl_path}")

    try:
        with open(cirl_path, 'r') as f:
            cirl_data = json.load(f)
        
        with open(cirl_cl_path, 'r') as f:
            cirl_cl_data = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: Could not find files. {e}")
        return

    snrs = cirl_data['snr']
    cirl_acc = cirl_data['accuracy']
    cirl_cl_acc = cirl_cl_data['accuracy']

    plt.figure(figsize=(10, 6))
    plt.plot(snrs, cirl_acc, marker='o', label='CIRL', linestyle='-', color='blue', linewidth=2)
    plt.plot(snrs, cirl_cl_acc, marker='s', label='CIRL+CL', linestyle='--', color='red', linewidth=2)
    
    plt.title('Recognition Accuracy vs SNR Comparison')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Accuracy (%)')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.legend()
    plt.xticks(snrs) # Ensure all SNR points are shown on x-axis
    
    output_file = 'comparison_snr_acc.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to {os.path.abspath(output_file)}")

if __name__ == "__main__":
    plot_comparison()
