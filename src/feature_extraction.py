import os
import time
import numpy as np
import pandas as pd
from Bio import SeqIO

def needleman_wunsch(seq1, seq2, match=2, mismatch=-1, gap=-2):
    n, m = len(seq1), len(seq2)
    score_matrix = np.zeros((n + 1, m + 1))
    
    for i in range(1, n + 1):
        score_matrix[i][0] = i * gap
    for j in range(1, m + 1):
        score_matrix[0][j] = j * gap
        
    # DP Table
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if seq1[i-1] == seq2[j-1]:
                diagonal = score_matrix[i-1][j-1] + match
            else:
                diagonal = score_matrix[i-1][j-1] + mismatch
                
            atas = score_matrix[i-1][j] + gap
            kiri = score_matrix[i][j-1] + gap
            score_matrix[i][j] = max(diagonal, atas, kiri)
    return score_matrix[n][m]

def load_references(ref_dir):
    references = {}
    for filename in sorted(os.listdir(ref_dir)):
        if filename.startswith("ref_denv") and filename.endswith(".fasta"):
            # Ekstrak angka jenis (misal '1' dari 'ref_denv1.fasta')
            tipe = filename.replace("ref_denv", "").replace(".fasta", "")
            serotype = f"DENV-{tipe}"
            filepath = os.path.join(ref_dir, filename)
            record = SeqIO.read(filepath, "fasta")
            references[serotype] = str(record.seq).upper()
    return references

def load_samples(sample_dir):
    samples = []
    for filename in sorted(os.listdir(sample_dir)):
        if filename.startswith("denv_") and filename.endswith(".fasta"):
            # Ekstrak true label (misal '1' dari 'denv_1.fasta')
            tipe = filename.replace("denv_", "").replace(".fasta", "")
            true_label = f"DENV-{tipe}"
            filepath = os.path.join(sample_dir, filename)
            for record in SeqIO.parse(filepath, "fasta"):
                sample_id = record.description.split(' ')[0]
                samples.append({
                    "id": sample_id,
                    "sequence": str(record.seq).upper(),
                    "true_label": true_label
                })
    return samples

def main():
    print("="*60)
    print("GENOMIC FEATURE EXTRACTION PIPELINE (NEEDLEMAN-WUNSCH)")

    ref_dir = "../data/references"
    sample_dir = "../data/samples"

    print("[*] Membaca data...")
    references = load_references(ref_dir)
    samples = load_samples(sample_dir)
    
    print(f"    -> Found {len(references)} Referensi: {list(references.keys())}")
    print(f"    -> Found {len(samples)} Sampel Evaluasi.")
    print("-" * 60)
    
    dataset_features = []
    start_time = time.time()
    
    # Teks -> fitur 4D
    print("[*] Memulai proses penyelarasan (Alignment) DP Matrix...")
    
    for i, sample in enumerate(samples):
        print(f"    Memproses Sampel {i+1}/{len(samples)} [{sample['id']}] (Label Asli: {sample['true_label']})... ", end="", flush=True)
        feature_row = {
            "Sample_ID": sample['id'],
            "True_Label": sample['true_label']
        }    
        # comare ke semua referensi
        for ref_name, ref_seq in references.items():
            score = needleman_wunsch(sample['sequence'], ref_seq)
            feature_row[f"Score_{ref_name}"] = score
            
        dataset_features.append(feature_row)
        print("Selesai.")
        
    end_time = time.time()
    print("-" * 60)
    print(f"[*] Total waktu komputasi: {(end_time - start_time)/60:.2f} menit.")
    
    # save ke csv
    print("[*] Menyimpan hasil ekstraksi fitur...")
    df = pd.DataFrame(dataset_features)
    
    cols = ['Sample_ID', 'Score_DENV-1', 'Score_DENV-2', 'Score_DENV-3', 'Score_DENV-4', 'True_Label']
    df = df[cols]
    
    output_path = "../data/genomic_features.csv"
    df.to_csv(output_path, index=False)
    
    print(f"    -> SUKSES! Dataset tersimpan di: {output_path}")
    print("="*60)

if __name__ == "__main__":
    main()