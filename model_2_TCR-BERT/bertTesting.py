from transformers import BertTokenizerFast, BertForSequenceClassification
import torch
import csv
import argparse
from torch.utils.data import DataLoader, Dataset

class PeptideTCRDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        peptide, tcr_sequence = self.data[idx]
        return peptide, tcr_sequence

def preprocess_input(peptide, tcr_sequence):
    def process(input_text):
        input_text = input_text.upper()
        return ' '.join(input_text)

    processed_peptide = process(peptide)
    processed_tcr_sequence = process(tcr_sequence)
    return processed_peptide, processed_tcr_sequence

def predict_batch(batch, model, tokenizer, device):
    peptides, tcr_sequences = batch

    processed_peptides, processed_tcr_sequences = zip(*[preprocess_input(p, t) for p, t in zip(peptides, tcr_sequences)])

    inputs = tokenizer(
        list(processed_peptides),
        list(processed_tcr_sequences),
        return_tensors='pt',
        truncation=True,
        padding=True,
        max_length=64  
    )
    
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
        predicted_labels = torch.argmax(probabilities, dim=-1)

    return predicted_labels, probabilities

if __name__ == "__main__":
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on {device}")

    with open('tcr_split_test.csv', 'r') as f:
        lines = f.readlines()
    
    input = []
    for each in lines:
        temp = each.split(',')
        input.append([temp[0].strip(), temp[1].strip()])
    
    batch_size = 16  
    dataset = PeptideTCRDataset(input)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = BertForSequenceClassification.from_pretrained('./bert_TCR_Split')
    tokenizer = BertTokenizerFast.from_pretrained('wukevin/tcr-bert')
    model.to(device)  
    
    results = []
    print(f"Total samples: {len(input)}")
    
    for i, batch in enumerate(dataloader):
        peptides, tcr_sequences = batch
        
        predicted_labels, probabilities = predict_batch(batch, model, tokenizer, device)
        
        for predicted_label,prob in zip(predicted_labels,probabilities):
            results.append({"Predicted_Label": predicted_label.item(),
                            "Probability_1": prob[1].item()})
        
        if i % 10 == 0:
            print(f"Processed batch {i + 1}/{len(dataloader)}")
    
    with open('BertTcrSplitCompetitionResult.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["Predicted_Label","Probability_1"])
        writer.writerows(results)

    print("Testing completed and results saved.")
