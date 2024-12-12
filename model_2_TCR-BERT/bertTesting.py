from transformers import BertTokenizerFast, BertForSequenceClassification
import torch
import csv
import argparse

def preprocess_input(peptide, tcr_sequence):
    
    def process(input_text):
        input_text = input_text.upper()
        return ' '.join(input_text)

    processed_peptide = process(peptide)
    processed_tcr_sequence = process(tcr_sequence)
    return processed_peptide, processed_tcr_sequence

def predict(peptide, tcr_sequence, device):
   
    model_path = './bert_EPI_Split'  
    tokenizer = BertTokenizerFast.from_pretrained('wukevin/tcr-bert')
    model = BertForSequenceClassification.from_pretrained('./bert_EPI_Split')
    
    model.to(device)
    
    processed_peptide, processed_tcr_sequence = preprocess_input(peptide, tcr_sequence)

    # Tokenize input data
    inputs = tokenizer(
        text=processed_peptide,
        text_pair=processed_tcr_sequence,
        return_tensors='pt',
        truncation=True,
        padding='max_length',
        max_length=64  
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1).squeeze()
        predicted_label = torch.argmax(probabilities).item()

    return predicted_label, probabilities

if __name__ == "__main__":
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on {device}")

    with open('epitope_split_test.csv', 'r') as f:
        lines = f.readlines()
    
    input=[]
    for each in lines:
        temp=each.split(',')
        input.append([temp[0].strip(),temp[1].strip()])
        
    results = []
    print(len(input))
    for i,row in enumerate (input):
        peptide, tcr_sequence = row[0].strip(), row[1].strip()
        label, probabilities = predict(peptide, tcr_sequence,device)
        print(i,label)
        results.append({"Predicted_Label": label,})
    
    with open('epiSplitResult', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["Predicted_Label"])
        writer.writerows(results)
