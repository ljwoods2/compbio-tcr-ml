from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
from transformers import pipeline

pipe = pipeline("text-classification", model="wukevin/tcr-bert")

from transformers import AutoTokenizer, AutoModelForSequenceClassification

def process(input):
    
    input=input.upper()
    res=""
    for each in input:
        res+=each+ " "
    
    return res[:-1]

df_train = pd.read_csv('tcrSplit/train.csv', header=None, names=['peptide', 'tcr_sequence', 'label'])
df_test = pd.read_csv('tcrSplit/test.csv', header=None, names=['peptide', 'tcr_sequence', 'label'])
df_train['label'] = df_train['label'].astype(int)
df_test['label'] = df_test['label'].astype(int)

for split in ['peptide', 'tcr_sequence']:
    df_train[split] = df_train[split].map(process)
    df_test[split] = df_test[split].map(process)

# print(df_train['peptide'][1],len(df_train['peptide'][1]))

train_dataset = Dataset.from_pandas(df_train)
test_dataset = Dataset.from_pandas(df_test)
dataset = {'train': train_dataset, 'test': test_dataset}


# labels = df_train['label']
# print(f"Class distribution: {np.bincount(labels)}")

# tokenizer = AutoTokenizer.from_pretrained("wukevin/tcr-bert")
# model = AutoModelForSequenceClassification.from_pretrained("wukevin/tcr-bert")

tokenizer = BertTokenizerFast.from_pretrained('wukevin/tcr-bert')
# t="E A A G I G I L T V"
# tokens = tokenizer.tokenize(t)
# print(f"Tokens for 'EAAGIGILTV': {tokens}")

# tokens = tokenizer.tokenize("C A S S P V T G G I Y G Y T F")
# print(f"Tokens for 'CASSQEEGGGSWGNTIYF': {tokens}")
model = BertForSequenceClassification.from_pretrained('wukevin/tcr-bert', num_labels=2,ignore_mismatched_sizes=True)

def tokenize_function(example):
    return tokenizer(
        text=example['peptide'],
        text_pair=example['tcr_sequence'],
        truncation=True,
        padding='max_length',
        max_length=64,  
    )

tokenized_datasets = {}
# print(tokenizer(t, "C A S S P V T G G I Y G Y T F", truncation=True, padding=True))
for split in ['train', 'test']:
    tokenized_datasets[split] = dataset[split].map(tokenize_function, batched=True)
    
print(len(tokenized_datasets['train']))
for i in range(0,3):
    print(tokenized_datasets['train'][i])

for split in ['train', 'test']:
    tokenized_datasets[split] = tokenized_datasets[split].remove_columns(['peptide', 'tcr_sequence'])
    tokenized_datasets[split].set_format('torch')
    
# print(tokenized_datasets['test'][0])

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), axis=1).numpy()
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary', zero_division=1)
    acc = accuracy_score(labels, predictions)
    return {
        'accuracy': round(acc, 3),
        'f1': round(f1, 3),
        'precision': round(precision, 3),
        'recall': round(recall, 3)
    }

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=5,  
    per_device_train_batch_size=32,  
    learning_rate=5e-5,  
    warmup_steps=900,  
    weight_decay=0.01,
    logging_dir='./logs',
    eval_strategy="epoch",
    save_strategy="epoch",
    # save_total_limit=2,  
    load_best_model_at_end=True, 
    metric_for_best_model="accuracy",
    report_to=[],
)
trainer = Trainer(
    model=model,                                
    args=training_args,                         
    train_dataset=tokenized_datasets['train'],  
    eval_dataset=tokenized_datasets['test'],   
    compute_metrics=compute_metrics, 
)

trainer.train()
trainer.save_model('./wukevin_tcr_bert_model')

# test_results = trainer.evaluate(tokenized_datasets['test'])
# print(test_results) 

