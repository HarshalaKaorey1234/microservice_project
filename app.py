#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Install required Libraries

get_ipython().system('pip install fastapi uvicorn torch torchvision torchtext spacy')
get_ipython().system('python -m spacy download en_core_web_sm')


# In[2]:


# Import required libraries

# Import required libraries
import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.model_selection import train_test_split
import pandas as pd


# In[3]:


# **Step 1: Vocabulary Creation**

# Example data for vocabulary creation
data = pd.DataFrame({
    'text': ["This is an example sentence.", "Another example text."],
    'summary': ["Example summary.", "Another summary."]
})

# Split the dataset into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Tokenizer setup
tokenizer = get_tokenizer('spacy', language='en_core_web_sm')

# Function to yield tokens from a dataset column
def yield_tokens(data, col):
    for _, row in data.iterrows():
        yield tokenizer(row[col])

# Build vocabularies for text and summary
TEXT_VOCAB = build_vocab_from_iterator(
    yield_tokens(train_data, 'text'),
    specials=['<unk>', '<pad>', '<sos>', '<eos>']
)
TEXT_VOCAB.set_default_index(TEXT_VOCAB['<unk>'])

SUMMARY_VOCAB = build_vocab_from_iterator(
    yield_tokens(train_data, 'summary'),
    specials=['<unk>', '<pad>', '<sos>', '<eos>']
)
SUMMARY_VOCAB.set_default_index(SUMMARY_VOCAB['<unk>'])

# Save vocabularies
torch.save(TEXT_VOCAB, 'text_vocab.pt')
torch.save(SUMMARY_VOCAB, 'summary_vocab.pt')
print("Vocabularies created and saved successfully.")


# In[4]:


# **Step 2: Model Definition**

# Define Encoder
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)
        return hidden, cell

# Define Decoder
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        input = input.unsqueeze(1)
        embedded = self.dropout(self.embedding(input))
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(1))
        return prediction, hidden, cell

# Define Seq2Seq Model
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        hidden, cell = self.encoder(src)

        input = trg[:, 0]
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[:, t, :] = output
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            input = trg[:, t] if teacher_force else output.argmax(1)

        return outputs


# In[5]:


# **Step 3: Save the Model**

# Initialize model parameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
INPUT_DIM = len(TEXT_VOCAB)
OUTPUT_DIM = len(SUMMARY_VOCAB)
ENC_EMB_DIM = DEC_EMB_DIM = 256
HID_DIM = 512
N_LAYERS = 2
ENC_DROPOUT = DEC_DROPOUT = 0.5

# Initialize Encoder, Decoder, and Seq2Seq Model
encoder = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
decoder = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
model = Seq2Seq(encoder, decoder, device).to(device)

# Save the model
torch.save(model.state_dict(), 'seq2seq-model.pt')
print("Model saved successfully.")


# In[6]:


# **Step 4: Microservice Deployment**

# Load vocabularies and model
TEXT_VOCAB = torch.load('text_vocab.pt')
SUMMARY_VOCAB = torch.load('summary_vocab.pt')
model.load_state_dict(torch.load('seq2seq-model.pt', map_location=device))
model.eval()

# Define FastAPI application
app = FastAPI()

# Request and Response structure
class TextInput(BaseModel):
    text: str

class SummaryOutput(BaseModel):
    summary: str

# Define pipelines for tokenization
text_pipeline = lambda x: [TEXT_VOCAB['<sos>']] + [TEXT_VOCAB[token] for token in tokenizer(x)] + [TEXT_VOCAB['<eos>']]

# Endpoint for summarization
@app.post("/summarize", response_model=SummaryOutput)
def generate_summary(input: TextInput):
    text = input.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    tokenized_input = text_pipeline(text)
    input_tensor = torch.tensor(tokenized_input).unsqueeze(0).to(device)

    with torch.no_grad():
        hidden, cell = model.encoder(input_tensor)
        input_token = torch.tensor([SUMMARY_VOCAB['<sos>']]).to(device)
        output_tokens = []

        for _ in range(50):  # Limit summary length
            output, hidden, cell = model.decoder(input_token, hidden, cell)
            predicted_token = output.argmax(1).item()
            if predicted_token == SUMMARY_VOCAB['<eos>']:
                break
            output_tokens.append(predicted_token)
            input_token = torch.tensor([predicted_token]).to(device)

    summary = ' '.join([SUMMARY_VOCAB.get_itos()[token] for token in output_tokens])
    return {"summary": summary}


# In[7]:


# **Step 5: Running Instructions**

"""
1. Save this code as a Python file (e.g., `app.py`).
2. Start the FastAPI server:
   - Run the command: `uvicorn app:app --host 0.0.0.0 --port 8000 --reload`
3. Test the endpoint using Postman or `curl`:
   - URL: `http://127.0.0.1:8000/summarize`
   - Request Body: `{"text": "Your input text here."}`
   - Example Response: `{"summary": "Generated summary"}`
"""


# In[ ]:




