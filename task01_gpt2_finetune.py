
# ============================================
# Task 01 - Text Generation with GPT-2
# Generative AI Internship
# ============================================

import torch
import warnings
warnings.filterwarnings("ignore")
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

# ── STEP 1: Create Dataset ──────────────────
articles = [
    "Artificial intelligence is the simulation of human intelligence by machines. AI systems are designed to perform tasks that typically require human cognition such as learning, reasoning, problem-solving, and understanding language. Modern AI is powered by large datasets and powerful computing hardware.",
    "Machine learning is a subset of artificial intelligence that enables systems to learn from data without being explicitly programmed. Algorithms identify patterns in data and improve their performance over time.",
    "Deep learning uses neural networks with many layers to learn representations of data. It has achieved remarkable results in image recognition, speech recognition, and natural language processing.",
    "Large language models are AI systems trained on vast amounts of text data to understand and generate human language. Models like GPT-4 and Claude can write essays, answer questions, and summarize documents.",
    "The transformer architecture revolutionized natural language processing in 2017. Transformers process entire sequences in parallel using self-attention, capturing long-range dependencies more effectively than older models.",
    "Generative AI refers to systems that can create new content including text, images, audio, and video. Examples include ChatGPT, DALL-E, Suno, and Sora.",
    "Neural networks consist of layers of interconnected nodes that process information. During training the connections are adjusted to minimize prediction errors.",
    "Natural language processing enables computers to understand and generate human language. Applications include translation, sentiment analysis, chatbots, and summarization.",
    "Reinforcement learning trains an agent to make decisions by rewarding good actions and penalizing bad ones. It has been used to train AI that plays games at superhuman levels.",
    "Computer vision enables machines to interpret visual information from images and videos. Applications include facial recognition, medical imaging, and autonomous vehicles.",
    "Fine-tuning is taking a pretrained model and training it further on a smaller task-specific dataset. It is far more efficient than training from scratch.",
    "Prompt engineering is designing inputs to AI models to get the best outputs. Clear instructions and examples lead to better results.",
    "GPT-2 is a language model by OpenAI released in 2019 with 1.5 billion parameters. It generates surprisingly coherent text and is widely used for fine-tuning experiments.",
    "The attention mechanism allows models to focus on the most relevant parts of the input. Self-attention computes relationships between all tokens simultaneously.",
    "Data quality and quantity directly impact model performance. Collecting and cleaning data is often the most time-consuming part of building AI applications.",
    "AI ethics covers bias, privacy, job displacement, and misuse of AI tools. Researchers and policymakers are developing frameworks for responsible AI.",
    "Cloud computing made AI accessible to everyone. Platforms like Google Colab provide free GPU access without owning expensive hardware.",
    "Transfer learning reuses a model trained on one task for a different related task. It reduces the data and compute needed for new applications.",
    "Edge AI runs algorithms directly on devices like smartphones instead of the cloud. It reduces latency and preserves privacy.",
    "Explainable AI makes model decisions transparent and understandable. It is critical in healthcare, finance, and criminal justice.",
]

with open("dataset.txt", "w", encoding="utf-8") as f:
    for article in articles:
        f.write(article.strip() + "\n\n")

print("Dataset created!")

# ── STEP 2: Load Model ──────────────────────
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained("gpt2")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
print(f"Model loaded on {device}")

# ── STEP 3: Tokenize Dataset ────────────────
with open("dataset.txt", "r", encoding="utf-8") as f:
    text = f.read()

tokens = tokenizer.encode(text)
chunk_size = 64
chunks = [tokens[i:i+chunk_size] for i in range(0, len(tokens)-chunk_size, chunk_size)]

class TextDataset(Dataset):
    def __init__(self, chunks):
        self.chunks = chunks
    def __len__(self):
        return len(self.chunks)
    def __getitem__(self, i):
        t = torch.tensor(self.chunks[i])
        return {"input_ids": t, "labels": t}

train_dataset = TextDataset(chunks)
print(f"Total chunks: {len(chunks)}")

# ── STEP 4: Fine-tune ───────────────────────
loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
optimizer = AdamW(model.parameters(), lr=5e-5)
model.train()

print("Starting fine-tuning...")
for epoch in range(10):
    total_loss = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        labels    = batch["labels"].to(device)
        loss = model(input_ids=input_ids, labels=labels).loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/10 | Loss: {total_loss/len(loader):.4f}")

# ── STEP 5: Generate Text ───────────────────
model.eval()

def generate(prompt, max_new_tokens=100, temperature=0.9):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)

prompts = [
    "Artificial intelligence is",
    "The future of machine learning",
    "Deep learning models can",
]

for prompt in prompts:
    print(f"Prompt: {prompt}")
    print(generate(prompt))
    print("-" * 60)
