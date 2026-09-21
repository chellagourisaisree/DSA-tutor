import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. CLEAN QA DATASET
# ==========================================
qa_pairs = [
    ("What is an array?", "An array is a linear data structure that stores elements of the same type in contiguous memory locations. It provides O(1) fast access by index."),
    ("What is a linked list?", "A linked list is a linear data structure where elements are stored in nodes, and each node points to the next node using a reference pointer."),
    ("Compare Array vs Linked List.", "Arrays offer O(1) random access but require fixed contiguous memory. Linked lists allow dynamic sizing and O(1) insertions/deletions at known positions, but require O(N) sequential access."),
    ("What is a stack?", "A stack is a Last-In First-Out (LIFO) linear data structure where element insertion (push) and deletion (pop) occur at the top."),
    ("What is a queue?", "A queue is a First-In First-Out (FIFO) data structure where elements are inserted at the rear (enqueue) and removed from the front (dequeue)."),
    ("What is a hash table?", "A hash table stores key-value pairs and uses a hash function to compute an index, enabling average O(1) time complexity for lookup, insertion, and deletion."),
    ("How does hash collision occur?", "A collision happens when a hash function maps two distinct keys to the same bucket index. It is resolved using chaining or open addressing."),
    ("What is a binary tree?", "A binary tree is a hierarchical data structure where each parent node can have at most two child nodes, referred to as the left child and right child."),
    ("What is a binary search tree (BST)?", "A BST is a binary tree where all nodes in the left subtree have values smaller than the root, and all nodes in the right subtree have values larger than the root."),
    ("What is a heap?", "A heap is a specialized tree-based structure satisfying the heap property. In a Max-Heap, the root is always the maximum element; in a Min-Heap, it is the minimum."),
    ("What is a graph?", "A graph is a non-linear data structure consisting of a set of vertices connected by edges, which can be directed, undirected, weighted, or unweighted."),
    ("What is Depth First Search (DFS)?", "DFS traverses a graph by exploring as far as possible along each branch before backtracking. It uses a Stack or call recursion stack and takes O(V + E) time."),
    ("What is Breadth First Search (BFS)?", "BFS traverses a graph layer by layer, exploring all immediate neighbors first using a Queue. It is ideal for finding the shortest path in unweighted graphs."),
    ("What is Big O notation?", "Big O notation measures the asymptotic upper bound of an algorithm's time or memory usage as the input size N increases toward infinity."),
    ("What is O(1) time complexity?", "O(1) or constant time means the execution time remains the same regardless of how large the input size grows."),
    ("What is O(N) time complexity?", "O(N) or linear time means the execution time scales directly in proportion to the input size N."),
    ("What is O(log N) time complexity?", "O(log N) or logarithmic time means the problem size is halved at every step, as seen in Binary Search."),
    ("What is O(N^2) time complexity?", "O(N^2) or quadratic time occurs when operations involve nested loops over the dataset, like Bubble Sort or Selection Sort."),
    ("How does Binary Search work?", "Binary Search finds an element in a sorted array by repeatedly dividing the search space in half and comparing the target with the middle element in O(log N) time."),
    ("What is Bubble Sort?", "Bubble sort continuously steps through a list, compares adjacent elements, and swaps them if they are in the wrong order until the list is sorted in O(N^2) time."),
    ("What is Merge Sort?", "Merge Sort is a divide-and-conquer algorithm that recursively divides an array into halves, sorts them, and merges them back together in O(N log N) time."),
    ("What is Quick Sort?", "Quick Sort picks a pivot element, partitions the array into elements smaller and larger than the pivot, and recursively sorts the sub-arrays. Average time is O(N log N)."),
    ("What is Dynamic Programming?", "Dynamic Programming optimizes recursive problems by breaking them into overlapping subproblems, storing intermediate results to avoid duplicate effort."),
    ("What is the Greedy strategy?", "A greedy algorithm builds up a solution piece by piece, making the locally optimal choice at each step hoping it leads to a global optimum.")
]

# Simple single-newline delimiter for robust character-level stopping
raw_text = "".join([f"Q: {q}\nA: {a}\n" for q, a in qa_pairs])
full_dataset = raw_text * 60

# ==========================================
# 2. TOKENIZER & DATA LOADER
# ==========================================
class CharTokenizer:
    def __init__(self, text):
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

    def encode(self, x):
        return [self.stoi[ch] for ch in x if ch in self.stoi]

    def decode(self, x):
        return ''.join([self.itos[i] for i in x])

tokenizer = CharTokenizer(full_dataset)
encoded_tokens = torch.tensor(tokenizer.encode(full_dataset), dtype=torch.long)

def get_batch(batch_size, block_size):
    ix = torch.randint(len(encoded_tokens) - block_size, (batch_size,))
    x = torch.stack([encoded_tokens[i:i+block_size] for i in ix])
    y = torch.stack([encoded_tokens[i+1:i+block_size+1] for i in ix])
    return x, y

# ==========================================
# 3. TRANSFORMER ARCHITECTURE
# ==========================================
class MiniGPT(nn.Module):
    def __init__(self, vocab_size, n_embd=256, n_heads=4, n_layers=4, block_size=256):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_embd,
            nhead=n_heads,
            dim_feedforward=4*n_embd,
            activation='gelu',
            batch_first=True
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        pos = torch.arange(0, T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)

        mask = torch.triu(torch.ones(T, T, device=idx.device) * float('-inf'), diagonal=1)
        x = self.blocks(x, mask=mask, is_causal=True)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]

            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            idx = torch.cat((idx, idx_next), dim=1)

            # Check string ending in real time to stop immediately when answer completes
            current_str = tokenizer.decode(idx[0].tolist())
            if "A:" in current_str:
                ans_part = current_str.split("A:")[1]
                if "\n" in ans_part or "Q:" in ans_part:
                    break
        return idx

# ==========================================
# 4. MODEL TRAINING
# ==========================================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = MiniGPT(vocab_size=tokenizer.vocab_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

print(f"Training on {device.upper()}...")
model.train()
for step in range(1500):
    xb, yb = get_batch(batch_size=32, block_size=128)
    xb, yb = xb.to(device), yb.to(device)

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if step % 300 == 0 or step == 1499:
        print(f"Step {step:4d} | Loss: {loss.item():.4f}")

print("\nModel Trained Successfully!")

# ==========================================
# 5. TERMINAL CHAT INTERFACE WITH EXIT LOOP
# ==========================================
def run_dsa_tutor(model, tokenizer, device, max_new_tokens=150):
    print("=" * 60)
    print("🤖 GEMINI DSA TUTOR IS LIVE!")
    print("Type your question below, or type 'exit' or 'quit' to stop.")
    print("=" * 60)

    model.eval()
    while True:
        user_query = input("\n👤 You: ").strip()

        if user_query.lower() in ['exit', 'quit']:
            print("\n🤖 DSA Tutor: Goodbye! Happy coding!")
            break

        if not user_query:
            continue

        prompt = f"Q: {user_query}\nA:"
        encoded = tokenizer.encode(prompt)

        if not encoded:
            print("🤖 DSA Tutor: Sorry, I couldn't parse those characters.")
            continue

        idx = torch.tensor(encoded, dtype=torch.long, device=device).unsqueeze(0)

        with torch.no_grad():
            out_idx = model.generate(idx, max_new_tokens=max_new_tokens)

        raw_output = tokenizer.decode(out_idx[0].tolist())

        # Cleanly extract answer and strip any trailing artifacts
        try:
            response = raw_output.split("A:")[1].split("\n")[0].split("Q:")[0].strip()
        except IndexError:
            response = raw_output.strip()

        print(f"🤖 DSA Tutor: {response}")

run_dsa_tutor(model, tokenizer, device)
