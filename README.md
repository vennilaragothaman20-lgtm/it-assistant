# IT Troubleshooting Assistant (LLM + RAG)

A simple RAG-based IT troubleshooting chatbot. Uses:
- **ChromaDB** — local vector database (no server needed)
- **HuggingFace embeddings** (`all-MiniLM-L6-v2`) — free, runs locally, no API key required
- **Claude (Anthropic API)** — generates the final step-by-step answer
- **Streamlit** — chat interface

## Project structure
```
it-assistant/
├── data/
│   └── it_tickets.csv      # knowledge base (sample IT errors + solutions)
├── ingest.py                # builds the vector database from the CSV
├── app.py                   # Streamlit chat app
├── requirements.txt
├── render.yaml               # Render deployment config
├── .gitignore
└── README.md
```

## Setup

### 1. Install Python dependencies
```bash
cd it-assistant
python -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set your Anthropic API key
Get a key from https://console.anthropic.com/

```bash
export ANTHROPIC_API_KEY="your-api-key-here"     # on Windows (PowerShell): $env:ANTHROPIC_API_KEY="your-api-key-here"
```

### 3. Build the vector database
This reads `data/it_tickets.csv`, creates embeddings, and stores them in `chroma_db/`.
```bash
python ingest.py
```
You should see:
```
Loaded 10 documents.
Loading embedding model...
Creating ChromaDB vector store...
Done. Vector store saved to 'chroma_db/'.
```

### 4. Run the chat app
```bash
streamlit run app.py
```
This opens a browser tab at `http://localhost:8501` with the chat interface.

## Try it
Ask things like:
- "My Wi-Fi keeps dropping on my laptop"
- "VPN won't connect, says connection timed out"
- "Outlook isn't getting new emails"
- "I'm getting a blue screen with CRITICAL_PROCESS_DIED"

The app retrieves the closest matching past tickets from `it_tickets.csv` and asks
Claude to turn them into a clear step-by-step answer for the user's specific question.

## Add your own data
Edit `data/it_tickets.csv` and add rows with two columns:
```
error_description,solution
"your error text","your fix steps"
```
Then re-run `python ingest.py` to rebuild the vector database.

## Deploying on Render

### 1. Push the project to GitHub
```bash
cd it-assistant
git init
git add .
git commit -m "Initial commit: IT troubleshooting assistant"
git branch -M main
git remote add origin https://github.com/<your-username>/it-assistant.git
git push -u origin main
```

### 2. Create the service on Render
This project includes a `render.yaml` (Infrastructure-as-Code), so Render can auto-configure everything:

1. Go to https://dashboard.render.com/
2. Click **New +** → **Blueprint**
3. Connect your GitHub account and select the `it-assistant` repo
4. Render reads `render.yaml` and shows the `it-troubleshooting-assistant` web service — click **Apply**

If you'd rather set it up manually instead of using the Blueprint:
1. **New +** → **Web Service** → connect your repo
2. **Runtime**: Python 3
3. **Build Command**: `pip install -r requirements.txt && python ingest.py`
4. **Start Command**: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
5. **Plan**: Free (or Starter for always-on, since the free tier spins down when idle)

### 3. Set your API key
In the Render dashboard, go to your service → **Environment** → **Add Environment Variable**:
```
Key:   ANTHROPIC_API_KEY
Value: your-api-key-here
```
Save — Render will redeploy automatically.

### 4. Done
Render gives you a live URL like `https://it-troubleshooting-assistant.onrender.com`. Open it and the chat UI loads.

**Notes on how this deployment works:**
- The vector database (`chroma_db/`) is rebuilt fresh on every deploy by running `ingest.py` as part of the build step — this keeps things simple since Render's free-tier disk is ephemeral (wiped on redeploy/restart).
- If your knowledge base grows large (thousands of tickets) and rebuilding on every deploy gets slow, add a **persistent disk** in Render (Service → Disks) mounted at `/opt/render/project/src/chroma_db`, and change `ingest.py` to only rebuild if the folder is empty.
- Free-tier Render services spin down after 15 minutes of inactivity and take ~30–60 seconds to wake up on the next request — upgrade to a paid plan if you need it always-on.

## Next steps to scale this up
- Swap the sample CSV for your real ticket history / KB exports
- Add hybrid search (keyword + vector) for exact error-code matches
- Deploy as a Slack/Teams bot instead of (or alongside) Streamlit
- Add a 👍/👎 feedback button on answers to track quality over time
- Swap ChromaDB for Pinecone/Qdrant when your knowledge base grows large