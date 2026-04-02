
# 🔥 AI Car Damage Evaluation

![pillow](https://img.shields.io/badge/pillow-12.0.0-green)
![requests](https://img.shields.io/badge/requests-2.32.5-yellow)
![gradio](https://img.shields.io/badge/gradio-6.0.1-blue)

# Car Damage with local evaluation and body shops search**

**Run locally**

`
clone the repository

git clone https://github.com/claudio1975/AI_Car_Damage_Evaluation.git

-create the environment

python -m venv venv

-activate the environment

venv\Scripts\activate

-install requirements

pip install -r requirements.txt # for streamlit app

pip install -r requirements - gradio.txt # for gradio app

-launch the app

python app_gradio_multiagent.py # gradio app

streamlit run app_streamlit_multiagent.py # streamlit app
`

**Run on web**

go to hugging face app: -> https://huggingface.co/spaces/towardsinnovationlab/AI_Car_Damage_Evaluation

go to streamlit app -> https://aicardamageevaluation.streamlit.app/

```
┌─────────────────────────────────────────────────────────────────────┐
│                    GRADIO WEB INTERFACE                             │
│                                                                     │
│  [Photo Upload]  [Location Input]  [API Keys]  [Analyze Button]     |
│                                                                     │
│  Outputs: Damage Description │ OEM Quote │ Aftermarket Quote │ Shops│
└──────────────────────────┬──────────────────────────────────────────┘
                           │ analyze_with_multi_agent_system()
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     ORCHESTRATOR AGENT                              │
│                                                                     │
│  1. orchestrator_plan()  →  builds execution plan (priority queue)  │
│  2. orchestrator_execute() → coordinates agents in priority order   │
│  3. Aggregates & returns results to UI                              │
└────┬──────────────────┬──────────────────────┬──────────────────────┘
     │ priority 1       │ priority 2            │ priority 3
     ▼                  ▼                       ▼
┌──────────┐   ┌────────────────────────────┐  ┌──────────────────────┐
│  VISION  │   │      COST AGENTS (x2)      │  │   SHOP FINDER AGENT  │
│  AGENT   │   │                            │  │                      │
│          │   │ ┌──────────┐ ┌──────────┐  │  │  shop_finder_search()│
│ perceive │   │ │  OEM /   │ │Aftermkt/ │  │  │                      │
│  image   │   │ │ Dealer   │ │Independ. │  │  │  Needs: location     │
│          │   │ └────┬─────┘ └────┬─────┘  │  │  Skips if no loc.    |
│ Detects  │   │      │            │        │  └──────────┬───────────┘
│ severity │   │ cost_agent_estimate()      │             │
│ (auto)   │   └────────────────────────────┘             │
└────┬─────┘            │            │                    │
     │                  │            │                    │
     │  damage_info     │            │                    │
     │ (desc+severity)  │            │                    │
     └──────────────────┘            │                    │
                                     │                    │
┌────────────────────────────────────▼────────────────────▼──────────┐
│                        EXTERNAL APIs                               │
│                                                                    │
│  ┌──────────────────────────┐    ┌──────────────────────────────┐  │
│  │  OpenAI API              │    │  Perplexity API              │  │
│  │  model: gpt-5.2          │    │  model: sonar-pro            │  │
│  │  (Vision Agent only)     │    │  (Cost Agents + Shop Finder) │  │
│  │  → Image analysis        │    │  → Cost estimates (JSON)     │  │
│  │  → Damage description    │    │  → Local shop search         │  │
│  └──────────────────────────┘    └──────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘

```
