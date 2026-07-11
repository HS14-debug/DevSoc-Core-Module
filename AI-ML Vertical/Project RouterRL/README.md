Hi!

About the Project :

RouterRL is trained to select an appropriate policy for sending each turn of conversation to one out of four candidate LLMs, depending on a limited budget of tokens.

Tools Used :

-> Data Source : HuggingFace (JiaqiXue/mmr-routing-20k)

-> RL environment : custom Gymnasium environment (RouterEnv)

-> RL Agent : DQN via Stable-Baselines3

Data Tools : pandas, huggingface_hub

Workflow :

-> build_dataset.py : Merges HF dataset into a parquet routing table.

-> eda.py : Confirms problem difficulty and model cost/quality spreads.

-> router_env.py : Custom Gym environment (60-step episodes, 10k token budget, cost-aware reward).

-> train_dqn.py : Trains SB3 DQN against basic fixed-action baselines.

-> evaluate_heuristic.py : Evaluation of DQN against a rule-based difficulty router.
