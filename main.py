import requests
import json
from collections import Counter, defaultdict
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# --- Configuration ---
OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "phi3" # Recommended: phi3, llama3, or other small, fast models

# --- Conversational Roles Definition ---
CONVERSATIONAL_ROLES = {
    "Initiator/Question-Asker": "Starts new topics, asks questions to drive the conversation forward, or introduces new ideas.",
    "Information-Giver": "Provides facts, data, detailed explanations, or expert knowledge.",
    "Tension-Breaker/Humorist": "Uses jokes, lighthearted comments, or humor to ease tension and maintain a positive atmosphere.",
    "Opinion-Giver/Elaborator": "Shares personal beliefs, expands on others' ideas, or provides their own perspective.",
    "Supporter/Encourager": "Shows agreement, offers praise, or provides positive reinforcement to others.",
    "Summarizer/Clarifier": "Recaps the conversation, pulls together different threads, or seeks to clarify ambiguous points.",
    "Challenger/Devil's Advocate": "Questions assumptions, points out potential flaws, or plays the devil's advocate to test ideas."
}

# --- Simulated Chat Data ---
def get_simulated_chat_log():
    """Provides a sample chat log for demonstration purposes."""
    return [
        {'user': 'Alice', 'message': "Hey everyone, what do you think about the new project proposal for Q3? I'm not sure where to start."},
        {'user': 'Bob', 'message': "Good question, Alice. I read through it. The proposal requires a 15% increase in resource allocation for the data science team."},
        {'user': 'Charlie', 'message': "Whoa, 15%? That sounds like a lot! Are we sure that's feasible?"},
        {'user': 'Alice', 'message': "That's what I was thinking. It feels a bit steep."}, 
        {'user': 'Dana', 'message': "Haha, remember last year when we thought a 5% increase was the end of the world? Good times."},
        {'user': 'Bob', 'message': "True, Dana. To be fair, the proposal justifies it with projected ROI of 40% by year-end, based on the market analysis on page 12."},
        {'user': 'Eve', 'message': "Great point, Bob. Your attention to detail is always so helpful! I feel more confident about it now."},
        {'user': 'Alice', 'message': "Okay, that context helps. So, we have a high resource cost but also a high potential reward."},
        {'user': 'Charlie', 'message': "I'm still skeptical. What if those market projections are too optimistic? We've been burned by that before."},
        {'user': 'Dana', 'message': "Let's not get stuck in the past. I think it's a bold move, and sometimes we need that! Let's be innovative!"},
        {'user': 'Eve', 'message': "I agree with Dana. It's an exciting opportunity!"},
        {'user': 'Bob', 'message': "So, to summarize: Alice raised the concern, I provided the key numbers, Charlie is playing devil's advocate on the projections, Dana and Eve are providing encouragement, and we all agree it's a major decision."},
        {'user': 'Alice', 'message': "Perfect summary, Bob. Thanks! This makes the path forward much clearer."},
    ]

# --- Core Logic ---
def check_ollama_status():
    """Checks if the Ollama server is running and the model is available."""
    try:
        response = requests.post(OLLAMA_ENDPOINT, json={"model": OLLAMA_MODEL, "prompt": "Hello"}, stream=True, timeout=5)
        response.raise_for_status()
        # Consume the stream to avoid errors
        for _ in response.iter_lines():
            pass
        return True
    except (requests.exceptions.RequestException, requests.exceptions.HTTPError) as e:
        print(f"\n[ERROR] Could not connect to Ollama at {OLLAMA_ENDPOINT.split('/api')[0]}.")
        print("Please ensure Ollama is running and you have pulled the model:")
        print(f"ollama run {OLLAMA_MODEL}\n")
        return False

def classify_message_role(message):
    """Uses a local LLM via Ollama to classify a message into a conversational role."""
    roles_text = "\n".join([f"- {role}: {desc}" for role, desc in CONVERSATIONAL_ROLES.items()])
    prompt = f"""
You are a sociologist analyzing a group chat. Your task is to classify the following message into ONE of the predefined conversational roles.

Here are the roles:
{roles_text}

Message: "{message}"

Based on the message content, which single role does the sender best fit? Respond with ONLY the role name (e.g., 'Initiator/Question-Asker').

Role:
"""
    
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0
        }
    }
    
    try:
        response = requests.post(OLLAMA_ENDPOINT, json=payload)
        response.raise_for_status()
        result_json = response.json()
        role = result_json.get('response', '').strip()
        # Clean up potential LLM verbosity
        for r in CONVERSATIONAL_ROLES.keys():
            if r in role:
                return r
        return "Unknown" # Default if parsing fails
    except requests.exceptions.RequestException as e:
        print(f"Error communicating with Ollama: {e}")
        return "Unknown"

def analyze_chat_dynamics(chat_log):
    """Analyzes the chat log for roles and interactions."""
    user_roles = defaultdict(Counter)
    user_message_count = Counter()
    interactions = defaultdict(int)
    
    print(f"Analyzing {len(chat_log)} messages...")
    for i, entry in enumerate(chat_log):
        user = entry['user']
        message = entry['message']
        
        role = classify_message_role(message)
        print(f"  - User '{user}': Role '{role}'")
        
        if role != "Unknown":
            user_roles[user][role] += 1
        user_message_count[user] += 1
        
        # Simple interaction tracking: a message is a reply to the previous one
        if i > 0:
            prev_user = chat_log[i-1]['user']
            if user != prev_user:
                interactions[(prev_user, user)] += 1
                
    return user_roles, user_message_count, interactions

# --- Visualization ---
def plot_role_distribution(user_roles):
    """Creates a stacked bar chart of role distributions for each user."""
    users = list(user_roles.keys())
    roles = list(CONVERSATIONAL_ROLES.keys())
    
    role_counts = {role: [user_roles[user].get(role, 0) for user in users] for role in roles}
    
    # Normalize to percentages
    totals = [sum(counts) for counts in zip(*role_counts.values())]
    role_percentages = {role: [(count / total * 100 if total > 0 else 0) for count, total in zip(counts, totals)] for role, counts in role_counts.items()}

    fig, ax = plt.subplots(figsize=(12, 8))
    bottom = np.zeros(len(users))
    
    colors = plt.cm.get_cmap('tab20', len(roles))

    for i, (role, percentages) in enumerate(role_percentages.items()):
        ax.bar(users, percentages, label=role, bottom=bottom, color=colors(i))
        bottom += np.array(percentages)
        
    ax.set_title('Conversational Role Distribution per User', fontsize=16)
    ax.set_ylabel('Percentage of Messages (%)')
    ax.set_xlabel('User')
    ax.legend(title='Conversational Roles', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    fig.savefig("role_distribution.png")
    print("\nSaved role distribution plot to role_distribution.png")

def plot_interaction_graph(interactions, user_message_count):
    """Creates a directed graph of user interactions."""
    G = nx.DiGraph()
    
    for (sender, receiver), weight in interactions.items():
        G.add_edge(sender, receiver, weight=weight)
        
    node_sizes = [user_message_count[node] * 200 for node in G.nodes()]
    edge_widths = [G[u][v]['weight'] * 1.5 for u, v in G.edges()]
    
    pos = nx.spring_layout(G, k=0.9, iterations=50, seed=42)
    
    fig, ax = plt.subplots(figsize=(12, 12))
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='skyblue', alpha=0.9, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=10, ax=ax)
    nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color='gray', alpha=0.6, 
                           arrowstyle='->', arrowsize=20, ax=ax)
    
    ax.set_title('Group Communication Flow', fontsize=16)
    plt.axis('off')
    fig.savefig("interaction_graph.png")
    print("Saved interaction graph to interaction_graph.png")


if __name__ == "__main__":
    print("--- Group Chat Social Dynamics Analyzer ---")
    if not check_ollama_status():
        exit(1)
        
    chat_data = get_simulated_chat_log()
    user_roles, user_message_count, interactions = analyze_chat_dynamics(chat_data)
    
    if not user_roles:
        print("No data to analyze. Exiting.")
        exit()
        
    plot_role_distribution(user_roles)
    plot_interaction_graph(interactions, user_message_count)
    
    print("\nAnalysis complete. Opening plots...")
    plt.show()
