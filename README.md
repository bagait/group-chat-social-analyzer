# Group Chat Social Dynamics Analyzer

This project provides anonymized sociological insights into group chat conversations. It uses a local Large Language Model (LLM) to classify messages into conversational roles and generates visualizations of user participation and communication flow, all without storing or sending your private conversations to the cloud.

![Role Distribution Chart](role_distribution.png)
![Interaction Graph](interaction_graph.png)

## Features

- **Privacy-First Analysis**: Uses a local LLM via Ollama, ensuring your chat data never leaves your machine.
- **Conversational Role Classification**: Identifies who is playing what role in a conversation (e.g., 'Initiator', 'Tension-Breaker', 'Summarizer').
- **Role Distribution Visualization**: Generates a stacked bar chart showing the percentage of each conversational role per user.
- **Interaction Network Mapping**: Creates a directed graph to visualize who talks to whom most frequently.
- **Anonymized Insights**: Focuses on metadata and patterns, not the content of the messages.

## How It Works

The analysis is a two-step process:

1.  **Role Analysis**: The script iterates through each message in a chat log. For each message, it sends a request to a locally running Ollama instance. A specially crafted prompt asks the LLM to classify the message into one of several predefined sociological roles (e.g., `Initiator`, `Supporter`, `Challenger`). The results are aggregated for each user.

2.  **Network Analysis**: The script builds a communication graph based on a simple temporal heuristic: a message is considered a reply to the one immediately preceding it. This helps map the flow of conversation. The `networkx` library is used to build and visualize this graph, where nodes are users (sized by message count) and edges represent the frequency of interaction.

## Installation

### 1. Install Ollama

First, you need a local LLM server. This project is built with [Ollama](https://ollama.com/).

1.  Download and install Ollama for your operating system.
2.  Pull a small, fast model to handle the classification task. `phi3` is recommended.

    bash
    ollama run phi3
    

    Once the model is downloaded, you can exit the Ollama chat prompt.

### 2. Clone and Install Project Dependencies

Now, set up the Python environment for this project.

bash
# Clone the repository
git clone https://github.com/bagait/group-chat-social-analyzer.git

# Navigate into the project directory
cd group-chat-social-analyzer

# Install the required Python packages
pip install -r requirements.txt


## Usage

Make sure your Ollama application is running in the background. Then, simply run the main script:

bash
python main.py


The script will start processing the simulated chat data in `main.py`. You will see the classification progress in your terminal. Once complete, it will save two images (`role_distribution.png` and `interaction_graph.png`) to the project directory and display them on your screen.

### Analyzing Your Own Data

To analyze your own chat data, you can modify the `get_simulated_chat_log()` function in `main.py`. Ensure your data follows the same format: a list of dictionaries, where each dictionary has a `'user'` and a `'message'` key.

python
# Example of the required data structure in main.py
def get_simulated_chat_log():
    return [
        {'user': 'YourFriend1', 'message': 'This is the first message.'},
        {'user': 'You', 'message': 'This is your reply.'},
        # ... and so on
    ]


**Disclaimer**: This tool is for educational and illustrative purposes. The sociological classifications are based on the interpretations of the LLM and a simplified interaction model.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
