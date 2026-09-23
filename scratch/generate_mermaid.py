import yaml
import os

def generate_mermaid():
    if not os.path.exists("config/workflow.yaml"):
        print("workflow.yaml not found.")
        return
        
    with open("config/workflow.yaml", "r") as f:
        config = yaml.safe_load(f)

    mermaid_code = ["graph TD", "    %% Triggers"]
    for t in config.get("triggers", []):
        t_id = f"trigger_{t['type']}"
        mermaid_code.append(f"    {t_id}([{t['type'].upper()} TRIGGER])")

    mermaid_code.append("    %% Nodes")
    for node in config.get("nodes", []):
        node_id = node["id"]
        label = f"{node_id.replace('_', ' ').title()}\\n({node['module'].split('.')[-1]})"
        mermaid_code.append(f"    {node_id}[\"{label}\"]")
        
        # Connect triggers to the first node (if no dependencies)
        if not node.get("depends_on"):
            for t in config.get("triggers", []):
                t_id = f"trigger_{t['type']}"
                mermaid_code.append(f"    {t_id} --> {node_id}")

        for dep in node.get("depends_on", []):
            mermaid_code.append(f"    {dep} --> {node_id}")

    # Add DB/Broker visual nodes
    mermaid_code.append("    subgraph External Interfaces")
    mermaid_code.append("        broker[(Fyers Broker)]")
    mermaid_code.append("        db[(SQLite Database)]")
    mermaid_code.append("        llm{{LLM API (Gemini/OpenAI)}}")
    mermaid_code.append("    end")
    
    mermaid_code.append("    broker --> data_collection")
    mermaid_code.append("    db --> data_collection")
    mermaid_code.append("    data_collection -.-> db")
    mermaid_code.append("    ai_analysis <--> llm")

    print("\n".join(mermaid_code))

if __name__ == "__main__":
    generate_mermaid()
