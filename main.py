from agent.llm import SmolLLM
from agent.agent import Agent
from config.settings import VERBOSE

# Import tools
from tools.calculator import calculator
from tools.filesystem import write_file, read_file, list_files
from tools.web import web_search


def main():
    print("🚀 Starting SmolLM Agent...\n")

    # Initialize LLM
    llm = SmolLLM()

    # Register tools
    tools = {
        "calculator": calculator,
        "write_file": write_file,
        "read_file": read_file,
        "list_files": list_files,
        "web_search": web_search,
    }

    # Initialize Agent with tools
    agent = Agent(llm, tools=tools)

    # Interactive loop
    while True:
        try:
            user_input = input("\nUser > ").strip()

            if user_input.lower() in {"exit", "quit"}:
                print("\n👋 Exiting agent.")
                break

            response = agent.run(user_input)

            if VERBOSE:
                print("\n🤖 Agent Response:")
            print(response)

        except KeyboardInterrupt:
            print("\n👋 Interrupted by user.")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()
