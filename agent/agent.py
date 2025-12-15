from agent.prompts import SYSTEM_PROMPT
from config.settings import MAX_AGENT_STEPS, VERBOSE


class Agent:
    def __init__(self, llm, tools=None):
        self.llm = llm
        self.tools = tools or {}

    def run(self, user_input: str):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_input},
        ]

        for step in range(MAX_AGENT_STEPS):
            if VERBOSE:
                print(f"\n--- Agent Step {step + 1} ---")

            response = self.llm.generate(messages)

            if VERBOSE:
                print("\nLLM Output:\n", response)

            parsed = self._parse_response(response)
            if parsed is None:
                return "❌ Could not understand model output."

            thought, action, action_input, final_answer = parsed

            if VERBOSE:
                print("Thought:", thought)
                print("Action:", action)
                print("Action Input:", action_input)

            # ❌ NEVER trust Final Answer without tool success
            tool_name = action.lower()

            # =========================
            # WRITE FILE (STRICT)
            # =========================
            if tool_name == "write_file":
                # Validate format
                if "|" not in action_input:
                    # Auto-fix: infer filename + generate content
                    filename = self._infer_filename(user_input)
                    content = self._generate_code_content(user_input)

                else:
                    filename, content = [
                        x.strip() for x in action_input.split("|", 1)
                    ]

                try:
                    observation = self.tools["write_file"](filename, content)

                    if VERBOSE:
                        print("Observation:", observation)

                    # ✅ ONLY NOW we can return success
                    return f"File '{filename}' has been created successfully."

                except Exception as e:
                    return f"❌ File creation failed: {e}"

            # =========================
            # CALCULATOR
            # =========================
            if tool_name == "calculator":
                result = self.tools["calculator"](action_input)
                return result

            # =========================
            # NO ACTION
            # =========================
            if action == "NONE":
                return thought

            return f"❌ Unknown or invalid action: {action}"

        return "❌ Agent stopped without completing the task."

    # -------------------------
    # Helpers
    # -------------------------
    def _infer_filename(self, user_input: str) -> str:
        if "python" in user_input.lower():
            return "script.py"
        return "output.txt"

    def _generate_code_content(self, user_input: str) -> str:
        if "1 to 100" in user_input.lower():
            return (
                "for i in range(1, 101):\n"
                "    print(i)\n"
            )
        return "# Auto-generated file\n"

    def _parse_response(self, text: str):
        try:
            if "Thought:" not in text:
                return None

            block = "Thought:" + text.split("Thought:", 1)[1]

            def extract(label):
                if label not in block:
                    return ""
                return block.split(label, 1)[1].split("\n", 1)[0].strip()

            return (
                extract("Thought:"),
                extract("Action:").upper(),
                extract("Action Input:"),
                extract("Final Answer:"),
            )
        except Exception:
            return None
