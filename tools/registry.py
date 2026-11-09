from models.llm import model
from tools.add_tool import add
from tools.devide_tool import divide
from tools.multiply_tool import multiply

tools = [add, multiply, divide]

tools_by_name = {
    tool.name: tool for tool in tools
}

model_with_tools = model.bind_tools(tools)