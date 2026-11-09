"""一个无需第三方依赖的简单工具型 Agent。运行：python simple_agent.py"""
from datetime import datetime
import ast
import operator


# --- 小工具 ---
def current_time():
    """返回本地当前时间。"""
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def count_words(text):
    """统计文本中的非空白片段数量。"""
    return len(text.split())


def calculate(expression):
    """安全计算只包含数字和基本算术运算的表达式。"""
    operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    def evaluate(node):
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in operators:
            # 限制幂运算，避免意外的大数计算。
            if isinstance(node.op, ast.Pow) and abs(evaluate(node.right)) > 100:
                raise ValueError("幂运算指数不能超过 100")
            return operators[type(node.op)](evaluate(node.left), evaluate(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in operators:
            return operators[type(node.op)](evaluate(node.operand))
        raise ValueError("表达式仅支持数字、括号和 + - * / // % ** 运算")

    return evaluate(ast.parse(expression, mode="eval"))


TOOLS = {
    "time": current_time,
    "count_words": count_words,
    "calculate": calculate,
}


class SimpleAgent:
    """用简单规则选择工具；可在此基础上接入大模型完成工具选择。"""

    def run(self, message):
        text = message.strip()
        lower = text.lower()

        if any(word in text for word in ("时间", "几点")) or "time" in lower:
            return TOOLS["time"]()
        if any(word in text for word in ("计算", "算一下", "算术")) or lower.startswith("calc "):
            expression = text[5:].strip() if lower.startswith("calc ") else text
            for prefix in ("请计算", "计算", "算一下", "算术"):
                expression = expression.replace(prefix, "", 1).strip()
            try:
                return str(TOOLS["calculate"](expression))
            except (ValueError, SyntaxError, ZeroDivisionError, OverflowError) as exc:
                return f"计算失败：{exc}"
        if any(word in text for word in ("字数", "词数", "统计")) or lower.startswith("count "):
            content = text[6:].strip() if lower.startswith("count ") else text
            for prefix in ("统计字数", "统计", "字数", "词数"):
                content = content.replace(prefix, "", 1).strip()
            return f"共 {TOOLS['count_words'](content)} 个空白分隔片段"
        return "我还不会处理这个请求。试试：现在几点？计算 12 * (3 + 4)，或统计字数 hello world"


def main():
    agent = SimpleAgent()
    print("简单 Agent 已启动（输入 q 退出）")
    while True:
        try:
            message = input("你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break
        if message.lower() in {"q", "quit", "exit", "退出"}:
            print("再见！")
            break
        print("Agent：" + agent.run(message))


if __name__ == "__main__":
    main()
