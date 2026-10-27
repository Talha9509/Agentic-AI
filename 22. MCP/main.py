from mcp.server.mcpserver import MCPServer

mcp = MCPServer("my_mcp")

@mcp.tool()
def add(a: int, b: int) -> int:
  """Add 2 numbers using calculator"""
  return a+b

if __name__ == "__main__":
  mcp.run()
  
