# 单agent，接入高德地图MCP，实现自驾游规划，包括路径规划、交通状况分析、饮食，天气等兴趣点推荐等功能。
import numpy as np
import math
import matplotlib.pyplot as plt
from typing import List, Tuple
from mcp import MCPClient  # 假设有一个MCP客户端库  