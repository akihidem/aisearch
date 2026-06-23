import sys
from pathlib import Path

# リポジトリルートを import パスに入れる（どこから pytest を起動しても aisearch を解決）
sys.path.insert(0, str(Path(__file__).resolve().parent))
