# aisearch

**複数 LLM の合議生成 × 再帰的自己改善ループ × モデル/設定のメタ探索**で、
*最も優秀な生成物を出す AI 設定* を自動で探索する Python パッケージ。

「1個の LLM に1回投げる」のではなく、

1. **合議 (council)** — 複数の役割を持つ提案者が案を出し、相互批評して統合する
2. **再帰的自己改善 (refine)** — その成果物を judge で採点し、批評→改稿を改善が止まるまで反復する
3. **メタ探索 (search)** — どのモデル/温度/役割構成/合議サイズが最良かを進化的に探索する

の3層を重ねて、`(best_config, best_artifact, best_score)` を返す。

```
┌─ meta-search ──────────────────────────────────────────────┐
│  Config 母集団（model / temperature / roles / size …）を     │
│  進化的に探索し、各候補を ↓ で評価してスコア最大を選ぶ        │
│                                                            │
│   ┌─ refine（再帰的自己改善）─────────────────────────┐      │
│   │  loop:  judge 採点 → 自己批評 → 改稿              │      │
│   │         （plateau / max_iters / budget で停止）   │      │
│   │   ┌─ council（合議）──────────────────────────┐   │      │
│   │   │  propose → critique → aggregate          │   │      │
│   │   │  （proposer ごとに別 role を巡回割当）     │   │      │
│   │   └──────────────────────────────────────────┘   │      │
│   └───────────────────────────────────────────────────┘      │
└────────────────────────────────────────────────────────────┘
                     ▼
        best (Config, artifact, score)
```

## 設計の要：実 LLM 本番 × 決定的テスト

生成器に実 LLM、fitness に LLM-as-judge を使う（どちらも非決定的）一方で、
**ロジックのテスト (L0) は完全に決定的**に保つ。両立の仕組みは:

- すべてのコンポーネントが `LLMClient` / `Judge` プロトコルを**注入**で受け取る。
  本番は `ClaudeClient` / `OllamaClient` / `MLXClient` と `LLMJudge`、
  テストは `FakeLLM` / `FakeJudge`（同一入力→同一出力）を注入する。
- 既定のテストは API を一切叩かない。`tests/test_l0_path_makes_no_network` が
  ソケットを遮断して**通信ゼロ**を実証している。実 API 挙動は
  `pytest -m integration`（手動・課金あり）に隔離。
- **LLM-judge の非決定性は seed 固定 + N 票 + 中央値集約**で再現可能に丸める。
  refine は固定 seed で「動かない物差し」にし、plateau 判定が seed ノイズで誤発火しない。
- **メタ探索は評価を Config 単位でキャッシュ**する。エリートの再評価による下振れを防ぎ、
  noisy/実 LLM evaluator でも世代ベストが単調非減少になり、再評価コストも払わない。

## インストール

```bash
git clone git@github.com:akihidem/aisearch.git
cd aisearch
# L0（テスト/デモ）は依存ゼロ。Python 3.10+
# 実 Claude バックエンドを使う時だけ:
pip install anthropic
```

## 使い方

### 1. デモ（API 不要・決定的）

```bash
python -m aisearch.search --demo --seed 0 --out best.json
# → best.json に best_config / best_artifact / best_score / score_history
```

### 2. 実 LLM で探索

```python
from aisearch import ClaudeClient, LLMJudge, SearchSpace, search, make_refine_evaluator

task = "Write a short, vivid haiku about recursion."

client = ClaudeClient(model="claude-opus-4-8")   # or OllamaClient(), MLXClient()
judge  = LLMJudge(client, votes=3)               # seed固定+N票+中央値
evaluator = make_refine_evaluator(task, client, judge)

result = search(task, SearchSpace(), evaluator,
                generations=5, pop_size=6, seed=0)

print(result.best_score, result.best_config)
print(result.best_artifact)
```

### 3. 個別の層を直接使う

```python
from aisearch import Config, FakeLLM, FakeJudge, generate, refine

cfg = Config(model="claude-opus-4-8", council_size=3,
             roles=("generalist", "critic", "contrarian"), max_iters=3)

council = generate(task, cfg, client)                 # 合議だけ
result  = refine(task, cfg, client, judge)            # 合議→自己改善ループ
```

## 自分のバックエンドを足す

`LLMClient` プロトコル（`complete(prompt, *, temperature, seed) -> LLMResponse`）を
満たすクラスを書いて注入するだけ。`Judge`（`score(task, artifact, *, seed) -> Judgement`）も同様。

## モジュール

| module | 役割 |
|---|---|
| `aisearch.clients` | `LLMClient` プロトコル + Claude/Ollama/MLX アダプタ + `FakeLLM` |
| `aisearch.judge`   | `Judge` プロトコル + `FakeJudge` + `LLMJudge`（seed/N票/中央値） |
| `aisearch.config`  | `Config` / `SearchSpace`（役割ロスター含む）/ `CostTracker` / seed |
| `aisearch.council` | `generate()` — propose → critique → aggregate（障害フォールバック・予算打ち切り） |
| `aisearch.refine`  | `refine()` — 採点→改稿ループ（plateau / max_iters / budget で停止） |
| `aisearch.search`  | `search()` — 進化的メタ探索 + 評価キャッシュ + CLI |

## テスト

```bash
python -m pytest -q -m "not integration"   # L0: 決定的・API不要（48 tests）
python -m pytest -q -m integration         # 実API smoke（ANTHROPIC_API_KEY 要・課金）
```

## 開発手法

合格基準を凍結 → 各反復で「実装 → 決定的チェック(L0) → 敵対的 validator 検品 → 欠陥のみ修正」を回し、
**builder（実装）と checker（検品）を分離**して Goodhart を断つ自己検証ループで作っている。

## ステータス / ロードマップ

- ✅ 4 層（foundation / council / refine / search）+ 役割多様化、48 テスト緑
- ⏳ 実 API smoke の実走（鍵設定）
- ⏳ メタ探索の bandit 化 / roster 長と council_size の連動 / diversity-aware fitness

## ライセンス

未設定（必要なら追加）。
