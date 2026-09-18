# soldierAI
lightningAIで攻城戦用のAIを動かしたい！

リプレイ機能

利用可能な世代の一覧を見る
python replay.py --list

最新世代を再生用HTMLにする
python replay.py

特定の世代を指定する
python replay.py --generation 5

$env:RTS_BASE_DIR="$PWD\PPO_RTS"; python replay.py

どちらも実行すると、最後にこのような出力が出ます。
tag_game.py 簡易ヘルプ
学習
# デフォルト：10兵士・100世代
python tag_game.py

# 兵士数を指定
python tag_game.py --soldiers 1

# 世代数を指定
python tag_game.py --updates 1000

# 例：1兵士で1000世代
python tag_game.py --soldiers 1 --updates 1000

--n-soldiers と --tag-updates も同じ意味。

Replay
# 最新のPolicy / Bout
python tag_game.py --replay

# 特定Policy
python tag_game.py --replay --policy 120

# 特定Policy・Bout
python tag_game.py --replay --policy 120 --bout 7
Modal
# Modal → ローカルへ同期
python tag_game.py --sync

# 学習結果をModalへ保存
python tag_game.py --upload

# 同期 → 1000世代学習 → Upload
python tag_game.py --sync --soldiers 1 --updates 1000 --upload
その他
# 古いTag policyを圧縮
python tag_game.py --compact

# コマンド一覧
python tag_game.py --help
現在の進化仕様
8 teams → 4 → 2 → 1
1 matchup = 8 games
各兵士 = 独立NN
次世代のSoldier親 = 勝利team内で最もSoldier-local rewardが高い兵士
Commander kill = 進化用Soldier reward +2.50
壁 = 0～15個
25%以上のCommander kill率 × 3世代 → 壁 +1
25%未満のCommander kill率 × 5世代 → 壁 -1