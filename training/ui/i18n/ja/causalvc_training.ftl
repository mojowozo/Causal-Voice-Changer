app-title = CausalVC トレーナー

page-voice = 新しい声
page-builds = ビルド
page-log = ログ

service-starting = 起動しています…
service-starting-detail = 学習ツールを読み込んでいます。
service-unavailable = トレーナーを起動できませんでした
service-retrying = 数秒後にもう一度試します。原因は下のメッセージを確認してください。

section-dataset = データセット
audio = 録音
audio-detail = 目標の声の録音フォルダー、または長い録音 1 つ。形式やサンプルレートは問わず、無音部分もそのままで構いません。
audio-none = まだ選択されていません
choose-folder = フォルダー…
choose-file = ファイル…
pick-audio-folder = 録音フォルダーを選択
pick-audio-file = 録音を選択
voice-name = 声の名前
voice-name-detail = 英数字、'-'、'_'、'.' が使えます。同じ名前でもう一度ビルドすると続きから再開します。

section-training = 学習
steps = ステップ数
steps-detail = 学習の長さです。まずは 10,000 がおすすめで、増やすほど声に近づきます。
save-every = 保存間隔
save-every-detail = 保存するバージョンの間隔（NAME-1000s、NAME-2000s など）。比べて一番良いものを残せます。
batch = バッチサイズ
batch-detail = 一度に学習するクリップ数です。GPU のメモリが足りない場合は下げてください。
fixes = 品質の改善
fixes-detail = 自然な息づかいとささやき、安定して明るい声質、元の声の混じりを減らします。約 25% 遅くなります。
restart = 最初からやり直す
restart-detail = この名前の前回の進行を無視して最初から学習します。次のビルドだけに適用されます。

section-hardware = 出発点とハードウェア
base = ベース
base-detail = 学習の出発点となる音声モデルです。自分のベースは custom-base フォルダーに置きます。
base-default = 既定
gpu = GPU
gpu-detail = 声を学習するグラフィックカードです。
gpu-loading = GPU を探しています…
gpu-cpu = GPU が見つかりません（非常に遅くなります）

section-install = ボイスチェンジャー
install = ボイスチェンジャーに追加
install-detail = 保存されたバージョンは、書き出されるたびにボイスチェンジャーのライブラリに表示されます。
install-folder = 声のフォルダー
pick-install-folder = ボイスチェンジャーの声のフォルダーを選択
change = 変更…

start = 学習を開始
queue-build = キューに追加
stop = 停止
end-now = 今すぐ終了
stop-detail = 停止するとまずチェックポイントを保存します。同じ声をもう一度ビルドすると再開します。もう一度クリックすると保存せずにすぐ終了します。

phase-idle = 準備ができました。声の情報を入力して学習を開始してください。
phase-starting = { $name } を開始しています（モデルの読み込みに少し時間がかかります）…
phase-training = { $name } を学習中
phase-stopping = { $name } を停止中: まずチェックポイントを保存しています…
phase-finished = { $name } が完了しました
phase-stopped = { $name } を停止しました。もう一度ビルドすると続きから再開します
phase-failed = { $name } が失敗しました（終了コード { $code }）。原因はログを確認してください
progress = { $total } ステップ中 { $step }
progress-times = 経過 { $elapsed } · 残り { $remaining } · { $rate } ステップ/秒

builds-current = 実行中
builds-none = 学習中のものはありません。
builds-queue = 次のビルド
builds-queue-empty = キューは空です。「キューに追加」で複数の声を順番に学習できます。
builds-steps = { $steps } ステップ
remove = 削除
close-note = このウィンドウを閉じると、停止と同じようにチェックポイントを保存してから学習を止めます。

log-empty = 学習のメッセージがここに表示されます。
clear-log = クリア

error-invalid_name = 声の名前には英数字、'-'、'_'、'.' を使ってください。
error-missing_audio = 録音のフォルダーまたはファイルが見つかりません。
error-invalid_numbers = ステップ数、保存間隔、バッチサイズは 1 以上の整数にしてください。
error-busy = すでに別の声を学習中です。こちらはキューに追加してください。
error-other = { $message }
