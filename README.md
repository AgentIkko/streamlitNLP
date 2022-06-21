# TODO

### 最優先
- multipage app 化

### cache系
- upload ファイル反映するには memo, singleton は使えないが、計算わりと遅い
- なんで memo か singleton をつけて、upload ファイルのデータをクリアできるモジュールをつける
- データはたぶん週一ペースで裏で計算したほうがいい（イレギュラーはその場で計算なので、incheck）
- 既存ファイルの読み込みか、その場で計算かの判断はsession_state

### データセット形成
- 業界ごとのデータ収集
- 無料上位の区分

### 基本 algorithm と効果向上させるための algorithm との区別
- advanced: 計算においていろいろいじれる所あるけど、今は実装しない