# TODO

### 修正待ち
- phase 1
    - ~~文単位解析と原稿単位解析のフォントの大きさが不一致？~~
    - 上位原稿解析中にプログレスバー追加
    - ~~上位原稿群は事前にデータ計算~~
- phase 2
    - entity: 既存読み込み、新規追加
    - entity: demo用に1000個限定（DB連携待ち）
    - entity: 計算にプログレスバー追加

### 最優先
- ~~multipage app 化~~
- auroraDB
- ~~sagemaker で preprocessing~~
- ~~smartphone preview page~~
- ~~50k words に制限~~
- phase0 on_click change -> flag control
- ~~400k word / keyword はだいたい 90 分~~
- ~~freq >= 5: 18600 words~~

### cache系
- multipage と session_state でだいたい解決 
- ~~session~~
- ~~upload ファイル反映するには memo, singleton は使えないが、計算わりと遅い~~
- ~~なんで memo か singleton をつけて、upload ファイルのデータをクリアできるモジュールをつける~~
- ~~データはたぶん週一ペースで裏で計算したほうがいい（イレギュラーはその場で計算なので、incheck~~
- ~~既存ファイルの読み込みか、その場で計算かの判断はsession_state~~

### データセット形成
- 業界ごとのデータ収集
- 無料上位の区分

### 基本 algorithm と効果向上させるための algorithm との区別
- advanced: 計算においていろいろいじれる所あるけど、今は実装しない
- chitra: bert の last hidden layer 使えるけど、今は実装しない
- ~~chive の full version を読み込むには 12GB 以上のメモリが必要なので、今は実装しない~~