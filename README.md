# A simple CNN/ViT training code

CNN/ViT を使って学習する単純な練習用コードです．

## 準備

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.pytorch.txt
pip install -r requirements.txt

```

## 使い方

```bash
python3 main.py -w 24 -b 8 -e 5 -d ImageFolder -r /mnt/NAS-TVS872XT/dataset-lab/Tiny-ImageNet/  --use_dp
python3 main_pl.py -w 24 -b 8 -e 5 -d ImageFolder -r /mnt/NAS-TVS872XT/dataset-lab/Tiny-ImageNet/ --devices 3
```

`_pl`がついたファイルは Pytorch lightning のコードを使用（ddp のみ対応）．

### multi-GPU 学習

GPU の指定には`CUDA_VISIBLE_DEVICES`を使用すること．

- dp (data parallel) は`main.py`で利用可能
- ddp (distributed data parallel)は lightning の`main_pl.py`で利用可能
  - 注意：複数 GPU を用いる dp や ddp が動作しなくなるため，コード内で GPU 番号を指定するような`torch.device("cuda:0")`は**使わない**．dp や ddp のために，コード内では`torch.device("cuda")`としておく．

```bash
CUDA_VISIBLE_DEVICES=0,1 python3 main.py -w 24 -b 8 -e 5 -d ImageFolder -r /mnt/NAS-TVS872XT/dataset-lab/Tiny-ImageNet/  --use_dp
CUDA_VISIBLE_DEVICES=0,1 python3 main_pl.py -w 24 -b 8 -e 5 -d ImageFolder -r /mnt/NAS-TVS872XT/dataset-lab/Tiny-ImageNet/ --devices 3
```

デバッグ用には [launch.json](.vscode/launch.json) を以下のように設定する．

```json
            "env": {
                "CUDA_VISIBLE_DEVICES": "0,1",
            },
```

task 用には[tasks.json](.vscode/tasks.json)に次のように設定する．

```json
            "options": {
                "env": {
                    "CUDA_VISIBLE_DEVICES": "0,1",
                },
            },
```

### option

詳しくは`args.py`を参照．主なオプションは以下の通り．

- `-r`：データセットの root フォルダ
- `-b`：バッチサイズ
- `-w`：データローダーのワーカー数
- `-e`：エポック数
- `-d`：データセット
  - `CIFAR10`：[torchvision の CIFAR10](https://pytorch.org/vision/main/generated/torchvision.datasets.CIFAR10.html)
  - `ImageFolder`：`-r`で指定したフォルダ以下に`train/`と`val/`のディレクトリがあり，それ以下はカテゴリ名のサブディレクトリに分かれて保存されている画像データセット（[torchvision の ImageFolder](https://pytorch.org/vision/main/generated/torchvision.datasets.ImageFolder.html)）
- `--use_dp`：dp (Data Parallel)で複数 GPU を使用する（lightning ではない場合）
- `--devices`: lightning の ddp で使用する GPU 番号（-1 は全 GPU を使用）

#### help

```bash
python3 main.py -h
python3 main_pl.py -h
```

を実行すると以下が表示される．

```text
usage: main.py [-h] [-r str] [-d {CIFAR10,ImageFolder,VideoFolder,ZeroImages}] [-td str] [-vd str] [--torch_home str]
               [-m {resnet18,resnet50,x3d,abn_r50,vit_b,zero_output_dummy}] [--use_pretrained] [--scratch] [--frames_per_clip int]
               [--clip_duration float] [--clips_per_video int] [-b int] [-w int] [-e int] [-vi int] [-li int] [--optimizer_name {SGD,Adam}]
               [--grad_accum int] [-lr float] [--momentum float] [--weight_decay float] [--use_scheduler] [--no_scheduler] [--use_dp]
               [--devices str] [--comet_log_dir str] [--tf_log_dir str] [--save_checkpoint_dir str] [--checkpoint_to_resume str] [--disable_comet]

simple image/video classification

options:
  -h, --help            show this help message and exit
  -r str, --root str    root of dataset. (default: ./downloaded_data)
  -d {CIFAR10,ImageFolder,VideoFolder,ZeroImages}, --dataset_name {CIFAR10,ImageFolder,VideoFolder,ZeroImages}
                        name of dataset. (default: CIFAR10)
  -td str, --train_dir str
                        subdier name of training dataset. (default: train)
  -vd str, --val_dir str
                        subdier name of validation dataset. (default: val)
  --torch_home str      TORCH_HOME environment variable where pre-trained model weights are stored. (default: ./pretrained_models)
  -m {resnet18,resnet50,x3d,abn_r50,vit_b,zero_output_dummy}, --model_name {resnet18,resnet50,x3d,abn_r50,vit_b,zero_output_dummy}
                        name of the model (default: resnet18)
  --use_pretrained      use pretrained model weights (default) (default: True)
  --scratch             do not use pretrained model weights, instead train from scratch (not default) (default: True)
  --frames_per_clip int
                        frames per clip. (default: 16)
  --clip_duration float
                        duration of a clip (in second). (default: 2.6666666666666665)
  --clips_per_video int
                        sampling clips per video for validation (default: 1)
  -b int, --batch_size int
                        batch size. (default: 8)
  -w int, --num_workers int
                        number of workers. (default: 2)
  -e int, --num_epochs int
                        number of epochs. (default: 25)
  -vi int, --val_interval_epochs int
                        validation interval in epochs. (default: 1)
  -li int, --log_interval_steps int
                        logging interval in steps. (default: 1)
  --optimizer_name {SGD,Adam}
                        optimizer name. (default: SGD)
  --grad_accum int      steps to accumlate gradients. (default: 1)
  -lr float             learning rate. (default: 0.0001)
  --momentum float      momentum of SGD. (default: 0.9)
  --weight_decay float  weight decay. (default: 0.0005)
  --use_scheduler       use scheduler (not default) (default: False)
  --no_scheduler        do not use scheduler (default) (default: False)
  --use_dp              GPUs with data parallel (dp); not for lightning (default: False)
  --devices str         GPUs used for ddp strategy (only for lightning). '-1' for all gpus. (default: 1)
  --comet_log_dir str   dir to comet log files. (default: ./comet_logs/)
  --tf_log_dir str      dir to TensorBoard log files. (default: ./tf_logs/)
  --save_checkpoint_dir str
                        dir to save checkpoint files. (default: ./log)
  --checkpoint_to_resume str
                        path to the checkpoint file to resume from. (default: None)
  --disable_comet, --no_comet
                        do not use comet.ml (default: use comet) (default: False)
```

## Comet の設定

comet の設定は，

- このディレクトリの`./.comet.config`と，
- ホームの`~/.comet.config`

の 2 つのファイルを利用する．詳しくは[comet のドキュメント](https://www.comet.com/docs/v2/api-and-sdk/python-sdk/advanced/configuration/)を参照．コード中には API キーなどは書かないこと（[logger.py](./logger/logger.py)参照）．

### ホームでの全体設定

- `~/.comet.config`：すべてに共通する設定を書く．
  - comet の API キー，デフォルトの comet workspace を設定．
  - `hide_api_key`は True にすること（しないとログに API キーが残ってしまう）

```ini
[comet]
api_key=XXXXXHereIsYourAPIKeyXXXXXXXX
workspace=tttamaki

[comet_logging]
hide_api_key=True
```

### フォルダごとの設定

- このディレクトリの`./.comet.config`：このディレクトリで使用する設定を書く．
  - comet project name を設定．
  - （ここで設定する内容はホームの`~/.comet.config`よりも優先されて，上書きされる）

```ini
[comet]
project_name=simple_cnn_20230309

[comet_logging]
display_summary_level=0
file=comet_logs/comet_{project}_{datetime}.log

[comet_auto_log]
env_details=True
env_gpu=True
env_host=True
env_cpu=True
cli_arguments=True
```

### コード内での設定

- コード
  - `Experiment`オブジェクトに comet experiment name を設定．必要なら tag を設定する．
  - コード中には **API キーなどは書かない**．
  - （コード中で設定する内容は，ディレクトリごとの`./.comet.config`よりも優先される）

```python
    experiment = Experiment()  # ここでは何も設定しない

    exp_name = datetime.now().strftime('%Y-%m-%d_%H:%M:%S:%f')  # これは日時をexperiment nameに設定する例．
    experiment.set_name(exp_name)
    experiment.add_tag(args.model)  # これはモデル名をタグに設定する例．
```
