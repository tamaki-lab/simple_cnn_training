import argparse
import yaml
from argparse import Namespace


class CustomFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.MetavarTypeHelpFormatter
):
    """show default values of argparse.
    see
    https://stackoverflow.com/questions/18462610/argumentparser-epilog-and-description-formatting-in-conjunction-with-argumentdef
    for details.
    """


def load_from_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def merge_dicts(*dicts):
    merged = {}
    for d in dicts:
        merged.update(d)
    return merged


class ArgParse:

    @staticmethod
    def get() -> argparse.Namespace:
        """generate argparse object

        Returns:
            args (argparse.Namespace): object of command line arguments
        """
        parser = argparse.ArgumentParser(
            description="simple image/video classification",
            formatter_class=CustomFormatter
        )
        parser.add_argument("--common_config", type=str, required=True, help="共通設定のYAMLファイル")
        parser.add_argument("--model_config", type=str, required=True, help="モデル固有設定のYAMLファイル")
        parser.add_argument("--devices", type=str, default="-1", help="GPU ID のみはコマンドライン指定")
        parser.add_argument("--dataset_config", type=str, required=True, help="データセット固有の設定")

        # 一旦パースして設定ファイルを取得
        temp_args, _ = parser.parse_known_args()

        # YAMLから設定を読み込み
        common_cfg = load_from_yaml(temp_args.common_config)
        model_cfg = load_from_yaml(temp_args.model_config)
        dataset_cfg = load_from_yaml(temp_args.dataset_config)

        # マージして Namespace 化
        merged_cfg = merge_dicts(common_cfg, model_cfg, dataset_cfg)
        merged_cfg["devices"] = temp_args.devices  # devicesだけはコマンドライン優先

        return Namespace(**merged_cfg)
