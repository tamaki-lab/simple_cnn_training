import argparse


class CustomFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.MetavarTypeHelpFormatter
):
    """show default values of argparse.
    see
    https://stackoverflow.com/questions/18462610/argumentparser-epilog-and-description-formatting-in-conjunction-with-argumentdef
    for details.
    """


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

         # config
        parser.add_argument(
            "--config-path",
            type=str,
            default="configs",
            help="root of config path",
        )
        parser.add_argument(
            "--config-name",
            type=str,
            default="default",
            help="name of config file",
        )

        args = parser.parse_args()

        print(args)

        return args
