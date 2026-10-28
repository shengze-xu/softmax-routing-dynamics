"""Download the two AG News source CSVs."""
import argparse
from pathlib import Path
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('data/ag_news_csv'))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base = 'https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/'
    for name in ('train.csv', 'test.csv'):
        path = args.output_dir / name
        if not path.exists():
            urllib.request.urlretrieve(base + name, path)
        print(path)


if __name__ == '__main__':
    main()
