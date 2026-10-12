"""Download the two AG News source CSVs and verify their recorded hashes."""
import argparse
from pathlib import Path
import urllib.request
from experiments.data.prepare_agnews_corpus import SOURCE_DIGESTS, file_digest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=Path('data/ag_news_csv'))
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    base='https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/'
    for name,hashes in SOURCE_DIGESTS.items():
        path=args.output_dir/name
        if not path.exists():urllib.request.urlretrieve(base+name,path)
        if file_digest(path)!=hashes['sha256']:
            raise ValueError(f'AG News checksum mismatch: {path}')
        print(f'verified {path}')


if __name__=='__main__':main()
