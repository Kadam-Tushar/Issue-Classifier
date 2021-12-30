for split in test train
do
    echo Downloading and extracting $split data
    curl "https://tickettagger.blob.core.windows.net/datasets/github-labels-top3-803k-$split.tar.gz" | tar -xz
    mv github-labels-top3-803k-$split.csv $split.csv
done