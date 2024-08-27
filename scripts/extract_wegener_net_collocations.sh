#1/bin/bash

echo "Extracting data for year $1 and month $2."
speed extract_data gmi wegener_net /xdata/simon/raw/gmi/wegener_net/ $1 $2
