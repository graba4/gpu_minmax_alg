res='final_results.csv'

for t in 400 450 500 600 650 700 750 800 850 900 950 1000 1024; do
        ./lin-sys-solv -v 900 -c 1 -i 0 -r 3 -t $t -f $res
done
