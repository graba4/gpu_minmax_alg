res='final_results.csv'

echo "own sys" >> $res
: '
for i in 0 1; do
	for c in 1 2 4 8; do
		echo "size,cores,time $c,impl" >> $res
		./lin-sys-solv -v 900 -c $c -i $i -r 3 -f $res
		./lin-sys-solv -v 1000 -c $c -i $i -r 3 -f $res
		./lin-sys-solv -v 1100 -c $c -i $i -r 3 -f $res
		./lin-sys-solv -v 1200 -c $c -i $i -r 3 -f $res
		./lin-sys-solv -v 1300 -c $c -i $i -r 3 -f $res
		./lin-sys-solv -v 1400 -c $c -i $i -r 3 -f $res
		./lin-sys-solv -v 1500 -c $c -i $i -r 3 -f $res
		./lin-sys-solv -v 1800 -c $c -i $i -r 3 -f $res
		./lin-sys-solv -v 2300 -c $c -i $i -r 3 -f $res
		./lin-sys-solv -v 2500 -c $c -i $i -r 3 -f $res
		./lin-sys-solv -v 3000 -c $c -i $i -r 3 -f $res
		./lin-sys-solv -v 3200 -c $c -i $i -r 3 -f $res
		./lin-sys-solv -v 3600 -c $c -i $i -r 3 -f $res
		./lin-sys-solv -v 3800 -c $c -i $i -r 3 -f $res
		./lin-sys-solv -v 4200 -c $c -i $i -r 3 -f $res
		./lin-sys-solv -v 4500 -c $c -i $i -r 3 -f $res
		echo "" >> $res
	done
	echo "" >> $res
done
'

echo "static v test:" >> $res

for i in 0 1; do
	for c in 1 2 3 4 5 6 7 8; do
		./lin-sys-solv -v 5500 -c $c -i $i -r 3 -f $res
	done
done

:'
for t in 1 2 3 4 5 6 7 8 9 10 15 20 25 30 40 50 60 80 100 150 200 250 300 400 500 700 900 1024; do
	./lin-sys-solv -v 300 -c 1 -i 0 -r 3 -t $t -f $res
done
'