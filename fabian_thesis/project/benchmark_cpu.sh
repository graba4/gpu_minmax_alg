res='final_results.csv'

echo "own sys cpu" >> $res

for i in 0; do
		echo "size,cores,time $c,impl" >> $res
		./lin-sys-solv -v 4000 -c -1 -i $i -r 1 -f $res
		./lin-sys-solv -v 4200 -c -1 -i $i -r 1 -f $res
		./lin-sys-solv -v 4500 -c -1 -i $i -r 1 -f $res
	echo "" >> $res
done