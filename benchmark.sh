res='final_results.csv'

echo "impl0 static cores alg only" >> $res

echo "size,window,cores,average_gpu,average_cpu,impl,seed" >> $res
for w in 10 20 30 40 60 100 150 200 300 400 500 600 700; do
	./min_max -v 10000000 -w $w -c 8 -t 1024 -i 0 -r 3 -f $res -a
	./min_max -v 20000000 -w $w -c 8 -t 1024 -i 0 -r 3 -f $res -a
	./min_max -v 40000000 -w $w -c 8 -t 1024 -i 0 -r 3 -f $res -a
	./min_max -v 60000000 -w $w -c 8 -t 1024 -i 0 -r 3 -f $res -a
	./min_max -v 80000000 -w $w -c 8 -t 1024 -i 0 -r 3 -f $res -a
	./min_max -v 100000000 -w $w -c 8 -t 1024 -i 0 -r 3 -f $res -a
	echo "" >> $res
done
echo "" >> $res

echo "impl0 inc cores alg only" >> $res
for c in 1 2 3 4 5 6 7 8; do
	./min_max -v 60000000 -w 30 -c $c -t 1024 -i 0 -r 3 -f $res -a
done

echo "" >> $res

echo "impl0 inc threads alg only" >> $res
for t in 1 2 3 4 5 6 7 8 9 10 15 20 25 30 40 50 60 80 100 150 200 250 300 400 500 700 900 1024; do
	./min_max -v 60000000 -w 30 -c $c -t $t -i 0 -r 3 -f $res -a
done

echo "" >> $res

echo "impl1 inc cores alg only" >> $res
for c in 1 2 3 4 5 6 7 8; do
	./min_max -v 60000000 -w 30 -c $c -t 1024 -i 1 -r 3 -f $res -a
done

echo "" >> $res

echo "impl1 inc threads alg only" >> $res
for t in 1 2 3 4 5 6 7 8 9 10 15 20 25 30 40 50 60 80 100 150 200 250 300 400 500 700 900 1024; do
	./min_max -v 60000000 -w 30 -c $c -t $t -i 1 -r 3 -f $res -a
done

echo "" >> $res