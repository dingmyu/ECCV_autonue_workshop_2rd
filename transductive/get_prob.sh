#for((i=1;i<=10;i++));  
#do   
#nohup srun -p Test -n1 -c2 python -u get_prob.py $(expr $i \* 50) $(expr $i \* 50 + 50)  & 
#done

for((i=0;i<=40;i++));  
do   
nohup srun -p Segmentation1080 -n1 -c4 python -u get_prob.py $(expr $i \* 50) $(expr $i \* 50 + 50)  & 
done
