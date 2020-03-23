# echo "K=13" > config.py
# python main.py --dataset uow_dataset_full -b 8 --gpu 0 --epochs 40

# echo "K=16" > config.py
# python main.py --dataset uow_dataset_full -b 8 --gpu 0 --epochs 40

#echo "K=24" > config.py
#python main.py --dataset uow_dataset_full -b 8 --gpu 0 --epochs 40

# echo "K=83" > config.py
# python main.py --dataset uow_dataset_full -b 6 --gpu 0 --epochs 40

# echo "K=99" > config.py
# python main.py --dataset uow_dataset_full -b 6 --gpu 0 --epochs 40

# echo "K=82" > config.py
# python main.py --dataset uow_dataset_full -b 6 --gpu 0 --epochs 40
# for K in 12 15 17 72 73 74 82 83 85
# do
# 	echo "K=$K" > config.py
#     python main.py --dataset uow_dataset_full -b 6 --gpu 0 --epochs 20
# done


for K in 14 24 26
do
	echo "K=$K" > config.py
    python main.py --dataset uow_dataset_full -b 6 --gpu 0 --epochs 20
done

# for K in 12 15 17 72 73 74 82 83 85
# do
# 	echo "K=$K" > config.py
#     python main.py --dataset uow_dataset_full -b 6 --gpu 0 --epochs 20
# done
