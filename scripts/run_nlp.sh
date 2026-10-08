
for dataset in superglue_cb superglue_rte superglue_boolq superglue_wic superglue_wsc; do
        for peftmode in bitfit lora; do
        if [ $dataset == "superglue_cb" ]; then
                cls_num=3
                numclient=5
                bs=16
                mbs=4
                if [ $peftmode == "bitfit" ]; then
                        lr=1e-3
                else
                        lr=1e-5
                fi
        else
                cls_num=2
                numclient=10
                bs=32
                mbs=8
                if [ $peftmode == "bitfit" ]; then
                        lr=1e-2
                else
                        lr=1e-4
                fi
        fi
        # C2Fed
        python c2fed.py --model_name cardiffnlp/twitter-roberta-base-sentiment-latest \
                --dataset $dataset --partition noniid-labeluni --cls_num $cls_num --n_parties $numclient --sample 0.5 \
                --comm_round 100  --test_round 99 --lr $lr -K 5 --batch_size $bs --device cuda \
                --train fwd --fwd_head_tuning --h 0.001 --momentum_type direct --momentum 0.9 \
                -N 8 --micro_batch_size $mbs --num_micro_batch 2 --sample_selection uncertainty\
                --peftmode $peftmode --logdir ./exps/$dataset/$peftmode/

        # DeComFL
        python c2fed.py --model_name cardiffnlp/twitter-roberta-base-sentiment-latest \
                --dataset $dataset --partition noniid-labeluni --cls_num $cls_num --n_parties $numclient --sample 0.5 \
                --comm_round 100  --test_round 99 --lr $lr -K 5 --batch_size $bs --device cuda \
                --train fwd --h 0.001 \
                -N 10 --micro_batch_size $bs --num_micro_batch 1 --sample_selection all\
                --peftmode $peftmode --logdir ./exps/$dataset/$peftmode/


        # # FedAvg
        python c2fed.py --model_name cardiffnlp/twitter-roberta-base-sentiment-latest \
                --dataset $dataset --partition noniid-labeluni --cls_num $cls_num --n_parties $numclient --sample 0.5 \
                --comm_round 100  --test_round 99 --lr $lr -K 5 --batch_size $bs --device cuda \
                --train bp --momentum 0 --dampening 0 \
                --peftmode $peftmode --logdir ./exps/$dataset/$peftmode/

        # # FedAvg-m
        python c2fed.py --model_name cardiffnlp/twitter-roberta-base-sentiment-latest \
                --dataset $dataset --partition noniid-labeluni --cls_num $cls_num --n_parties $numclient --sample 0.5 \
                --comm_round 100  --test_round 99 --lr $lr -K 5 --batch_size $bs --device cuda \
                --train bp --momentum_type fedavg-m --momentum 0.5 --dampening 0.5 \
                --peftmode $peftmode --logdir ./exps/$dataset/$peftmode/

        # FwdLLM
        python fwdllm.py --model_name cardiffnlp/twitter-roberta-base-sentiment-latest \
                --dataset $dataset --partition noniid-labeluni --cls_num $cls_num --n_parties $numclient --sample 0.5 \
                --comm_round 100  --test_round 99 --lr $lr -N 5 -K 5 --batch_size $bs  --device cuda \
                --train fwdllm --h 0.001 --var_threshold 0.2\
                --peftmode $peftmode --logdir ./exps/$dataset/$peftmode/
        done
done
