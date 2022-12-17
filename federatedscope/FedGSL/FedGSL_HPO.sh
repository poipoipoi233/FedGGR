set -e

cd ../../

cudaid=0
dataset=pubmed #1
splitter='random'

method=fedgsl #3
client_num=3  #pleas note this hyper-parameter
generator='MLP-D'
patience=300

op=('Adam')
hidden=(64)
gsl_gnn_hids=(256)
dropout=(0.)
local_updates=(1)
lrs=(0.03)
batch_size=(1)
k_for_knn=(30)
loc_gnn_outsize=(160)
glob_gnn_outsize=(160)
#loc_gnn_outsize=(32 64 96 128 160 192 224 256)
#glob_gnn_outsize=(32 64 96 128 160 192 224 256)
weight_decay=(0.)

if [[ $dataset = 'cora' ]]; then
  out_channels=7
  pretrain_out_channels=64
  patience=30
  weight_decay=(0)
  hidden=(256)
elif [[ $dataset = 'citeseer' ]]; then
  out_channels=6
  pretrain_out_channels=64
  patience=50
elif [[ $dataset = 'pubmed' ]]; then
  out_channels=3
  pretrain_out_channels=256
  patience=100
elif [[ $dataset = 'computers' ]]; then
  out_channels=10
  pretrain_out_channels=256
fi



outdir=../../../data/zhl_FedGSL_exp_out/${method}_on_${dataset}

echo "HPO starts..."
temp=0
for ((b = 0; b < ${#batch_size[@]}; b++)); do
  for ((h = 0; h < ${#hidden[@]}; h++)); do
    for ((d = 0; d < ${#dropout[@]}; d++)); do
      for ((i = 0; i < ${#local_updates[@]}; i++)); do
        for ((j = 0; j < ${#lrs[@]}; j++)); do
          for ((w = 0; w < ${#weight_decay[@]}; w++)); do
            for ((o = 0; o < ${#op[@]}; o++)); do
              for ((gh = 0; gh < ${#gsl_gnn_hids[@]}; gh++)); do
                for ((lout = 0; lout < ${#loc_gnn_outsize[@]}; lout++)); do
                  for ((gout = 0; gout < ${#glob_gnn_outsize[@]}; gout++)); do
                    for ((k_num = 0; k_num < ${#k_for_knn[@]}; k_num++)); do
                      for k in {1..2}; do
                        let temp+=1

                      done
                    done
                  done
                done
              done
            done
          done
        done
      done
    done
  done
done
echo "total epoch=="
echo -e "$temp" #打印共执行多少轮

temp=0
for ((b = 0; b < ${#batch_size[@]}; b++)); do
  for ((h = 0; h < ${#hidden[@]}; h++)); do
    for ((d = 0; d < ${#dropout[@]}; d++)); do
      for ((i = 0; i < ${#local_updates[@]}; i++)); do
        for ((j = 0; j < ${#lrs[@]}; j++)); do
          for ((w = 0; w < ${#weight_decay[@]}; w++)); do
            for ((o = 0; o < ${#op[@]}; o++)); do
              for ((gh = 0; gh < ${#gsl_gnn_hids[@]}; gh++)); do
                for ((lout = 0; lout < ${#loc_gnn_outsize[@]}; lout++)); do
                  for ((gout = 0; gout < ${#glob_gnn_outsize[@]}; gout++)); do
                    for ((k_num = 0; k_num < ${#k_for_knn[@]}; k_num++)); do
                      for k in {0..2}; do
                        let temp+=1

#                        if [[ $temp -le 31 ]];then
#                          continue
#                        fi

                        python federatedscope/main.py \
                          --cfg federatedscope/FedGSL/FedGSL.yaml \
                          device ${cudaid} \
                          federate.client_num ${client_num} \
                          federate.sample_client_num ${client_num} \
                          dataloader.batch_size ${batch_size[b]} \
                          train.optimizer.type ${op[$o]} \
                          data.type ${dataset} \
                          data.splitter ${splitter} \
                          federate.method $method \
                          model.out_channels $out_channels \
                          model.hidden ${hidden[$h]} \
                          model.dropout ${dropout[$d]} \
                          train.local_update_steps ${local_updates[$i]} \
                          train.optimizer.lr ${lrs[$j]} \
                          train.optimizer.weight_decay ${weight_decay[$w]} \
                          seed $k \
                          model.fedgsl.gsl_gnn_hids ${gsl_gnn_hids[$gh]} \
                          model.fedgsl.server_lr ${lrs[$j]} \
                          model.fedgsl.loc_gnn_outsize ${loc_gnn_outsize[$lout]} \
                          model.fedgsl.glob_gnn_outsize ${glob_gnn_outsize[$gout]} \
                          model.fedgsl.k ${k_for_knn[$k_num]} \
                          model.fedgsl.pretrain_out_channels $pretrain_out_channels \
                          model.fedgsl.HPO True \
                          early_stop.patience $patience \
                          model.fedgsl.generator $generator \
                          outdir ${outdir}/${hidden[$h]}_${dropout[$d]}_${local_updates[$i]}_${lrs[$j]}_${weight_decay[$w]}_on_${dataset}_${splitter}
                        #note: yaml文件路径需要修改;划分方式需要确认
                      done
                    done
                  done
                done
              done
            done
          done
        done
      done
    done
  done
done

echo "FedGSL HPO ends."L
