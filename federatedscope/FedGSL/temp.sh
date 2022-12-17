set -e

cd ../../

cudaid=0
dataset=citeseer #1
splitter='random'

method=fedgsl #3
client_num=5  #pleas note this hyper-parameter
generator='MLP-D'
patience=200

op=('Adam')
hidden=(64 256)
gsl_gnn_hids=(64 256)
dropout=(0 0.5)
local_updates=(1 2 3)
lrs=(0.01)
batch_size=(1)
k_for_knn=(5 10 15 20)
loc_gnn_outsize=(64 256)
glob_gnn_outsize=(64 256)
weight_decay=(0 5e-4)

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
                      for k in {0..4}; do
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