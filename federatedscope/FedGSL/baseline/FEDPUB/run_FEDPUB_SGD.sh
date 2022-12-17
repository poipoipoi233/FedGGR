set -e

cd ../../../../

cudaid=0
dataset=cora
splitter=random
echo "HPO starts..."

if [[ $dataset = 'cora' ]]; then
    out_channels=7
elif [[ $dataset = 'citeseer' ]]; then
    out_channels=6
elif [[ $dataset = 'pubmed' ]]; then
    out_channels=3
elif [[ $dataset = 'computers' ]]; then
    out_channels=10
fi

client_num=(10)
hidden=(16 64 256)
dropout=(0 0.5)
#l1=(0 1e-3 1e-5)
#loc_l2=(0 1e-3 1e-5)
local_updates=(1 2 3)
lrs=(0.1 0.25 0.3 0.5)
weight_decay=(0 5e-4)
method=fedpub


outdir=federatedscope/FedGSL/baseline/FEDPUB/exp_out/${dataset}
for (( m=0;m<${#client_num[@]};m++))
do
  for (( h=0; h<${#hidden[@]}; h++ ))
  do
      for (( d=0; d<${#dropout[@]}; d++ ))
      do
          for (( i=0; i<${#local_updates[@]}; i++ ))
          do
              for (( j=0; j<${#lrs[@]}; j++))
              do
                  for (( w=0; w<${#weight_decay[@]}; w++))
                  do
                      for k in {0..4}
                      do
                          python federatedscope/main.py --cfg federatedscope/FedGSL/baseline/FEDPUB/FEDPUB.yaml device ${cudaid} data.type ${dataset} data.splitter ${splitter} federate.client_num ${client_num[T]} model.hidden ${hidden[$h]} model.dropout ${dropout[$d]} train.local_update_steps ${local_updates[$i]} train.optimizer.lr ${lrs[$j]} train.optimizer.weight_decay ${weight_decay[$w]} seed $k outdir ${outdir}/${hidden[$h]}_${dropout[$d]}_${local_updates[$i]}_${lrs[$j]}_${weight_decay[$weight_decay]}_on_${dataset}_${splitter}
                      done
                  done
              done

          done
      done
  done
done
echo "HPO ends."L


