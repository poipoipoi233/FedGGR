set -e

cd ../../../../

cudaid=0
dataset=pubmed
splitter=random
op=Adam
method=fedpub
client_num=(3 5)
echo "run starts..."

if [[ $method = 'fedpub' ]]; then
  if [[ $dataset = 'cora' ]];then
    hidden=64
    out_channels=7
    dropout=0.5
    local_updates=1
    lr=0.01
    weight_decay=5e-4
  elif [[ $dataset = 'citeseer' ]]; then
    hidden=256
    out_channels=6
    dropout=0.5
    local_updates=2
    op=Adam
    lr=0.001
    weight_decay=5e-4
  elif [[ $dataset = 'pubmed' ]]; then
    hidden=256
    out_channels=3
    dropout=0.5
    local_updates=1
    op=Adam
    lr=0.01
    weight_decay=0
  elif [[ $dataset = 'computers' ]]; then
    hidden=256
    out_channels=10
    dropout=0.
    local_updates=3
    op=Adam
    lr=0.01
    weight_decay=0
  fi

fi

outdir=../../../data/zhl_FedGSL_exp_out/${method}_on_${dataset}

for (( m=0;m<${#client_num[@]};m++))
do
  for k in {0..4}
  do
      python federatedscope/main.py --cfg federatedscope/FedGSL/baseline/FEDPUB/FEDPUB.yaml device ${cudaid} \
      federate.client_num ${client_num[m]} \
      federate.sample_client_num ${client_num[$m]} \
      data.type ${dataset} data.splitter ${splitter} model.out_channels ${out_channels} model.hidden ${hidden} model.dropout ${dropout} train.local_update_steps ${local_updates} train.optimizer.lr ${lr} train.optimizer.weight_decay ${weight_decay} train.optimizer.type $op seed $k outdir ${outdir}/${client_num[$m]}_clients
  done
done

echo "HPO ends."L


