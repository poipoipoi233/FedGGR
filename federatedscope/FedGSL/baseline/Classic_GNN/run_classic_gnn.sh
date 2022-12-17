set -e

cd ../../../../

cudaid=0
dataset=citeseer
method=gpr
splitter=random
client_num=(3 5 10)


batch_size=1 #来自zhl的提醒:用sage方法的时候需要在if里面指定特殊的batch_size
patience=30
seed=(0 1 2 3 4)
echo "run starts..."
if [[ $dataset = 'cora' ]]; then
  out_channels=7
  op=Adam
  hidden=256
  dropout=0.5
  local_updates=1
  weight_decay=0.
  gsl_gnn_hids=256
  loc_gnn_outsize=256
  glob_gnn_outsize=256
  pretrain_out_channels=64
  k_for_knn=15
  lr=0.01
elif [[ $dataset = 'citeseer' ]]; then
  if [[ $method = 'gcn' ]]; then
    model_type='gcn'
    out_channels=6
    op=Adam
    hidden=64
    dropout=0.5
    local_updates=2
    weight_decay=0.
    gsl_gnn_hids=64
    loc_gnn_outsize=64
    glob_gnn_outsize=64
    pretrain_out_channels=64
    k_for_knn=15
    lr=0.001
  elif [[ $method = 'gat' ]]; then
    model_type='gat'
    out_channels=6
    op=Adam
    hidden=64
    dropout=0.5
    local_updates=2
    weight_decay=0.
    lr=0.001
  elif [[ $method = 'gpr' ]]; then
    model_type='gpr'
    out_channels=6
    op=Adam
    hidden=64
    dropout=0.5
    local_updates=2
    weight_decay=0.
    lr=0.001
  fi

elif [[ $dataset = 'pubmed' ]]; then
  out_channels=3
  op=Adam
  hidden=64
  dropout=0.
  local_updates=1
  weight_decay=0.
  gsl_gnn_hids=256
  loc_gnn_outsize=64
  glob_gnn_outsize=256
  pretrain_out_channels=256
  k_for_knn=30
  lr=0.03
  server_lr=0.03 #note 其他dataset里每设置这个
elif [[ $dataset = 'computers' ]]; then
  if [[ $method = 'gcn' ]]; then
    out_channels=10
    op=Adam
    hidden=64
    dropout=0.5
    local_updates=1
    weight_decay=0.
    lr=0.01
  fi
fi

outdir=../../../data/zhl_FedGSL_exp_out/${method}_on_${dataset}

#请确定HPO是否为False
for ((m = 0; m < ${#client_num[@]}; m++)); do
  for ((k = 0; k < ${#seed[@]}; k++)); do
    python federatedscope/main.py \
      --cfg federatedscope/FedGSL/baseline/Classic_GNN/fedavg_gnn_node_fullbatch_citation.yaml \
      device ${cudaid} \
      federate.client_num ${client_num[$m]} \
      federate.sample_client_num ${client_num[$m]} \
      dataloader.batch_size ${batch_size} \
      train.optimizer.type ${op} \
      data.type ${dataset} \
      data.splitter ${splitter} \
      federate.method $method \
      model.type $model_type \
      model.out_channels $out_channels \
      model.hidden ${hidden} \
      model.dropout ${dropout} \
      train.local_update_steps ${local_updates} \
      train.optimizer.lr $lr \
      train.optimizer.weight_decay $weight_decay \
      seed ${seed[k]} \
      model.fedgsl.HPO False \
      early_stop.patience $patience \
      outdir ${outdir}/${hidden}_${dropout}_${local_updates}_${lr}_${weight_decay}_on_${dataset}_${splitter}
    #      python  federatedscope/main.py --cfg federatedscope/FedGSL/baseline/GAT/GAT.yaml device ${cudaid} dataloader.batch_size ${batch_size} federate.client_num ${client_num[$m]}  data.type ${dataset} data.splitter ${splitter} model.out_channels ${out_channels} model.hidden ${hidden} model.dropout ${dropout} train.local_update_steps ${local_updates} train.optimizer.lr ${lr} train.optimizer.weight_decay ${weight_decay} train.optimizer.type $op seed $k outdir ${outdir}/${client_num[$m]}_clients
  done
done

echo "Run ends."
