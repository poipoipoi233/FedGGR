set -e

cd ../../

cudaid=$1
dataset=$2
method=fedgsl
splitter=random #or louvain
client_num=(3 5 10)
backbone='gcn'

patience=200
seed=(0 1 2 3 4)
model_type='gnn_fedgsl_gcn'
generator='MLP-D'
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
elif [[ $dataset = 'computers' ]]; then
  out_channels=10
  op=Adam
  hidden=64
  dropout=0.5
  local_updates=3
  weight_decay=0.0
  gsl_gnn_hids=64
  loc_gnn_outsize=256
  glob_gnn_outsize=64
  pretrain_out_channels=256
  k_for_knn=20
  lr=0.01
fi

echo "run starts..."

for ((m = 0; m < ${#client_num[@]}; m++)); do
  for ((k = 0; k < ${#seed[@]}; k++)); do
    python federatedscope/main.py \
      --cfg federatedscope/FedGSL/FedGSL.yaml \
      device ${cudaid} \
      federate.client_num ${client_num[$m]} \
      federate.sample_client_num ${client_num[$m]} \
      train.optimizer.type ${op} \
      data.type ${dataset} \
      data.splitter ${splitter} \
      federate.method $method \
      model.type $model_type \
      model.out_channels $out_channels \
      model.hidden ${hidden} \
      model.dropout ${dropout} \
      train.local_update_steps ${local_updates} \
      train.optimizer.lr ${lr} \
      train.optimizer.weight_decay $weight_decay \
      seed ${seed[k]} \
      model.fedgsl.gsl_gnn_hids ${gsl_gnn_hids} \
      model.fedgsl.server_lr ${lr} \
      model.fedgsl.loc_gnn_outsize ${loc_gnn_outsize} \
      model.fedgsl.glob_gnn_outsize ${glob_gnn_outsize} \
      model.fedgsl.k $k_for_knn \
      model.fedgsl.pretrain_out_channels $pretrain_out_channels \
      model.fedgsl.HPO False \
      early_stop.patience $patience \
      model.fedgsl.generator $generator
  done
done

echo "Run ends."
