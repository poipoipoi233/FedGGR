set -e

cd ../../

cudaid=0
dataset=pubmed
method=fedgsl
splitter=random
client_num=(3 5 10)
backbone='gat' #gcn gat gpr
echo "run starts..."

batch_size=1 #来自zhl的提醒:用sage方法的时候需要在if里面指定特殊的batch_size
generator='MLP-D' #指定GSL图生成器类型
patience=30
seed=(0 1 2 3 4)
if [[ $backbone = 'gcn' ]];then
  model_type='gnn_fedgsl_gcn'
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
    server_lr=0.03 #note 其他dataset里每设置这个
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
elif [[ $backbone = 'gat' ]]; then
  model_type='gnn_fedgsl_gat'
  if [[ $dataset = 'citeseer' ]]; then
    out_channels=6
    op=Adam
    hidden=256
    dropout=0.5
    local_updates=3
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
    server_lr=0.03 #note 其他dataset里每设置这个
  fi
elif [[ $backbone = 'gpr' ]]; then
  model_type='gnn_fedgsl_gpr'
  if [[ $dataset = 'citeseer' ]]; then
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
    k_for_knn=0.5
    lr=0.001
    server_lr=0.001
  fi
elif [[ $backbone = 'gin' ]]; then
  model_type='gnn_fedgsl_gin'
  if [[ $dataset = 'citeseer' ]]; then
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
    k_for_knn=0.5
    lr=0.001
    server_lr=0.001
  fi
fi

outdir=../../../data/zhl_FedGSL_exp_out/${method}_on_${dataset}

#请确定HPO是否为False
for ((m = 0; m < ${#client_num[@]}; m++)); do
  for ((k = 0; k < ${#seed[@]}; k++)); do
    python federatedscope/main.py \
      --cfg federatedscope/FedGSL/FedGSL.yaml \
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
      model.fedgsl.gsl_gnn_hids ${gsl_gnn_hids} \
      model.fedgsl.server_lr ${server_lr} \
      model.fedgsl.loc_gnn_outsize ${loc_gnn_outsize} \
      model.fedgsl.glob_gnn_outsize ${glob_gnn_outsize} \
      model.fedgsl.k $k_for_knn \
      model.fedgsl.pretrain_out_channels $pretrain_out_channels \
      model.fedgsl.HPO False \
      early_stop.patience $patience \
      model.fedgsl.generator $generator \
      outdir ${outdir}/${hidden}_${dropout}_${local_updates}_${lr}_${weight_decay}_on_${dataset}_${splitter}
    #      python  federatedscope/main.py --cfg federatedscope/FedGSL/baseline/GAT/GAT.yaml device ${cudaid} dataloader.batch_size ${batch_size} federate.client_num ${client_num[$m]}  data.type ${dataset} data.splitter ${splitter} model.out_channels ${out_channels} model.hidden ${hidden} model.dropout ${dropout} train.local_update_steps ${local_updates} train.optimizer.lr ${lr} train.optimizer.weight_decay ${weight_decay} train.optimizer.type $op seed $k outdir ${outdir}/${client_num[$m]}_clients
  done
done

echo "Run ends."
