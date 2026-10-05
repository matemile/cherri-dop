47c2f3eb7843fc6641e32e1c6889c746ae253951 (just for testing...)

milemate@uan02:/scratch/project_465003270/milemate/cherri-dop> uv pip list | grep -i anemoi
anemoi-datasets                    0.5.44
anemoi-graphs                      0.9.7.post32
anemoi-inference                   0.12.0
anemoi-models                      0.19.0.post32
anemoi-registry                    0.3.3
anemoi-training                    0.17.0.post32
anemoi-transform                   0.4.3
anemoi-utils                       0.5.12


RICHER-BATCH latest HEAD (2026.10.05.):
anemoi-training = { git = "https://github.com/ecmwf/anemoi-core.git", rev = "638ebda82e03c48b78ad33a5e4c31e19a8ae529f", subdirectory = "training" }

milemate@uan02:/scratch/project_465003270/milemate/cherri-dop> uv pip list | grep -i anemoi
anemoi-datasets                    0.5.44
anemoi-graphs                      0.9.7.post407
anemoi-inference                   0.12.0
anemoi-models                      0.19.0.post407
anemoi-registry                    0.3.3
anemoi-training                    0.17.0.post407
anemoi-transform                   0.4.3
anemoi-utils                       0.5.12

BUG in richer-batch

anemoi-core/models/src/anemoi/models/models/base.py
Line187
- self._graph_name_hidden = model_config.model.model.hidden_nodes_name
+ self._graph_name_hidden = model_config.model.hidden_nodes_name

milemate@uan03:/scratch/project_465003270/milemate/cherri-dop> uv pip list | grep -i anemoi
anemoi-datasets                    0.5.44
anemoi-graphs                      0.9.7.post407
anemoi-inference                   0.12.0
anemoi-models                      0.19.0.post407      /pfs/lustrep3/scratch/project_465003270/milemate/bris-gitsource/anemoi-core/models
anemoi-registry                    0.3.3
anemoi-training                    0.17.0.post407
anemoi-transform                   0.4.3
anemoi-utils                       0.5.12
