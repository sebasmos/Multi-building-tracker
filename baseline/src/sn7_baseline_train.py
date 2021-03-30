import solaris as sol
import os
print(sol.__version__)
config_path = '../yml/sn7_baseline_train.yml'
#os.chmod(path, 0444)
config = sol.utils.config.parse(config_path)
print('Config:')
print(config)

# make model output dir
#print(!sudo)
os.makedirs(os.path.dirname(config['training']['model_dest_path']), exist_ok=True)

trainer = sol.nets.train.Trainer(config=config)
trainer.train()
