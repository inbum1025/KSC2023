import glob

import torch

model_list = ['34', '50']
alpha_list = [0.3, 0.5, 0.7, 0.9]

data_name = 'vien-tri'
batch_size = 32

with open('cls.txt', 'a', encoding='utf-8-sig') as file:
    from baseline_basic import Model, Datamodule

    for model_type in model_list:
        model_name = f"UT-R{model_type}-vien-tri"
        datamodule = Datamodule(data_name, batch_size, alpha=0.9)
        datamodule.setup('test')

        model_path = glob.glob(f"model/{model_name}_basic*.ckpt")[0]
        model = Model.load_from_checkpoint(model_path, model_name=f"resnet{model_type}", data_name=data_name, map_location=torch.device('cuda:5'))
        print(model_path)
        file.write(f"{model_type}_basic\t{model.evaluate(datamodule.train_dataloader_no_shuffle(), datamodule.test_dataloader())}\n")
    # from baseline_cutmix import Model, Datamodule
    #
    # for model_type in model_list:
    #     model_name = f"UT-R{model_type}-vien-tri"
    #     for alpha in alpha_list:
    #         datamodule = Datamodule(data_name, batch_size, alpha)
    #         datamodule.setup('test')
    #
    #         model_path = glob.glob(f"model/{model_name}-{alpha}_cutmix*.ckpt")[0]
    #         model = Model.load_from_checkpoint(model_path, model_name=f"resnet{model_type}", data_name=data_name, map_location=torch.device('cuda:5'))
    #         print(model_path)
    #         file.write(f"{model_type}-{alpha}_cutmix\t{model.evaluate(datamodule.train_dataloader_no_shuffle(), datamodule.test_dataloader())}\n")
    #
    # from baseline_mixup import Model, Datamodule
    #
    # for model_type in model_list:
    #     model_name = f"UT-R{model_type}-vien-tri"
    #     for alpha in alpha_list:
    #         datamodule = Datamodule(data_name, batch_size, alpha)
    #         datamodule.setup('test')
    #
    #         model_path = glob.glob(f"model/{model_name}-{alpha}_mixup*.ckpt")[0]
    #         model = Model.load_from_checkpoint(model_path, model_name=f"resnet{model_type}", data_name=data_name, map_location=torch.device('cuda:5'))
    #         print(model_path)
    #         file.write(f"{model_type}-{alpha}_mixup\t{model.evaluate(datamodule.train_dataloader_no_shuffle(), datamodule.test_dataloader())}\n")