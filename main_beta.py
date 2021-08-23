# -*- coding: utf-8 -*-
import torch

from package.dataset.dense import Dataset, DataLoader
from package.model.beta import BetaModel, BetaConfig
from tqdm import tqdm

lr = 5e-3
num_workers = 0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config = BetaConfig(pretrained_model_name_or_path='bert-base-multilingual-cased', pos_embedding_dim=64,
                    fc_1_hidden_size=768, fc_1_dropout_rate=0.2,
                    fc_2_hidden_size=768, fc_2_dropout_rate=0.2,
                    pre_tag_size=3, arg_tag_size=9
                    )
model = BetaModel(config).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

dataset = Dataset('./resource/SAOKE/SAOKE_DATA.json.json', model.tokenizer)
dataloader = DataLoader(dataset, batch_size=128, collate_fn=dataset.collate_fn, shuffle=True, num_workers=num_workers)

if __name__ == '__main__':

    for i in range(50):
        with tqdm(total=len(dataloader), desc=f'Epoch {i}: Training ...') as t:
            for batch in dataloader:
                optimizer.zero_grad()

                input_ids, mask, pre_label_all, pre_label, arg_label = [_.to(device) for _ in batch]

                loss_pre, loss_arg = model.loss(input_ids, mask, pre_label_all, pre_label, arg_label)

                loss = loss_pre + loss_arg

                loss.backward()
                optimizer.step()

                t.update()
                t.set_postfix(loss=loss.item(), loss_pre=loss_pre.item(), loss_arg=loss_arg.item())

        if (i + 1) % 25 == 0:
            torch.save(model.state_dict(), f"model_{i}.pth")
