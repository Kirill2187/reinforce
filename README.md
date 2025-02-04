## Reward model 

Обучается в [train_reward_model.py](scripts/train_reward_model.py). Столкнулся с проблемой что в pad_token это <|im_end|>, а в реализации LLamaForSequenceClassification эмбеддинг для предсказания берется из токена предшествующего первому pad токену, так что в нашем случае для классификации использутеся только промпт. Поэтому пришлось подменить pad_token на eos_token. С предложенными параметрами lr=5e-5 и 1 эпохой получилась точность 0.66 на валидации.

## Finetuning

Реализация REINFORCE в [reinforce.py](scripts/reinforce.py), а скрипт для запуска обучения в [finetune.py](scripts/finetune.py). Графики обучения: https://api.wandb.ai/links/tlab-task/tl9rrplh. Не уверен, что реализация полностью корректная, обучение очень слабенькое, но все таки на валидации средняя награда немного увеличивается (статистически значимо) - это проверяю в [validation.ipynb](validation.ipynb)

