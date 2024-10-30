import apebench


def test_trainer_parsing():
    advection_scenario = apebench.scenarios.difficulty.Advection()

    trainer = advection_scenario.get_trainer(
        train_config="one",
    )

    assert trainer.loss_configuration.num_rollout_steps == 1
    del trainer

    trainer = advection_scenario.get_trainer(train_config="sup;3")

    assert trainer.loss_configuration.num_rollout_steps == 3
    assert trainer.loss_configuration.cut_bptt is False
    del trainer

    trainer = advection_scenario.get_trainer(train_config="sup;3;True")

    assert trainer.loss_configuration.num_rollout_steps == 3
    assert trainer.loss_configuration.cut_bptt is True
    del trainer

    trainer = advection_scenario.get_trainer(train_config="sup;3;False")

    assert trainer.loss_configuration.num_rollout_steps == 3
    assert trainer.loss_configuration.cut_bptt is False
    del trainer

    trainer = advection_scenario.get_trainer(train_config="div;3")

    assert trainer.loss_configuration.num_rollout_steps == 3
    assert trainer.loss_configuration.cut_bptt is False
    assert trainer.loss_configuration.cut_div_chain is False
    del trainer

    trainer = advection_scenario.get_trainer(train_config="div;3;True")

    assert trainer.loss_configuration.num_rollout_steps == 3
    assert trainer.loss_configuration.cut_bptt is True
    assert trainer.loss_configuration.cut_div_chain is False
    del trainer

    trainer = advection_scenario.get_trainer(train_config="div;3;False")

    assert trainer.loss_configuration.num_rollout_steps == 3
    assert trainer.loss_configuration.cut_bptt is False
    assert trainer.loss_configuration.cut_div_chain is False
    del trainer

    trainer = advection_scenario.get_trainer(train_config="div;3;True;True")

    assert trainer.loss_configuration.num_rollout_steps == 3
    assert trainer.loss_configuration.cut_bptt is True
    assert trainer.loss_configuration.cut_div_chain is True
    del trainer

    trainer = advection_scenario.get_trainer(train_config="div;3;False;True")

    assert trainer.loss_configuration.num_rollout_steps == 3
    assert trainer.loss_configuration.cut_bptt is False
    assert trainer.loss_configuration.cut_div_chain is True
    del trainer
