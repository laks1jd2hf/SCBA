import loan_gradmask
import cifar_gradmask
import mnist_gradmask
import cifar100_gradmask
import fmnist_gradmask
import config


def train(helper, start_epoch, local_model, target_model, is_poison, agent_name_keys, noise_trigger, intinal_trigger):
    epochs_submit_update_dict = {}
    num_samples_dict = {}
    user_grad = []
    server_update = dict()
    if helper.params['type'] == config.TYPE_LOAN:
        epochs_submit_update_dict, num_samples_dict, user_grads, server_update, tuned_trigger = loan_gradmask.LoanTrain(
            helper,
            start_epoch,
            local_model,
            target_model,
            is_poison,
            agent_name_keys,
            noise_trigger,
            intinal_trigger)
    if helper.params['type'] == config.TYPE_CIFAR:
        epochs_submit_update_dict, num_samples_dict, user_grad, server_update, tuned_trigger = cifar_gradmask.ImageTrain(
            helper,
            start_epoch,
            local_model,
            target_model,
            is_poison,
            agent_name_keys,
            noise_trigger,
            intinal_trigger)
    if helper.params['type'] == config.TYPE_CIFAR100:
        epochs_submit_update_dict, num_samples_dict, user_grad, server_update, tuned_trigger = cifar100_gradmask.Cifar100Train(
            helper,
            start_epoch,
            local_model,
            target_model,
            is_poison,
            agent_name_keys,
            noise_trigger,
            intinal_trigger)

    if helper.params['type'] == config.TYPE_MNIST:
        epochs_submit_update_dict, num_samples_dict, user_grad, server_update, tuned_trigger = mnist_gradmask.MnistTrain(
            helper,
            start_epoch,
            local_model,
            target_model,
            is_poison,
            agent_name_keys,
            noise_trigger,
            intinal_trigger)
    if helper.params['type'] == config.TYPE_FMNIST:
        epochs_submit_update_dict, num_samples_dict, user_grads, server_update, tuned_trigger = fmnist_gradmask.FMnistTrain(
            helper,
            start_epoch,
            local_model,
            target_model,
            is_poison,
            agent_name_keys,
            noise_trigger,
            intinal_trigger)

    return epochs_submit_update_dict, num_samples_dict, user_grad, server_update, tuned_trigger
