import csv
import os
import torch
import pickle

# Update dictionary of metrics used in the main script
def update_metrics(metrics, loss, pearson, spearman, kappa):
    metrics["rmse"].append(loss)
    metrics["pearson"].append(pearson)
    metrics["spearman"].append(spearman)
    metrics["kappa"].append(kappa)
    return metrics

# Update writer in Tensorboard
def update_writer(writer, e, loss, pearson, spearman, kappa, is_eval=False):
    if is_eval:
        writer.add_scalar('Epoch validation loss', loss, e + 1)
        writer.add_scalar('Epoch validation pearson', pearson, e + 1)
        writer.add_scalar('Epoch validation spearman', spearman, e + 1)
        writer.add_scalar('Epoch validation kappa', kappa, e + 1)
    else:
        writer.add_scalar('Epoch training loss', loss, e + 1)
        writer.add_scalar('Epoch training pearson', pearson, e + 1)
        writer.add_scalar('Epoch training spearman', spearman, e + 1)
        writer.add_scalar('Epoch training kappa', kappa, e + 1)
    
# Update metrics in output CSV file
def update_csv(outfile, e, loss, pearson, spearman, kappa):
    with open(outfile, 'a') as outfile:
        wr = csv.writer(outfile, quoting=csv.QUOTE_ALL)
        wr.writerow([e, loss, pearson, spearman, kappa])
        
# Save Torch file with model and optimizer
def update_saved_model(metrics, model, optimizer, e, outfile):

    outfile_model_kappa = os.path.join(outfile, "checkpoint_kappa")
    outfile_model_rmse = os.path.join(outfile, "checkpoint_rmse")
    
    if metrics["valid"]["kappa"][-1] == max(metrics["valid"]["kappa"]):
        torch.save({
            'epoch': e + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, outfile_model_kappa + '.pth.tar')  # + " kappa_" + "{:.5f}".format(metrics["valid"]["kappa"][-1]) + " epoch_" + str(e + 1) + '.pth.tar')
        print('|\tBest validation accuracy for kappa: Model Saved!')
        
    if metrics["valid"]["rmse"][-1] == min(metrics["valid"]["rmse"]):
        torch.save({
            'epoch': e + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, outfile_model_rmse + '.pth.tar')  # + " loss_" + "{:.5f}".format(metrics["valid"]["rmse"][-1]) + " epoch_" + str(e + 1) + '.pth.tar')
        print('|\tBest validation accuracy for RMSE: Model Saved!')

# Update pickle containing metrics
def update_metrics_pickle(metrics, outfile_metrics):
    with open(outfile_metrics, 'wb') as outfile:
        pickle.dump(metrics, outfile)