from gru4rec_pytorch import SessionDataIterator
import torch

@torch.no_grad()
def batch_eval(gru, test_data, cutoff=[20], batch_size=512, mode='conservative', item_key='ItemId', session_key='SessionId', time_key='Time', total_items=0, new_items_set=set()):
    if gru.error_during_train: 
        raise Exception('Attempting to evaluate a model that wasn\'t trained properly (error_during_train=True)')
    
    # Assertions to catch initialization issues
    assert total_items > 0, "Error: total_items is zero. Ensure the dataset preprocessing is correct and total_items is initialized."
    assert len(new_items_set) > 0, "Error: new_items_set is empty. Ensure it contains new or unseen item IDs."

    # Initialize metrics
    recall = dict()
    mrr = dict()
    coverage = dict()
    new_coverage = dict()
    unique_items = set()
    new_item_hits = 0

    for c in cutoff:
        recall[c] = 0
        mrr[c] = 0
        coverage[c] = 0
        new_coverage[c] = 0

    H = []
    for i in range(len(gru.layers)):
        H.append(torch.zeros((batch_size, gru.layers[i]), requires_grad=False, device=gru.device, dtype=torch.float32))
    
    n = 0
    reset_hook = lambda n_valid, finished_mask, valid_mask: gru._adjust_hidden(n_valid, finished_mask, valid_mask, H)
    data_iterator = SessionDataIterator(test_data, batch_size, 0, 0, 0, item_key, session_key, time_key, device=gru.device, itemidmap=gru.data_iterator.itemidmap)

    for in_idxs, out_idxs in data_iterator(enable_neg_samples=False, reset_hook=reset_hook):
        for h in H: 
            h.detach_()
        O = gru.model.forward(in_idxs, H, None, training=False)
        oscores = O.T
        tscores = torch.diag(oscores[out_idxs])
        
        if mode == 'standard': 
            ranks = (oscores > tscores).sum(dim=0) + 1
        elif mode == 'conservative': 
            ranks = (oscores >= tscores).sum(dim=0)
        elif mode == 'median':  
            ranks = (oscores > tscores).sum(dim=0) + 0.5 * ((oscores == tscores).sum(dim=0) - 1) + 1
        else: 
            raise NotImplementedError

        # Track coverage metrics
        for c in cutoff:
            # Top-k recommendations
            top_k_items = oscores.topk(c, dim=0)[1].cpu().numpy().flatten()
            
            # Update unique items
            unique_items.update(top_k_items)
            
            # Count new items in the recommendations
            new_item_hits += sum(1 for item in top_k_items if item in new_items_set)
            
            # Update recall and mrr
            recall[c] += (ranks <= c).sum().cpu().numpy()
            mrr[c] += ((ranks <= c) / ranks.float()).sum().cpu().numpy()

        n += O.shape[0]

    # Finalize recall and mrr
    for c in cutoff:
        recall[c] /= n
        mrr[c] /= n

    # Compute coverage metrics
    for c in cutoff:
        coverage[c] = len(unique_items) / total_items
        new_coverage[c] = new_item_hits / n

    return recall, mrr, coverage, new_coverage
