def train_seq(model, Xseq, y, epochs=15, bs=512):
    opt = torch.optim.Adam(model.parameters(), 0.001)
    model.train()
    Xt = torch.tensor(Xseq, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    _m1_restore(opt)
    for _ in range(_m1_start, epochs):
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), bs):
            idx = perm[i:i + bs]
            xb, yb = (Xt[idx].to(DEV), yt[idx].to(DEV))
            opt.zero_grad()
            F.cross_entropy(model(xb)['logits'], yb).backward()
            opt.step()
        _m1_callback(_ + 1, model, opt)
    model.eval()
    return model
